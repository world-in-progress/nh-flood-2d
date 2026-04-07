"""
Pipe network preprocessing: parse SWMM .inp, map nodes to 2D grid elements,
and write pipe.fdb with CSR topology tables.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import taichi as ti

import fastdb4py as fdb

from ..input.domain import DomainConfig
from ..input.pipe import PipeConfig
from ..schema.feature import IndexLike, Node, PipeTopo
from ..util.ti import init_taichi


# ─── data classes ──────────────────────────────────────────────────────────────

class _NodeInfo(NamedTuple):
    name: str
    x: float
    y: float
    is_outfall: bool


# ─── SWMM .inp parsing ─────────────────────────────────────────────────────────

def _parse_inp_nodes(inp_path: str) -> list[_NodeInfo]:
    """
    Parse [COORDINATES] and [OUTFALLS] sections of a SWMM .inp file.
    Returns a list of _NodeInfo in file order.
    """
    outfall_names: set[str] = set()
    in_outfalls = False
    with open(inp_path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith('[outfalls]'):
                in_outfalls = True
                continue
            if s.startswith('[') and in_outfalls:
                break
            if in_outfalls and s and not s.startswith(';'):
                parts = s.split()
                if parts:
                    outfall_names.add(parts[0])

    nodes: list[_NodeInfo] = []
    in_coords = False
    with open(inp_path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith('[coordinates]'):
                in_coords = True
                continue
            if s.startswith('[') and in_coords:
                break
            if in_coords and s and not s.startswith(';'):
                parts = s.split()
                if len(parts) >= 3:
                    name = parts[0]
                    nodes.append(_NodeInfo(
                        name=name,
                        x=float(parts[1]),
                        y=float(parts[2]),
                        is_outfall=(name in outfall_names),
                    ))
    return nodes


# ─── Taichi nearest-neighbour kernel ───────────────────────────────────────────

@ti.kernel
def _find_nearest_kernel(
    n_elements: ti.i32,
    nx: ti.template(),   # element x coords, shape (n_elements,)
    ny: ti.template(),   # element y coords, shape (n_elements,)
    nz: ti.template(),   # element z (ground elev), shape (n_elements,)
    esl: ti.template(),  # element half-size, shape (n_elements,)
    n_nodes: ti.i32,
    node_x: ti.template(),
    node_y: ti.template(),
    dist_thresh: ti.f32,
    # outputs
    primary_ei: ti.template(),   # shape (n_nodes,)
):
    """
    For each node find the 2D element whose bbox contains the node (primary).
    Falls back to nearest-centroid within dist_thresh if none contains it.
    primary_ei[i] == 0 means no valid element found.
    """
    for i in range(n_nodes):
        px = node_x[i]
        py = node_y[i]
        best_ei = 0
        best_dist = ti.f32(1e18)
        for k in range(1, n_elements):
            half = esl[k] * 0.5
            cx = nx[k];  cy = ny[k]
            if px >= cx - half and px <= cx + half and py >= cy - half and py <= cy + half:
                best_ei = k
                best_dist = 0.0
                break  # bbox hit: take it
        if best_ei == 0:
            for k in range(1, n_elements):
                dx = nx[k] - px;  dy = ny[k] - py
                d = ti.sqrt(dx*dx + dy*dy)
                if d < best_dist:
                    best_dist = d
                    best_ei = k
        primary_ei[i] = best_ei if best_dist <= dist_thresh else 0


# ─── secondary (weak) topology: all elements within dist_thresh of each node ──

def _build_weak_topo(
    node_list: list[_NodeInfo],
    ex: np.ndarray,
    ey: np.ndarray,
    esl_arr: np.ndarray,
    primary_ei_arr: np.ndarray,
    dist_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (topo_ei_flat, topo_ptr) where:
      - topo_ptr is 0-based CSR offsets, length n_nodes+1
      - topo_ei_flat contains secondary element indices (primary excluded, outfall excluded)
    """
    n_nodes = len(node_list)
    topo_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    topo_rows: list[list[int]] = []

    for i, node in enumerate(node_list):
        row: list[int] = []
        if not node.is_outfall:
            p_ei = int(primary_ei_arr[i])
            for k in range(1, len(ex)):
                if k == p_ei:
                    continue
                dx = ex[k] - node.x
                dy = ey[k] - node.y
                half = float(esl_arr[k]) * 0.5
                # use bbox + distance fallback
                if abs(dx) <= half and abs(dy) <= half:
                    row.append(k)
                elif (dx*dx + dy*dy) <= dist_thresh**2:
                    row.append(k)
        topo_rows.append(row)
        topo_ptr[i + 1] = topo_ptr[i] + len(row)

    topo_ei_flat = np.array(
        [ei for row in topo_rows for ei in row], dtype=np.uint32
    )
    return topo_ei_flat, topo_ptr


# ─── main entry ────────────────────────────────────────────────────────────────

def prepare_pipe(
    pipe_cfg: PipeConfig,
    domain_cfg: DomainConfig,
) -> None:
    """
    Build pipe.fdb from SWMM .inp + ne.fdb.

    Tables written:
      Node[Node]                   – node info (index, name, x, y, is_outfall)
      IndexLike['node_primary_ei'] – primary element per node (1-based ei, 0=none)
      IndexLike['node_count_per_ei'] – how many nodes share each element (for flow split)
      PipeTopo[PipeTopo]           – flat secondary ei list (CSR data)
      IndexLike['topo_ptr']        – 0-based CSR offsets, length n_nodes+1
      IndexLike['node_count']      – single entry: total node count
    """
    init_taichi()

    # ── load ne.fdb ───────────────────────────────────────────────────────────
    from ..schema.feature import Ne, Ns
    ne_db = fdb.ORM.load(domain_cfg.ne_fdb, from_file=True)
    nes   = ne_db[Ne][Ne]
    ex_np  = nes.column.x.copy()
    ey_np  = nes.column.y.copy()
    e_num  = len(ex_np)  # includes virtual slot 0

    # compute esl (side length) per element:
    # matches solver_compact.py L232-233: esl[ei] = (ex[ei] - sx[isl_data[isl_ptr_l[ei]]]) * 2
    ns_db = fdb.ORM.load(domain_cfg.ns_fdb, from_file=True)
    sx_np = ns_db[Ns][Ns].column.x.copy()

    isl_data  = ne_db[IndexLike]['isl_data'].column.index.copy()
    isl_ptr_l = ne_db[IndexLike]['isl_ptr_l'].column.index.copy()

    esl_np = np.zeros(e_num, dtype=np.float32)
    for ei in range(1, e_num):
        lsi0 = int(isl_data[int(isl_ptr_l[ei])])
        esl_np[ei] = max((float(ex_np[ei]) - float(sx_np[lsi0])) * 2.0, 0.0001)

    del ne_db, ns_db

    # ── parse SWMM .inp ───────────────────────────────────────────────────────
    node_list = _parse_inp_nodes(pipe_cfg.inp)
    n_nodes   = len(node_list)
    if n_nodes == 0:
        raise ValueError(f'No nodes found in SWMM .inp: {pipe_cfg.inp}')

    # ── GPU nearest-neighbour: find primary_ei per node ───────────────────────
    weak_dist_thresh = pipe_cfg.weak_dist_thresh
    nx_field    = ti.field(dtype=ti.f32, shape=e_num)
    ny_field    = ti.field(dtype=ti.f32, shape=e_num)
    nz_field    = ti.field(dtype=ti.f32, shape=e_num)
    esl_field   = ti.field(dtype=ti.f32, shape=e_num)
    node_x_f    = ti.field(dtype=ti.f32, shape=n_nodes)
    node_y_f    = ti.field(dtype=ti.f32, shape=n_nodes)
    pei_field   = ti.field(dtype=ti.i32, shape=n_nodes)

    nx_field.from_numpy(ex_np)
    ny_field.from_numpy(ey_np)
    nz_field.from_numpy(np.zeros(e_num, dtype=np.float32))
    esl_field.from_numpy(esl_np)
    node_x_f.from_numpy(np.array([n.x for n in node_list], dtype=np.float32))
    node_y_f.from_numpy(np.array([n.y for n in node_list], dtype=np.float32))

    _find_nearest_kernel(
        e_num, nx_field, ny_field, nz_field, esl_field,
        n_nodes, node_x_f, node_y_f,
        float(weak_dist_thresh * 2),
        pei_field,
    )
    primary_ei_arr = pei_field.to_numpy()  # shape (n_nodes,), 1-based ei

    # ── build weak (secondary) CSR topology ───────────────────────────────────
    topo_ei_flat, topo_ptr = _build_weak_topo(
        node_list, ex_np, ey_np, esl_np, primary_ei_arr, weak_dist_thresh
    )

    # ── node_count_per_ei ─────────────────────────────────────────────────────
    nc_per_ei = np.zeros(e_num, dtype=np.uint32)
    for i in range(n_nodes):
        ei = int(primary_ei_arr[i])
        if ei > 0:
            nc_per_ei[ei] += 1
    # also count secondary mappings
    for ei in topo_ei_flat:
        if ei > 0:
            nc_per_ei[int(ei)] += 1

    # ── write pipe.fdb ────────────────────────────────────────────────────────
    # Node has STR field → fdb.ORM.truncate() doesn't support it.
    # Use create()+push() for all tables.

    Path(pipe_cfg.pipe_fdb).parent.mkdir(parents=True, exist_ok=True)
    db = fdb.ORM.create()

    # Node table
    for i, node in enumerate(node_list):
        row            = Node()
        row.index      = i
        row.name       = node.name
        row.x          = node.x
        row.y          = node.y
        row.is_outfall = node.is_outfall
        db.push(row)

    # node_primary_ei
    for i in range(n_nodes):
        il = IndexLike()
        il.index = int(primary_ei_arr[i])
        db.push(il, 'node_primary_ei')

    # node_count_per_ei (one entry per 2D element)
    for i in range(e_num):
        il = IndexLike()
        il.index = int(nc_per_ei[i])
        db.push(il, 'node_count_per_ei')

    # PipeTopo (CSR data, secondary element indices)
    n_topo = len(topo_ei_flat)
    if n_topo == 0:
        pt = PipeTopo()
        pt.ei = 0
        db.push(pt)
    else:
        for j in range(n_topo):
            pt = PipeTopo()
            pt.ei = int(topo_ei_flat[j])
            db.push(pt)

    # topo_ptr (0-based CSR offsets, length n_nodes+1)
    for i in range(n_nodes + 1):
        il = IndexLike()
        il.index = int(topo_ptr[i])
        db.push(il, 'topo_ptr')

    # node_count (single entry)
    il = IndexLike()
    il.index = n_nodes
    db.push(il, 'node_count')

    db.save(pipe_cfg.pipe_fdb)
    db.unlink()
    print(f'[prepare_pipe] wrote {n_nodes} nodes, {n_topo} secondary topo entries → {pipe_cfg.pipe_fdb}')
