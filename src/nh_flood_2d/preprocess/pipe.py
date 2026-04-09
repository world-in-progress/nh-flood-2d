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
from ..schema.feature import IndexLike, F32Value, Node, PipeTopo
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


def _parse_junction_rims(inp_path: str) -> dict[str, float]:
    """Parse [JUNCTIONS] to compute rim elevation per junction.

    Format: Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
    rim = Elevation + MaxDepth (ground level at junction opening).
    """
    rims: dict[str, float] = {}
    in_junctions = False
    with open(inp_path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith('[junctions]'):
                in_junctions = True
                continue
            if s.startswith('[') and in_junctions:
                break
            if in_junctions and s and not s.startswith(';'):
                parts = s.split()
                if len(parts) >= 3:
                    name = parts[0]
                    elevation = float(parts[1])
                    max_depth = float(parts[2])
                    rims[name] = elevation + max_depth
    return rims


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
    topo_rows: list[np.ndarray] = []

    # Pre-slice arrays (skip virtual element 0)
    ex1 = ex[1:]
    ey1 = ey[1:]
    half1 = esl_arr[1:] * 0.5
    dist_sq = dist_thresh ** 2

    for i, node in enumerate(node_list):
        if node.is_outfall:
            topo_ptr[i + 1] = topo_ptr[i]
            continue
        dx = ex1 - node.x
        dy = ey1 - node.y
        mask = ((np.abs(dx) <= half1) & (np.abs(dy) <= half1)) | ((dx * dx + dy * dy) <= dist_sq)
        p_ei = int(primary_ei_arr[i])
        if p_ei > 0:
            mask[p_ei - 1] = False
        row = np.where(mask)[0] + 1  # +1 to restore 1-based ei
        topo_rows.append(row)
        topo_ptr[i + 1] = topo_ptr[i] + len(row)

    topo_ei_flat = np.concatenate(topo_rows) if topo_rows else np.array([], dtype=np.uint32)
    return topo_ei_flat.astype(np.uint32), topo_ptr


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
    print('[prepare_pipe] loading NE/NS ...', flush=True)
    from ..schema.feature import Ne, Ns
    ne_db = fdb.ORM.load(domain_cfg.ne_fdb, from_file=True)
    nes   = ne_db[Ne][Ne]
    ex_np  = nes.column.x.copy()
    ey_np  = nes.column.y.copy()
    e_num  = len(ex_np)  # includes virtual slot 0

    # compute esl (side length) per element (vectorised):
    # matches solver_compact.py L232-233: esl[ei] = (ex[ei] - sx[isl_data[isl_ptr_l[ei]]]) * 2
    ns_db = fdb.ORM.load(domain_cfg.ns_fdb, from_file=True)
    sx_np = ns_db[Ns][Ns].column.x.copy()

    isl_data  = ne_db[IndexLike]['isl_data'].column.index.copy()
    isl_ptr_l = ne_db[IndexLike]['isl_ptr_l'].column.index.copy()

    lsi0_indices = isl_data[isl_ptr_l[1:e_num].astype(np.intp)]
    esl_np = np.zeros(e_num, dtype=np.float32)
    esl_np[1:] = np.maximum((ex_np[1:] - sx_np[lsi0_indices.astype(np.intp)]) * 2.0, 0.0001)

    del ne_db, ns_db
    print(f'[prepare_pipe] esl computed, e_num={e_num}', flush=True)

    # ── parse SWMM .inp ───────────────────────────────────────────────────────
    node_list = _parse_inp_nodes(pipe_cfg.inp)
    n_nodes   = len(node_list)
    if n_nodes == 0:
        raise ValueError(f'No nodes found in SWMM .inp: {pipe_cfg.inp}')
    print(f'[prepare_pipe] parsed {n_nodes} nodes from .inp', flush=True)

    # ── parse junction rim elevations ─────────────────────────────────────────
    junction_rims = _parse_junction_rims(pipe_cfg.inp)
    print(f'[prepare_pipe] parsed {len(junction_rims)} junction rims from .inp', flush=True)

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

    print('[prepare_pipe] running GPU nearest-neighbour kernel ...', flush=True)
    _find_nearest_kernel(
        e_num, nx_field, ny_field, nz_field, esl_field,
        n_nodes, node_x_f, node_y_f,
        float(weak_dist_thresh * 2),
        pei_field,
    )
    primary_ei_arr = pei_field.to_numpy()  # shape (n_nodes,), 1-based ei
    print('[prepare_pipe] nearest-neighbour done', flush=True)

    # ── build weak (secondary) CSR topology ───────────────────────────────────
    topo_ei_flat, topo_ptr = _build_weak_topo(
        node_list, ex_np, ey_np, esl_np, primary_ei_arr, weak_dist_thresh
    )
    print(f'[prepare_pipe] weak topo built, {len(topo_ei_flat)} secondary entries', flush=True)

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
    print('[prepare_pipe] writing pipe.fdb ...', flush=True)

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

    # node_rim — junction rim elevation per node (outfalls get 0.0)
    for i, node in enumerate(node_list):
        fv = F32Value()
        fv.value = junction_rims.get(node.name, 0.0) if not node.is_outfall else 0.0
        db.push(fv, 'node_rim')

    db.save(pipe_cfg.pipe_fdb)
    db.unlink()
    print(f'[prepare_pipe] wrote {n_nodes} nodes, {n_topo} secondary topo entries → {pipe_cfg.pipe_fdb}')
