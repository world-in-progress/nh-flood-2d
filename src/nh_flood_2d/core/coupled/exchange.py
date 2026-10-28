"""Conservative surface/pipe exchange for one coupling window.

All rates use m³/s. A positive source adds water to the surface; the
corresponding negative rate is applied to a SWMM junction in the same window.
"""

from __future__ import annotations

import math

import numpy as np


ORIFICE_COEFFICIENT = 0.85 * math.pi * 0.8
MAX_OVERFLOW_HEAD_M = 3.0
OVERFLOW_RELAXATION = 0.5


def compute_exchange(
    window_dt: float,
    depth: np.ndarray,
    bed: np.ndarray,
    area: np.ndarray,
    land_types: np.ndarray,
    node_names: list[str],
    is_outfall: np.ndarray,
    primary_ei: np.ndarray,
    topo_ei: np.ndarray,
    topo_ptr: np.ndarray,
    rims: np.ndarray,
    heads: dict[str, float],
) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, float], tuple[tuple[int, ...], ...]]:
    """Return F32 surface source, SWMM boundaries, diagnostics, and cell links.

    Topology and mesh shapes are validated once by PipeCoupling.
    An eligible cell has one or more unique junction relationships. Its total
    drainage is bounded by water present at the beginning of this window,
    then divided equally across those junctions. Outfalls receive only stage.
    Overflow uses the historical head cap and relaxation and enters the same
    junction as a negative inflow, exactly balancing the surface source.
    """
    depth = np.asarray(depth, dtype=np.float64)
    bed = np.asarray(bed, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    n_cells, n_nodes = len(depth), len(node_names)

    relationships: list[set[int]] = [set() for _ in range(n_cells)]
    for node in range(n_nodes):
        if is_outfall[node]:
            continue
        cells = [int(primary_ei[node])]
        cells.extend(int(ei) for ei in topo_ei[int(topo_ptr[node]):int(topo_ptr[node + 1])])
        for ei in set(cells):
            if ei and land_types[ei] != 8:
                relationships[ei].add(node)

    source = np.zeros(n_cells, dtype=np.float64)
    node_flow = np.zeros(n_nodes, dtype=np.float64)
    drainage = 0.0
    for ei in range(1, n_cells):
        attached = relationships[ei]
        if not attached:
            continue
        h = float(depth[ei])
        q = min(ORIFICE_COEFFICIENT * h ** 1.5, h * float(area[ei]) / window_dt)
        source[ei] -= q
        drainage += q
        for node in attached:
            node_flow[node] += q / len(attached)

    overflow = 0.0
    active_overflow_nodes = 0
    primary_counts = np.zeros(n_cells, dtype=np.int32)
    for node in range(n_nodes):
        if not is_outfall[node] and int(primary_ei[node]) > 0:
            primary_counts[int(primary_ei[node])] += 1
    for node, name in enumerate(node_names):
        ei = int(primary_ei[node])
        if is_outfall[node] or ei == 0 or land_types[ei] == 8:
            continue
        h = float(depth[ei])
        delta_h = float(heads[name]) - (float(rims[node]) + h)
        if delta_h <= 0:
            continue
        dh_eff = min(delta_h, MAX_OVERFLOW_HEAD_M)
        q = min(
            ORIFICE_COEFFICIENT * dh_eff ** 1.5,
            dh_eff * float(area[ei]) / window_dt / int(primary_counts[ei]),
        ) * OVERFLOW_RELAXATION
        source[ei] += q
        node_flow[node] -= q
        overflow += q
        active_overflow_nodes += 1

    # The device receives F32 source rates. Pair even that final rounding with
    # SWMM, keeping the original drainage and overflow formulas unchanged.
    source_f32 = source.astype(np.float32)
    for ei in range(1, n_cells):
        attached = relationships[ei]
        if attached:
            correction = (float(source_f32[ei]) - source[ei]) / len(attached)
            for node in attached:
                node_flow[node] -= correction

    boundaries = {}
    for node, name in enumerate(node_names):
        ei = int(primary_ei[node])
        stage = float(depth[ei] + bed[ei]) if ei else 0.0
        boundaries[name] = {
            'level': stage if is_outfall[node] else 0.0,
            'flow': 0.0 if is_outfall[node] else float(node_flow[node]),
        }
    source[0] = 0.0
    diagnostics = {
        'drainage_m3_s': drainage,
        'overflow_m3_s': overflow,
        'net_pipe_source_m3_s': float(source_f32.sum(dtype=np.float64)),
        'net_swmm_inflow_m3_s': float(node_flow.sum()),
        'overflow_nodes': active_overflow_nodes,
        'surface_pipe_balance_m3_s': float(source_f32.sum(dtype=np.float64) + node_flow.sum()),
    }
    return source_f32, boundaries, diagnostics, tuple(tuple(sorted(nodes)) for nodes in relationships)
