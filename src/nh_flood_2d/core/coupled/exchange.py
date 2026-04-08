"""
2D ↔ 1D exchange protocol (async / lagged coupling).

Provides four composable phases so the 2D process can overlap stepping
with 1D computation:

  compute_drainage()  — GPU→CPU + orifice drainage per node
  send_to_1d()        — push drainage dict to 1D (non-blocking)
  receive_from_1d()   — wait for 1D flood-return data
  apply_sources()     — merge drainage + flood-return → ssq_t

Note: With stepping coupling, 1D return values are already in m³/s
(read directly from SWMM solver API), so no volume→rate conversion
is needed.

Async main-loop pattern (lag-1):
  At exchange point N:
    1. IF N>0: receive_from_1d(results of window N-1)
    2. compute_drainage(current state)
    3. apply_sources(fresh drain + lagged flood-return)
    4. send_to_1d(current drainage, non-blocking)
    5. continue 2D stepping immediately
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...input.pipe import PipeConfig
    from .timer import CouplingTimer


def compute_drainage(
    window_dt: float,
    h_t, ez_t, esl_t,
    primary_ei: np.ndarray,
    topo_ei: np.ndarray,
    topo_ptr: np.ndarray,
    nc_per_ei: np.ndarray,
    node_names: list,
    node_is_outfall: np.ndarray,
) -> tuple:
    """Compute surface drainage into pipe nodes (GPU→CPU).

    Matches reference (Flood_new801.py L726-761):
      1. Secondary cells first — SET (=) semantics, no /nc, all nodes
      2. Primary cells second — accumulate (+=), /nc, node_type gating

    Returns (data_dict, q_source) where data_dict is sent to 1D and
    q_source is the per-element source/sink array (drainage part only;
    1D return is added later by apply_sources).
    """
    pi = math.pi
    ci = window_dt
    n_nodes = len(node_names)

    h_np = h_t.to_numpy()
    z_np = ez_t.to_numpy()
    esl_np = esl_t.to_numpy()

    q_source = np.zeros(len(h_np), dtype=np.float32)
    total_flow = np.zeros(n_nodes, dtype=np.float32)

    # Step 1: secondary cells — SET (=), no /nc, ALL nodes incl. outfalls
    for i in range(n_nodes):
        for j in range(int(topo_ptr[i]), int(topo_ptr[i + 1])):
            ei = int(topo_ei[j])
            if ei == 0:
                continue
            depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
            area = float(esl_np[ei]) ** 2
            q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci)
            q_source[ei] = -q
            total_flow[i] += q

    # Step 2: primary cells — accumulate (+=), /nc, node_type gating
    data_dict: dict = {}
    for i, name in enumerate(node_names):
        ei = int(primary_ei[i])
        if ei == 0:
            data_dict[name] = {'level': 0.0, 'flow': 0.0}
            continue

        depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
        area = float(esl_np[ei]) ** 2
        nc = max(int(nc_per_ei[ei]), 1)
        node_type = 0 if node_is_outfall[i] else 1

        # outfall level = WSE (h), junction level = depth
        level = float(h_np[ei]) if node_is_outfall[i] else depth

        q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
        q_source[ei] += -(q * node_type)

        primary_drain = q * node_type
        data_dict[name] = {
            'level': level,
            'flow': float(primary_drain + total_flow[i]),
        }

    return data_dict, q_source


def send_to_1d(shared, data_dict: dict, window_dt: float) -> None:
    """Push drainage data to the 1D process (non-blocking)."""
    data_dict['__window_dt__'] = {'level': float(window_dt), 'flow': 0.0}
    with shared['lock']:
        shared['2d_data'].clear()
        shared['2d_data'].update(data_dict)
        shared['2d_ready'].set()
        shared['1d_ready'].clear()


def receive_from_1d(
    shared,
    pipe_cfg: PipeConfig,
    timer: CouplingTimer | None = None,
) -> dict:
    """Wait for and receive 1D flood-return data.

    Returns a dict {node_name: {'level': ..., 'flow': ...}}.
    Updates *timer* with 1D-wait and 1D-step durations.
    """
    t_wait_start = time.perf_counter()

    if shared['stop'].is_set():
        return {}
    if not shared['1d_ready'].wait(timeout=pipe_cfg.exchange_timeout):
        raise TimeoutError('Timeout waiting for 1D pipe data')

    with shared['lock']:
        pipe_data = dict(shared['1d_data'])
        shared['1d_ready'].clear()

    t_wait = time.perf_counter() - t_wait_start

    t_1d = 0.0
    if '__1d_elapsed__' in pipe_data:
        t_1d = float(pipe_data.pop('__1d_elapsed__'))
    pipe_data.pop('__window_dt__', None)

    if timer is not None:
        timer.total_1d_wait += t_wait
        timer.total_1d_step += t_1d

    return pipe_data


def apply_sources(
    q_source: np.ndarray,
    flood_return: dict,
    ssq_t,
    primary_ei: np.ndarray,
    name_to_idx: dict,
) -> None:
    """Add 1D flood-return to q_source and write to ssq_t.

    q_source already contains drainage terms from compute_drainage().
    This adds 1D return flows at primary cell locations, matching
    reference: q_source[grid] += flows_np[i] (at primary cells).
    """
    for name, d in flood_return.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        q_source[ei] += float(d.get('flow', 0.0))

    q_source[0] = 0.0
    ssq_t.from_numpy(q_source)
