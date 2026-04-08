"""
2D ↔ 1D exchange protocol (pressure coupling, async lag-1).

Coupling model: pressure coupling (NOT drainage).
  - 2D → 1D: water levels at node locations (pressure boundary)
  - 1D → 2D: overflow / outfall discharge (flow injection)
  - Pipe INFLOWS: from .inp file definition (not overridden)

Provides four composable phases so the 2D process can overlap stepping
with 1D computation:

  compute_node_levels()  — GPU→CPU water levels at pipe node locations
  send_to_1d()           — push level dict to 1D (non-blocking)
  receive_from_1d()      — wait for 1D flood-return data
  apply_sources()        — flood-return → ssq_t

Async main-loop pattern (lag-1):
  At exchange point N:
    1. IF N>0: receive_from_1d(results of window N-1)
    2. compute_node_levels(current state)
    3. apply_sources(lagged flood-return)
    4. send_to_1d(current levels, non-blocking)
    5. continue 2D stepping immediately
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...input.pipe import PipeConfig
    from .timer import CouplingTimer


def compute_node_levels(
    h_t, ez_t,
    primary_ei: np.ndarray,
    node_names: list,
    node_is_outfall: np.ndarray,
) -> dict:
    """Extract 2D water levels at pipe node locations for pressure coupling.

    For junctions: level = water depth above ground (for surcharge depth).
    For outfalls:  level = water surface elevation (for backwater).
    """
    h_np = h_t.to_numpy()
    z_np = ez_t.to_numpy()

    data_dict: dict = {}
    for i, name in enumerate(node_names):
        ei = int(primary_ei[i])
        if ei == 0 or node_is_outfall[i]:
            level = float(h_np[ei]) if ei != 0 else 0.0
            data_dict[name] = {'level': level}
            continue
        depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
        data_dict[name] = {'level': depth}

    return data_dict


def send_to_1d(shared, data_dict: dict, window_dt: float) -> None:
    """Push water-level data to the 1D process (non-blocking)."""
    data_dict['__window_dt__'] = {'level': float(window_dt)}
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
    flood_return: dict,
    ssq_t,
    primary_ei: np.ndarray,
    name_to_idx: dict,
    ne_count: int,
) -> None:
    """Apply 1D flood-return / outfall flows as source terms in ssq_t."""
    ssq_np = np.zeros(ne_count, dtype=np.float32)

    for name, d in flood_return.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        ssq_np[ei] += float(d.get('flow', 0.0))

    ssq_t.from_numpy(ssq_np)
