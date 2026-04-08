"""
2D ↔ 1D exchange protocol (async / lagged coupling).

Provides four composable phases so the 2D process can overlap stepping
with 1D computation:

  compute_drainage()  — GPU→CPU + orifice drainage per node
  send_to_1d()        — push drainage dict to 1D (non-blocking)
  receive_from_1d()   — wait for 1D flood-return data
  apply_sources()     — merge drainage + flood-return → ssq_t

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

    Returns (data_dict, q_drain) where data_dict is sent to 1D and
    q_drain is a numpy array of per-element source/sink rates.
    """
    pi = math.pi
    ci = window_dt

    h_np = h_t.to_numpy()
    z_np = ez_t.to_numpy()
    esl_np = esl_t.to_numpy()

    data_dict: dict = {}
    q_drain = np.zeros(len(h_np), dtype=np.float32)

    for i, name in enumerate(node_names):
        ei = int(primary_ei[i])
        if ei == 0 or node_is_outfall[i]:
            level = float(h_np[ei]) if ei != 0 else 0.0
            data_dict[name] = {'level': level, 'flow': 0.0}
            continue
        depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
        area = float(esl_np[ei]) ** 2
        nc = max(int(nc_per_ei[ei]), 1)
        q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
        q_drain[ei] -= q
        data_dict[name] = {'level': depth, 'flow': float(q)}

    for i, name in enumerate(node_names):
        if node_is_outfall[i] or int(primary_ei[i]) == 0:
            continue
        for j in range(int(topo_ptr[i]), int(topo_ptr[i + 1])):
            ei = int(topo_ei[j])
            if ei == 0:
                continue
            depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
            area = float(esl_np[ei]) ** 2
            nc = max(int(nc_per_ei[ei]), 1)
            q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
            q_drain[ei] -= q
            data_dict[name]['flow'] += float(q)

    return data_dict, q_drain


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
    q_drain: np.ndarray,
    flood_return: dict,
    ssq_t,
    primary_ei: np.ndarray,
    name_to_idx: dict,
) -> None:
    """Merge drainage + flood-return into ssq_t Taichi field."""
    ssq_np = np.zeros(len(q_drain), dtype=np.float32)
    ssq_np += q_drain

    for name, d in flood_return.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        ssq_np[ei] += float(d.get('flow', 0.0))

    ssq_t.from_numpy(ssq_np)
