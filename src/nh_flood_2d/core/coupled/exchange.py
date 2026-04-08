"""
2D ↔ 1D exchange protocol.

Called from the 2D process each coupling window to:
  1. Compute surface drainage into pipe nodes
  2. Send water-level / flow data to the 1D process
  3. Wait for flood-return data from 1D
  4. Merge drainage + flood into the ssq Taichi field
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...input.pipe import PipeConfig
    from .timer import CouplingTimer


def exchange_with_1d(
    shared,
    pipe_cfg: PipeConfig,
    window_dt: float,
    h_t, ssq_t, ez_t, esl_t,
    primary_ei: np.ndarray,
    topo_ei: np.ndarray,
    topo_ptr: np.ndarray,
    nc_per_ei: np.ndarray,
    node_names: list,
    node_is_outfall: np.ndarray,
    name_to_idx: dict,
    timer: CouplingTimer | None = None,
) -> None:
    t_exc_start = time.perf_counter()
    pi = math.pi
    ci = window_dt

    h_np = h_t.to_numpy()
    z_np = ez_t.to_numpy()
    esl_np = esl_t.to_numpy()

    data_dict: dict = {}
    q_drain = np.zeros(len(h_np), dtype=np.float32)

    # ── primary-cell drainage (skip outfalls) ──────────────────────────────
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

    # ── secondary-cell drainage (CSR) ──────────────────────────────────────
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

    # ── send to 1D ─────────────────────────────────────────────────────────
    data_dict['__window_dt__'] = {'level': float(window_dt), 'flow': 0.0}
    with shared['lock']:
        shared['2d_data'].clear()
        shared['2d_data'].update(data_dict)
        shared['2d_ready'].set()
        shared['1d_ready'].clear()

    # ── wait for 1D flood return ───────────────────────────────────────────
    t_wait_start = time.perf_counter()

    if shared['stop'].is_set():
        return
    if not shared['1d_ready'].wait(timeout=pipe_cfg.exchange_timeout):
        raise TimeoutError('Timeout waiting for 1D pipe data')

    with shared['lock']:
        pipe_data = dict(shared['1d_data'])
        shared['1d_ready'].clear()

    t_wait_end = time.perf_counter()

    t_1d_elapsed = 0.0
    if '__1d_elapsed__' in pipe_data:
        t_1d_elapsed = float(pipe_data.pop('__1d_elapsed__'))

    # ── merge drain + flood → ssq_t ───────────────────────────────────────
    ssq_np = np.zeros(len(h_np), dtype=np.float32)
    ssq_np += q_drain

    for name, d in pipe_data.items():
        if name == '__window_dt__':
            continue
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        ssq_np[ei] += float(d.get('flow', 0.0))

    ssq_t.from_numpy(ssq_np)

    if timer is not None:
        t_total = time.perf_counter() - t_exc_start
        t_wait = t_wait_end - t_wait_start
        timer.total_exchange_overhead += t_total - t_wait
        timer.total_1d_wait += t_wait
        timer.total_1d_step += t_1d_elapsed
        timer.exchange_count += 1
