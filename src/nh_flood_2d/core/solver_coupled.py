"""
Coupled 2D surface-water / 1D pipe-network solver.

Architecture:
  solver_coupled()          – entry point; spawns two subprocesses
    _run_2d()               – GPU 2D hydrodynamic loop (mirrors solver_compact)
    _run_1d_pipe()          – CPU SWMM pipe-network loop
    _exchange_with_1d()     – called from _run_2d each coupling window

Shared-memory protocol (multiprocessing.Manager dicts/events):
  shared['2d_data']   dict  {node_name: {level, flow}, '__window_dt__': {level, flow}}
  shared['1d_data']   dict  {node_name: {level, flow}}
  shared['2d_ready']  Event set by 2D after writing 2d_data
  shared['1d_ready']  Event set by 1D after writing 1d_data
  shared['lock']      Lock  guards dict access
  shared['stop']      Event set by either side to signal shutdown

level semantics in 2d_data:
  junction node  →  water depth above DEM ground (h - z, m), used as surcharge head
  outfall node   →  absolute water-surface elevation (h, m), used as fixed-head BC
"""

import gc
import math
import shutil
import time
import multiprocessing
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import no_type_check

import taichi as ti
import fastdb4py as fdb

from ..input import DomainConfig, ForceConfig
from ..input.pipe import PipeConfig
from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import (
    Ne, Ns, IndexLike, SideTopoInfo, Rainfall, Tide, Gate,
    U8Value, UVH, Node, PipeTopo,
)


# ─── helpers ───────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class CouplingTimer:
    """Wall-clock timing statistics for the coupled solver."""

    def __init__(self):
        self.wall_start    = time.perf_counter()
        self.exchange_count = 0
        self.total_2d_step  = 0.0  # total wall time for 2D steps within exchange windows
        self.total_exchange = 0.0  # total wall time for exchange operations
        self.total_1d_step  = 0.0  # total wall time for 1D SWMM steps

    def report(self, prefix: str = '[timer]') -> None:
        elapsed = time.perf_counter() - self.wall_start
        n = max(self.exchange_count, 1)
        print(f'{prefix} ────────────── Coupling Statistics ──────────────')
        print(f'{prefix}   Exchange windows completed : {self.exchange_count}')
        print(f'{prefix}   Total wall time            : {elapsed:.1f} s')
        print(f'{prefix}   2D per exchange-step (avg)  : {self.total_2d_step / n:.3f} s')
        print(f'{prefix}   1D per exchange-step (avg)  : {self.total_1d_step / n:.3f} s')
        print(f'{prefix}   Exchange overhead (avg)     : {self.total_exchange / n:.3f} s')
        print(f'{prefix}   2D total                    : {self.total_2d_step:.1f} s')
        print(f'{prefix}   1D total                    : {self.total_1d_step:.1f} s')
        print(f'{prefix}   Exchange total              : {self.total_exchange:.1f} s')
        print(f'{prefix} ────────────────────────────────────────────────')


# ─── exchange function ──────────────────────────────────────────────────────────

def _exchange_with_1d(
    shared,
    pipe_cfg: PipeConfig,
    window_dt: float,           # actual exchange window length (s)
    h_t, ssq_t, ez_t, esl_t,   # Taichi fields
    primary_ei: np.ndarray,     # shape (n_nodes,), 1-based
    topo_ei: np.ndarray,        # CSR data (secondary only)
    topo_ptr: np.ndarray,       # 0-based CSR offsets, length n_nodes+1
    nc_per_ei: np.ndarray,      # per-element node count (for flow split)
    node_names: list,
    node_is_outfall: np.ndarray,
    name_to_idx: dict,
    timer: CouplingTimer | None = None,
) -> None:
    t_exc_start = time.perf_counter()
    pi = math.pi
    ci = window_dt  # use actual window, not nominal coupling_interval

    h_np   = h_t.to_numpy()
    z_np   = ez_t.to_numpy()
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
        area  = float(esl_np[ei]) ** 2
        nc    = max(int(nc_per_ei[ei]), 1)
        q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
        q_drain[ei] -= q
        data_dict[name] = {'level': depth, 'flow': float(q)}

    # ── secondary-cell drainage (CSR, outfalls and no-primary nodes skipped) ─
    # topo_ptr is 0-based; topo_ei values are 1-based; PipeTopo excludes primary_ei
    for i, name in enumerate(node_names):
        if node_is_outfall[i] or int(primary_ei[i]) == 0:
            continue
        for j in range(int(topo_ptr[i]), int(topo_ptr[i + 1])):
            ei = int(topo_ei[j])
            if ei == 0:
                continue
            depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
            area  = float(esl_np[ei]) ** 2
            nc    = max(int(nc_per_ei[ei]), 1)
            q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
            q_drain[ei] -= q
            data_dict[name]['flow'] += float(q)

    # ── send to 1D (include actual window_dt so 1D can compute V_f/window_dt) ─
    data_dict['__window_dt__'] = {'level': float(window_dt), 'flow': 0.0}
    with shared['lock']:
        shared['2d_data'].clear()
        shared['2d_data'].update(data_dict)
        shared['2d_ready'].set()
        shared['1d_ready'].clear()

    # ── wait for 1D flood return ────────────────────────────────────────────
    if shared['stop'].is_set():
        return
    if not shared['1d_ready'].wait(timeout=pipe_cfg.exchange_timeout):
        raise TimeoutError('Timeout waiting for 1D pipe data')

    with shared['lock']:
        pipe_data = dict(shared['1d_data'])
        shared['1d_ready'].clear()

    # Extract 1D timing if present
    t_1d_elapsed = 0.0
    if '__1d_elapsed__' in pipe_data:
        t_1d_elapsed = float(pipe_data.pop('__1d_elapsed__'))

    # ── merge drain + flood → ssq_t (rebuilt from zero each window) ────────
    ssq_np = np.zeros(len(h_np), dtype=np.float32)
    ssq_np += q_drain   # drain: negative (m³/s)

    for name, d in pipe_data.items():
        if name == '__window_dt__':
            continue
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        ssq_np[ei] += float(d.get('flow', 0.0))  # flood return: positive (m³/s)

    ssq_t.from_numpy(ssq_np)

    if timer is not None:
        timer.total_exchange += time.perf_counter() - t_exc_start
        timer.total_1d_step  += t_1d_elapsed
        timer.exchange_count += 1


# ─── Taichi helper ──────────────────────────────────────────────────────────────

@ti.func
@no_type_check
def horton_decay(initial: float, final: float, k: float, t: float) -> float:
    return final + (initial - final) * ti.exp(-k * t)


# ─── 2D subprocess ──────────────────────────────────────────────────────────────

def _run_2d(
    shared,
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig,
    start_time_step: int = 0,
) -> None:
    """2D GPU hydrodynamic loop; runs in a subprocess spawned by solver_coupled()."""
    try:
        init_taichi(use_gpu=True, profiler=False)

        ne_fdb       = fdb.ORM.load(str(Path(domain_cfg.ne_fdb)),       from_file=True)
        ns_fdb       = fdb.ORM.load(str(Path(domain_cfg.ns_fdb)),       from_file=True)
        tide_fdb     = fdb.ORM.load(str(Path(force_cfg.tide_fdb)),      from_file=True)
        gate_fdb     = fdb.ORM.load(str(Path(force_cfg.gate_fdb)),      from_file=True)
        rain_fdb     = fdb.ORM.load(str(Path(force_cfg.rain_fdb)),      from_file=True)
        boundary_fdb = fdb.ORM.load(str(Path(domain_cfg.boundary_fdb)), from_file=True)

        nes       = ne_fdb[Ne][Ne]
        nss       = ns_fdb[Ns][Ns]
        tides     = tide_fdb[Tide][Tide]
        gates     = gate_fdb[Gate][Gate]
        sbfs      = boundary_fdb[U8Value]['sbf']
        bdeis     = boundary_fdb[IndexLike]['bdei']
        rainfalls = rain_fdb[Rainfall][Rainfall]
        sts       = ns_fdb[SideTopoInfo][SideTopoInfo]

        tts: np.ndarray = tides.column.time.copy()
        tls: np.ndarray = tides.column.level.copy()
        rts: np.ndarray = rainfalls.column.time.copy()
        rqs: np.ndarray = rainfalls.column.quantity.copy()

        e_num = len(nes)
        s_num = len(nss)
        b_num = len(bdeis)
        t_num = len(tides)
        r_num = len(rainfalls)
        g_num = gate_fdb[IndexLike]['gate_count'][0].index

        n     = 0.033
        g     = 9.81
        afa   = domain_cfg.afa
        sita  = domain_cfg.sita
        min_h = domain_cfg.min_h

        esl_t   = ti.field(dtype=ti.f32, shape=e_num)
        eq_t    = ti.field(dtype=ti.f32, shape=(e_num, 4))
        enq_t   = ti.field(dtype=ti.f32, shape=(e_num, 4))
        ez_t    = copy_to_taichi(nes.column.z,    ti.f32, None)
        eu_t    = copy_to_taichi(nes.column.type, ti.u8,  None)
        bdei_t  = copy_to_taichi(bdeis.column.index, ti.i32, None)

        isl_data_t    = copy_to_taichi(ne_fdb[IndexLike]['isl_data'].column.index,  ti.i32, None)
        isl_ptr_l_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_l'].column.index, ti.i32, None)
        isl_ptr_r_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_r'].column.index, ti.i32, None)
        isl_ptr_b_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_b'].column.index, ti.i32, None)
        isl_ptr_top_t = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_t'].column.index, ti.i32, None)

        ndt_t   = ti.field(dtype=ti.f32, shape=())
        dh_t    = ti.field(dtype=ti.f32, shape=s_num)
        sq_t    = ti.field(dtype=ti.f32, shape=s_num)
        sqn_t   = ti.field(dtype=ti.f32, shape=s_num)
        sdc_t   = ti.field(dtype=ti.f32, shape=s_num)
        sl_t    = copy_to_taichi(nss.column.length,  ti.f32, None)
        sbf_t   = copy_to_taichi(sbfs.column.value,  ti.u8,  None)
        sts_t   = copy_to_taichi(sts.column.info,    ti.i32, [s_num, 3])
        gate_t  = copy_to_taichi(gates.column.info[:g_num * 100], ti.i32, [g_num, 100])

        u_t     = ti.field(dtype=ti.f32, shape=e_num)
        v_t     = ti.field(dtype=ti.f32, shape=e_num)
        h_t     = ti.field(dtype=ti.f32, shape=e_num)
        depth_t = ti.field(dtype=ti.f32, shape=e_num)
        ssq_t   = ti.field(dtype=ti.f32, shape=e_num)

        cumulative_rain_time_t = ti.field(dtype=ti.f32, shape=())
        fr1 = ti.field(dtype=ti.f32, shape=())
        fr2 = ti.field(dtype=ti.f32, shape=())
        fr3 = ti.field(dtype=ti.f32, shape=())
        fr4 = ti.field(dtype=ti.f32, shape=())
        fr5 = ti.field(dtype=ti.f32, shape=())
        fr6 = ti.field(dtype=ti.f32, shape=())
        fr7 = ti.field(dtype=ti.f32, shape=())

        current_time          = 0.0
        current_rain_idx      = 0
        current_tide_idx      = 0
        simulation_begin_time = 0.0

        gate_is_open = ti.field(dtype=ti.i32, shape=())

        @ti.kernel
        @no_type_check
        def init_gpu(
            ex_t: ti.template(), ey_t: ti.template(),
            isl_data: ti.template(), isl_ptr_l: ti.template(),
            sx_t: ti.template(),
        ):
            ndt_t[None] = 0.1
            cumulative_rain_time_t[None] = 0.0
            for ei in range(1, e_num):
                u_t[ei] = 0.0
                v_t[ei] = 0.0
                ssq_t[ei] = 0.0
                h_t[ei] = ez_t[ei] if ez_t[ei] > 0.0 else 0.0
                eq_t[ei, 0] = eq_t[ei, 1] = eq_t[ei, 2] = eq_t[ei, 3] = 0.0
                enq_t[ei, 0] = enq_t[ei, 1] = enq_t[ei, 2] = enq_t[ei, 3] = 0.0
                lsi0 = isl_data[isl_ptr_l[ei]]
                esl_t[ei] = ti.max((ex_t[ei] - sx_t[lsi0]) * 2.0, 0.0001)
                if eu_t[ei] == 8:
                    h_t[ei] = 2.0
            for si in range(1, s_num):
                sq_t[si] = 0.0
                sqn_t[si] = 0.0
                dh_t[si] = -999.0
                so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]
                if so == 2:
                    sdc_t[si] = ti.max(ti.abs(ex_t[eh] - ex_t[el]), 0.01)
                else:
                    sdc_t[si] = ti.max(ti.abs(ey_t[eh] - ey_t[el]), 0.01)
            fr1[None] = fr2[None] = fr3[None] = fr4[None] = fr5[None] = fr6[None] = fr7[None] = 0.0

        @ti.kernel
        @no_type_check
        def tick(tide: float, rainq: float) -> ti.f32:
            dt = ti.max(0.0001, ti.min(ndt_t[None], 1.0))
            ndt_t[None] = 1000.0
            if tide > 1.5 and gate_is_open[None] == 0:
                for i in range(1, e_num):
                    if eu_t[i] == 11:
                        ez_t[i] = 0.0
                gate_is_open[None] = 1
            for si in range(1, s_num):
                bf = ti.select(sbf_t[si] == 1, 0.0, 1.0)
                so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]
                if so == 2:
                    eil, eir = el, eh
                    side_l = isl_data_t[isl_ptr_l_t[eil]]
                    side_r = isl_data_t[isl_ptr_r_t[eir]]
                    hl, hr = h_t[eil], h_t[eir]
                    dx = sdc_t[si]
                    xq = sq_t[si]
                    xwh = ti.max(hl, hr) - ti.max(ez_t[eil], ez_t[eir], dh_t[si])
                    xdq = (-g * (hr - hl) / dx) * ti.max(xwh, 0.0) * dt
                    xf = 1.0 + g * dt * (n ** 2) * ti.abs(xq / (ti.max(xwh, 0.00001) ** (7.0 / 3.0)))
                    new_xq = (sita * xq + (1.0 - sita) / 2.0 * (sqn_t[side_l] + sqn_t[side_r]) + xdq) / xf
                    new_xq = ti.select(xwh < min_h, 0.0, new_xq)
                    new_xq *= bf
                    sq_t[si] = new_xq
                    sqn_t[si] = new_xq
                    xwh = ti.max(xwh, 0.01)
                    sdt = bf * afa * sl_t[si] / (ti.sqrt(g * xwh) + ti.abs(new_xq) / xwh)
                    sdt = ti.max((0.001 - sdt) * 100000.0, sdt)
                    ti.atomic_min(ndt_t[None], sdt)
                    flux = new_xq * sl_t[si]
                    ti.atomic_add(enq_t[eil, 1], flux)
                    ti.atomic_add(enq_t[eir, 0], flux)
                else:
                    eib, eit = el, eh
                    side_b = isl_data_t[isl_ptr_b_t[eib]]
                    side_t = isl_data_t[isl_ptr_top_t[eit]]
                    hb, ht = h_t[eib], h_t[eit]
                    dy = sdc_t[si]
                    yq = sq_t[si]
                    ywh = ti.max(hb, ht) - ti.max(ez_t[eib], ez_t[eit], dh_t[si])
                    ydq = (-g * (ht - hb) / dy) * ti.max(ywh, 0.0) * dt
                    yf = 1.0 + g * dt * (n ** 2) * ti.abs(yq / (ti.max(ywh, 0.00001) ** (7.0 / 3.0)))
                    new_yq = (sita * yq + (1.0 - sita) / 2.0 * (sqn_t[side_b] + sqn_t[side_t]) + ydq) / yf
                    new_yq = ti.select(ywh < min_h, 0.0, new_yq)
                    new_yq *= bf
                    sq_t[si] = new_yq
                    sqn_t[si] = new_yq
                    ywh = ti.max(ywh, 0.01)
                    sdt = bf * afa * sl_t[si] / (ti.sqrt(g * ywh) + ti.abs(new_yq) / ywh)
                    sdt = ti.max((0.001 - sdt) * 100000.0, sdt)
                    ti.atomic_min(ndt_t[None], sdt)
                    flux = new_yq * sl_t[si]
                    ti.atomic_add(enq_t[eib, 3], flux)
                    ti.atomic_add(enq_t[eit, 2], flux)
            if rainq > 0.0:
                cumulative_rain_time_t[None] += dt
                fr3[None] = fr5[None] = fr7[None] = horton_decay(3.0, 0.1, 2.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
                fr1[None] = fr2[None] = horton_decay(0.8, 0.02, 10.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
            for ei in range(1, e_num):
                ql = enq_t[ei, 0]
                qr = enq_t[ei, 1]
                qb = enq_t[ei, 2]
                qt = enq_t[ei, 3]
                tq = ql - qr + qb - qt + ssq_t[ei]
                eq_t[ei, 0], eq_t[ei, 1], eq_t[ei, 2], eq_t[ei, 3] = ql, qr, qb, qt
                enq_t[ei, 0] = enq_t[ei, 1] = enq_t[ei, 2] = enq_t[ei, 3] = 0.0
                eu = eu_t[ei]
                f1 = ti.select(eu == 1, 1.0, 0.0)
                f2 = ti.select(eu == 2, 1.0, 0.0)
                f3 = ti.select(eu == 3, 1.0, 0.0)
                f4 = ti.select(eu == 4, 1.0, 0.0)
                f5 = ti.select(eu == 5, 1.0, 0.0)
                f6 = ti.select(eu == 6, 1.0, 0.0)
                f7 = ti.select(eu == 7, 1.0, 0.0)
                ea = esl_t[ei] ** 2
                next_h = h_t[ei] + (tq * dt) / ea
                next_h += rainq * dt
                next_h -= (fr1[None] * f1 + fr2[None] * f2 + fr3[None] * f3 +
                           fr4[None] * f4 + fr5[None] * f5 + fr6[None] * f6 +
                           fr7[None] * f7) * dt
                next_h = ti.max(next_h, ez_t[ei])
                h_t[ei] = next_h
                depth_t[ei] = ti.max(next_h - ez_t[ei], 0.0)
            for count in range(b_num):
                bdei = bdei_t[count]
                h_t[bdei] = tide
            return ndt_t[None]

        @ti.kernel
        @no_type_check
        def update_velocities():
            for ei in range(1, e_num):
                esl   = esl_t[ei]
                depth = ti.max(h_t[ei] - ez_t[ei], 0.01)
                u_t[ei] = ti.select(depth < min_h, 0.0, (eq_t[ei, 0] + eq_t[ei, 1]) / esl / depth / 2.0)
                v_t[ei] = ti.select(depth < min_h, 0.0, (eq_t[ei, 2] + eq_t[ei, 3]) / esl / depth / 2.0)

        # ── Load pipe topology ────────────────────────────────────────────────
        pipe_fdb    = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes     = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl   = pipe_fdb[Node]['Node']
        primary_ei  = pipe_fdb[IndexLike]['node_primary_ei'].column.index.copy()
        nc_per_ei   = pipe_fdb[IndexLike]['node_count_per_ei'].column.index.copy()
        topo_ei     = pipe_fdb[PipeTopo]['PipeTopo'].column.ei.copy()
        topo_ptr    = pipe_fdb[IndexLike]['topo_ptr'].column.index.copy()
        node_is_outfall = np.array(
            [bool(nodes_tbl[i].is_outfall) for i in range(n_nodes)], dtype=bool
        )
        node_names = [str(nodes_tbl[i].name) for i in range(n_nodes)]
        del pipe_fdb
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        # ── Initialise GPU state ──────────────────────────────────────────────
        tide_step   = round((tts[1] - tts[0]) / 60.0)
        begin_index = max(0, (int(start_time_step * 5 / tide_step) - 1))
        current_time          = tts[begin_index]
        simulation_begin_time = current_time
        while rts[current_rain_idx] < simulation_begin_time:
            current_rain_idx += 1

        ex = copy_to_taichi(nes.column.x, ti.f32, None)
        ey = copy_to_taichi(nes.column.y, ti.f32, None)
        sx = copy_to_taichi(nss.column.x, ti.f32, None)
        init_gpu(ex, ey, isl_data_t, isl_ptr_l_t, sx)

        del ne_fdb, ns_fdb, tide_fdb, gate_fdb, rain_fdb, boundary_fdb
        del nes, nss, tides, gates, sbfs, bdeis, rainfalls, sts
        gc.collect()
        print('[2D] FDB freed, simulation starting.')

        # ── Output directory ──────────────────────────────────────────────────
        last_output_time  = current_time
        evolve_start_time = current_time
        output_uvh_fn = Path(domain_cfg.uvh_dir)
        if output_uvh_fn.exists():
            shutil.rmtree(output_uvh_fn)
        output_uvh_fn.mkdir(parents=True, exist_ok=True)

        # ── Main simulation loop ──────────────────────────────────────────────
        pipe_exchange_acc = 0.0
        coupling_interval = float(pipe_cfg.coupling_interval)
        timer = CouplingTimer()
        t_2d_window_start = time.perf_counter()

        while domain_cfg.duration == -1 or current_time - evolve_start_time < domain_cfg.duration:
            if shared['stop'].is_set():
                break

            if current_time >= tts[current_tide_idx + 1]:
                if current_tide_idx + 2 >= t_num:
                    break
                current_tide_idx += 1
            tide = _lerp(
                tls[current_tide_idx],
                tls[current_tide_idx + 1],
                (current_time - tts[current_tide_idx]) / (tts[current_tide_idx + 1] - tts[current_tide_idx]),
            )

            rainq = 0.0
            if current_time <= rts[r_num - 1]:
                if current_time > rts[current_rain_idx + 1]:
                    current_rain_idx += 1
                rainq = rqs[current_rain_idx + 1] / (rts[current_rain_idx + 1] - rts[current_rain_idx]) * 0.001

            dt = tick(tide, rainq)
            current_time      += dt
            pipe_exchange_acc += dt

            if pipe_exchange_acc >= coupling_interval:
                timer.total_2d_step += time.perf_counter() - t_2d_window_start
                window_dt = pipe_exchange_acc
                _exchange_with_1d(
                    shared=shared,
                    pipe_cfg=pipe_cfg,
                    window_dt=window_dt,
                    h_t=h_t, ssq_t=ssq_t, ez_t=ez_t, esl_t=esl_t,
                    primary_ei=primary_ei,
                    topo_ei=topo_ei,
                    topo_ptr=topo_ptr,
                    nc_per_ei=nc_per_ei,
                    node_names=node_names,
                    node_is_outfall=node_is_outfall,
                    name_to_idx=name_to_idx,
                    timer=timer,
                )
                pipe_exchange_acc -= coupling_interval
                t_2d_window_start = time.perf_counter()

            if current_time - last_output_time >= domain_cfg.yield_step:
                last_output_time += domain_cfg.yield_step
                cumulative_time = current_time - simulation_begin_time
                print(f'[2D] t={cumulative_time:.1f} s  dt={dt:.4f} s')
                update_velocities()
                uvh_db = fdb.ORM.truncate([fdb.TableDefn(UVH, e_num)])
                uvh_db[UVH][UVH].column.u[:] = u_t.to_numpy()
                uvh_db[UVH][UVH].column.v[:] = v_t.to_numpy()
                uvh_db[UVH][UVH].column.h[:] = h_t.to_numpy()
                time_str = datetime.fromtimestamp(last_output_time).strftime('%Y%m%d-%H%M%S')
                uvh_fn = output_uvh_fn / f'uvh_{time_str}.fdb'
                uvh_db.save(str(uvh_fn))
                uvh_db.unlink()

        timer.report('[2D]')

    finally:
        shared['stop'].set()


# ─── SWMM .inp file helpers ──────────────────────────────────────────────────────

def _find_index(inp_path: str):
    """Return (s1, s2, s3, e1, e2, e3): start/end line indices for JUNCTIONS, INFLOWS, OUTFALLS."""
    with open(inp_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    s1 = s2 = s3 = e1 = e2 = e3 = None
    for i, line in enumerate(lines):
        if line == '[JUNCTIONS]':
            s1 = i + 3
        elif s1 is not None and e1 is None and line == '':
            e1 = i
        if line == '[INFLOWS]':
            s2 = i + 3
        elif s2 is not None and e2 is None and line == '':
            e2 = i
        if line == '[OUTFALLS]':
            s3 = i + 3
        elif s3 is not None and e3 is None and line == '':
            e3 = i
    return s1, s2, s3, e1, e2, e3


def _update_start(inp_path: str, node_id_list: list, s2: int, e2: int) -> None:
    """Initialise [INFLOWS] section to zero for all junction nodes."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s2]
    for name in node_id_list:
        new_lines.append(
            f'{name:<15} FLOW             ""               FLOW     1.0      1.0      0\n'
        )
    new_lines.extend(lines[e2:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _initialize_junction_surdepth(inp_path: str, s1: int, e1: int) -> None:
    """Reset surcharge-depth column (index 4) to 0 for all junctions."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s1]
    for i in range(s1, e1):
        parts = lines[i].split()
        if len(parts) > 4:
            parts[4] = '0'
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e1:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _update_junction_surdepth(inp_path: str, level_dict: dict, s1: int, e1: int) -> None:
    """Update surcharge depth (column 4) for each junction from 2D level data."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s1]
    for i in range(s1, e1):
        parts = lines[i].split()
        if len(parts) > 4:
            name = parts[0]
            if name in level_dict:
                parts[4] = str(level_dict[name].get('level', 0.0))
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e1:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _update_inflows(inp_path: str, data: dict, node_id_list: list, s2: int, e2: int) -> None:
    """Rewrite [INFLOWS] section with flow values from 2D exchange data (m³/s)."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s2]
    for name in node_id_list:
        flow = data.get(name, {}).get('flow', 0.0)
        new_lines.append(
            f'{name:<15} FLOW             ""               FLOW     1.0      1.0      {flow}\n'
        )
    new_lines.extend(lines[e2:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _fixed_level(inp_path: str, data_dict: dict, s3: int, e3: int) -> None:
    """Update [OUTFALLS] tide level (column 3) with absolute WSE from 2D."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s3]
    for i in range(s3, e3):
        parts = lines[i].split()
        if len(parts) > 3:
            name = parts[0]
            if name in data_dict:
                parts[3] = str(data_dict[name].get('level', 0.0))
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e3:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _total_inflow_from_rpt(rpt_path: str) -> dict:
    """Parse SWMM .rpt Node Flooding Summary; returns {node_name: cumulative_flood_volume}.

    SWMM SI (CMS) reports 'Total Flood Volume' in 10^6 litres (= 1 ML = 1000 m³).
    Caller must diff successive calls and convert: flow_m3s = delta_ML * 1000 / window_dt.
    """
    with open(rpt_path, 'r', encoding='utf-8') as f:
        rows = [ln.strip().split() for ln in f.readlines()]
    result: dict = {}
    try:
        fi = rows.index(['Node', 'Flooding', 'Summary']) + 10
        for row in rows[fi:]:
            if not row:
                break
            result[row[0]] = float(row[5])
    except (ValueError, IndexError):
        pass
    return result


def _set_inp_duration(inp_path: str, duration_sec: float) -> None:
    """Rewrite [OPTIONS] END_DATE/END_TIME so SWMM runs exactly `duration_sec` seconds.

    The .inp must have START_DATE, START_TIME, END_DATE, END_TIME lines.
    We parse START and set END = START + duration_sec.
    """
    import re
    from datetime import datetime as _dt, timedelta

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    start_date = start_time = None
    for line in lines:
        m = re.match(r'^START_DATE\s+(\S+)', line)
        if m:
            start_date = m.group(1)
        m = re.match(r'^START_TIME\s+(\S+)', line)
        if m:
            start_time = m.group(1)

    if start_date is None or start_time is None:
        return  # cannot rewrite, let SWMM use original times

    # Parse MM/DD/YYYY HH:MM:SS
    start_dt = _dt.strptime(f'{start_date} {start_time}', '%m/%d/%Y %H:%M:%S')
    end_dt   = start_dt + timedelta(seconds=duration_sec)
    end_date_str = end_dt.strftime('%m/%d/%Y')
    end_time_str = end_dt.strftime('%H:%M:%S')

    new_lines = []
    for line in lines:
        if re.match(r'^END_DATE\s', line):
            new_lines.append(f'END_DATE             {end_date_str}\n')
        elif re.match(r'^END_TIME\s', line):
            new_lines.append(f'END_TIME             {end_time_str}\n')
        else:
            new_lines.append(line)

    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


# ─── 1D subprocess ───────────────────────────────────────────────────────────────

def _run_1d_pipe(shared, pipe_cfg: PipeConfig) -> None:
    """1D SWMM pipe-network loop; runs in a subprocess spawned by solver_coupled()."""
    import shutil as _shutil
    from pyswmm import Simulation, Output

    try:
        inp_runtime = str(pipe_cfg.inp_runtime)
        _shutil.copy2(pipe_cfg.inp, inp_runtime)

        s1, s2, s3, e1, e2, e3 = _find_index(inp_runtime)

        pipe_fdb  = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes   = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl = pipe_fdb[Node]['Node']
        node_names  = [str(nodes_tbl[i].name)        for i in range(n_nodes)]
        is_outfall  = [bool(nodes_tbl[i].is_outfall)  for i in range(n_nodes)]
        del pipe_fdb

        junction_names = [n for n, f in zip(node_names, is_outfall) if not f]

        _update_start(inp_runtime, junction_names, s2, e2)
        _initialize_junction_surdepth(inp_runtime, s1, e1)

        hotstart_dir = Path(pipe_cfg.hotstart_dir)
        hotstart_dir.mkdir(parents=True, exist_ok=True)

        prev_total_flood: dict = {}
        window_step = 0

        while True:
            if shared['stop'].is_set():
                break

            # Short-poll loop: check stop every 2s, raise after exchange_timeout
            waited = 0.0
            while not shared['2d_ready'].is_set():
                if shared['stop'].is_set():
                    break
                shared['2d_ready'].wait(timeout=2.0)
                waited += 2.0
                if waited >= float(pipe_cfg.exchange_timeout):
                    raise TimeoutError('[1D] Timeout waiting for 2D data')

            if shared['stop'].is_set():
                break

            with shared['lock']:
                raw_2d = dict(shared['2d_data'])
                shared['2d_ready'].clear()

            window_dt = float(raw_2d.get('__window_dt__', {}).get('level', pipe_cfg.coupling_interval))

            t_1d_start = time.perf_counter()

            _update_junction_surdepth(inp_runtime, raw_2d, s1, e1)
            _update_inflows(inp_runtime, raw_2d, junction_names, s2, e2)
            _fixed_level(inp_runtime, raw_2d, s3, e3)
            _set_inp_duration(inp_runtime, window_dt)

            hsf_in  = hotstart_dir / f'step_{window_step}.hsf'
            hsf_out = hotstart_dir / f'step_{window_step + 1}.hsf'
            with Simulation(inp_runtime) as sim:
                if hsf_in.exists():
                    sim.use_hotstart(str(hsf_in))
                for _ in sim:
                    pass
                sim.save_hotstart(str(hsf_out))

            # Read node levels from SWMM output
            out_path = str(Path(inp_runtime).with_suffix('.out'))
            node_levels: dict = {}
            try:
                with Output(out_path) as out:
                    for name in node_names:
                        try:
                            result = out.node_result(name, out.end)
                            node_levels[name] = float(list(result.values())[1])  # index 1 = depth/head
                        except Exception:
                            node_levels[name] = 0.0
            except Exception:
                pass  # output file may not exist on first step

            rpt_path = str(Path(inp_runtime).with_suffix('.rpt'))
            raw_flood = _total_inflow_from_rpt(rpt_path)

            data_1d: dict = {}
            for name in node_names:
                total_now = float(raw_flood.get(name, 0.0))
                prev_val  = prev_total_flood.get(name, 0.0)
                delta     = max(total_now - prev_val, 0.0)
                prev_total_flood[name] = total_now
                # 10^6 litres (ML) → m³ → m³/s
                flow_m3s = delta * 1000.0 / max(window_dt, 1.0)
                data_1d[name] = {
                    'level': node_levels.get(name, 0.0),
                    'flow': flow_m3s,
                }

            t_1d_elapsed = time.perf_counter() - t_1d_start

            with shared['lock']:
                shared['1d_data'].clear()
                shared['1d_data'].update(data_1d)
                shared['1d_data']['__1d_elapsed__'] = t_1d_elapsed
                shared['1d_ready'].set()
                shared['2d_ready'].clear()

            window_step += 1

    finally:
        shared['stop'].set()


# ─── Entry point ─────────────────────────────────────────────────────────────────

def solver_coupled(
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig,
    start_time_step: int = 0,
) -> None:
    """
    Coupled 2D/1D solver entry point.

    Spawns two subprocesses using the 'spawn' context (required for Taichi GPU):
      _run_2d()      – GPU 2D hydrodynamic simulation (mirrors solver_compact)
      _run_1d_pipe() – CPU SWMM pipe-network simulation

    The two processes exchange data every coupling_interval seconds via shared
    multiprocessing.Manager dicts and Events.
    """
    ctx = multiprocessing.get_context('spawn')
    mgr = ctx.Manager()

    shared = {
        '2d_data':  mgr.dict(),
        '1d_data':  mgr.dict(),
        '2d_ready': mgr.Event(),
        '1d_ready': mgr.Event(),
        'lock':     mgr.Lock(),
        'stop':     mgr.Event(),
    }

    p2d = ctx.Process(
        target=_run_2d,
        args=(shared, domain_cfg, force_cfg, pipe_cfg, start_time_step),
        daemon=False,
        name='solver-2d',
    )
    p1d = ctx.Process(
        target=_run_1d_pipe,
        args=(shared, pipe_cfg),
        daemon=False,
        name='solver-1d',
    )

    p2d.start()
    p1d.start()
    print(f'[coupled] 2D pid={p2d.pid}  1D pid={p1d.pid}')

    try:
        # join() with no timeout uses os.waitpid() — instant wake on exit,
        # no overflow risk, no Manager proxy issues.
        p2d.join()
        shared['stop'].set()
        p1d.join(timeout=120.0)
    except KeyboardInterrupt:
        print('[coupled] KeyboardInterrupt – signalling stop.')
        shared['stop'].set()
        p2d.join(timeout=30.0)
        p1d.join(timeout=30.0)
    finally:
        if p2d.is_alive():
            p2d.terminate()
        if p1d.is_alive():
            p1d.terminate()
        mgr.shutdown()
        print('[coupled] Both processes terminated.')
