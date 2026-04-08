"""
2D GPU hydrodynamic subprocess.

Contains:
  run_2d()       — main loop (subprocess target for multiprocessing.Process)
  horton_decay() — Taichi infiltration helper
  _lerp()        — linear interpolation
"""

import gc
import shutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import no_type_check

import taichi as ti
import fastdb4py as fdb

from ...input import DomainConfig, ForceConfig
from ...input.pipe import PipeConfig
from ...util.ti import init_taichi, copy_to_taichi
from ...schema.feature import (
    Ne, Ns, IndexLike, SideTopoInfo, Rainfall, Tide, Gate,
    U8Value, UVH, Node,
)
from .timer import CouplingTimer
from .exchange import compute_node_levels, send_to_1d, receive_from_1d, apply_sources


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@ti.func
@no_type_check
def horton_decay(initial: float, final: float, k: float, t: float) -> float:
    return final + (initial - final) * ti.exp(-k * t)


def run_2d(
    shared,
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None,
    start_time_step: int = 0,
) -> None:
    """2D GPU hydrodynamic loop; runs in a subprocess spawned by the driver."""
    try:
        _run_2d_impl(shared, domain_cfg, force_cfg, pipe_cfg, start_time_step)
    finally:
        has_pipe = pipe_cfg is not None
        if has_pipe:
            shared['stop'].set()


def _run_2d_impl(
    shared,
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None,
    start_time_step: int = 0,
) -> None:
    """Core implementation — separated from run_2d so finally-block stays clean."""
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

    # ── Taichi fields ─────────────────────────────────────────────────────
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

    # ── Taichi kernels (defined as closures over the fields above) ────────

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

    # TICK_KERNEL_PLACEHOLDER

    @ti.kernel
    @no_type_check
    def update_velocities():
        for ei in range(1, e_num):
            esl   = esl_t[ei]
            depth = ti.max(h_t[ei] - ez_t[ei], 0.01)
            u_t[ei] = ti.select(depth < min_h, 0.0, (eq_t[ei, 0] + eq_t[ei, 1]) / esl / depth / 2.0)
            v_t[ei] = ti.select(depth < min_h, 0.0, (eq_t[ei, 2] + eq_t[ei, 3]) / esl / depth / 2.0)

    # ── Load pipe topology (only when coupled) ─────────────────────────────
    has_pipe = pipe_cfg is not None
    if has_pipe:
        pipe_fdb    = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes     = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl   = pipe_fdb[Node]['Node']
        primary_ei  = pipe_fdb[IndexLike]['node_primary_ei'].column.index.copy()
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

    # ── Main simulation loop (async lag-1 coupling) ─────────────────────────
    pipe_exchange_acc = 0.0
    coupling_interval = float(pipe_cfg.coupling_interval) if has_pipe else 0.0
    timer = CouplingTimer()
    t_2d_window_start = time.perf_counter()
    prev_flood_return: dict = {}
    exchange_sent = False

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
        current_time += dt

        if has_pipe:
            pipe_exchange_acc += dt
            if pipe_exchange_acc >= coupling_interval:
                timer.total_2d_step += time.perf_counter() - t_2d_window_start
                window_dt = pipe_exchange_acc

                t_exc_start = time.perf_counter()
                prev_1d_wait = timer.total_1d_wait

                # receive previous 1D results (should be near-instant
                # since 1D finishes during the 2D stepping window)
                if exchange_sent:
                    prev_flood_return = receive_from_1d(shared, pipe_cfg, timer)

                # extract water levels at pipe node locations (pressure coupling)
                data_dict = compute_node_levels(
                    h_t, ez_t,
                    primary_ei, node_names, node_is_outfall,
                )

                # apply lagged flood-return as 2D source terms
                apply_sources(prev_flood_return, ssq_t, primary_ei, name_to_idx, e_num)

                # send to 1D (non-blocking) — 1D runs in parallel with next 2D window
                send_to_1d(shared, data_dict, window_dt)
                exchange_sent = True

                this_1d_wait = timer.total_1d_wait - prev_1d_wait
                timer.total_exchange_overhead += (time.perf_counter() - t_exc_start) - this_1d_wait
                timer.exchange_count += 1

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

    # drain final 1D result so the pipe process doesn't hang
    if has_pipe and exchange_sent:
        try:
            receive_from_1d(shared, pipe_cfg, timer)
        except (TimeoutError, OSError):
            pass
        timer.report('[2D]')
