"""Local-inertial solver with absolute-time spatial forcing and water accounting."""
import csv
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import no_type_check

import fastdb4py as fdb
import numpy as np
import taichi as ti

from ..input import DomainConfig, ForceConfig
from ..preprocess import preprocess
from ..preprocess.force import Forcing, MeshGeometry
from ..preprocess.hydraulics import connected_still_water, map_land_parameters, open_boundary_face_fractions
from ..schema.feature import Ne, Ns, IndexLike, SideTopoInfo, U8Value
from ..util.ti import init_taichi, copy_to_taichi
from .run_record import RunRecord, WaterBudget
from .device_budget import DeviceBudget
from .clock import StepClock, _seconds_to_ticks
from .output import (read_checkpoint, write_checkpoint, write_bed, write_depth_frame,
                     write_extrema, append_station_samples, resolve_station_cells)

_MIN_DT = 1e-4
_CHECKPOINT_INTERVAL_S = 30 * 60


def set_elevation(domain_cfg, elevate_meter: float):
    """
    Set the z data of element to a specified elevation (elevate_meter) if it is below that elevation.
    """
    pt_file = Path.cwd() / Path('./resource/elevate/123.txt')
    # Read xs and ys in pt_file as numpy
    if not pt_file.exists():
        raise FileNotFoundError(f'Elevate point file not found: {pt_file}')
    pts = np.loadtxt(str(pt_file), delimiter=',', dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    # Load ne.fdb and update z values based on the elevation points
    ne_fdb_fn = Path(domain_cfg.ne_fdb)
    ns_fdb_fn = Path(domain_cfg.ns_fdb) # use ns to calculate the bbox of ne
    ne_fdb = fdb.ORM.load(str(ne_fdb_fn), from_file=True)
    ns_fdb = fdb.ORM.load(str(ns_fdb_fn), from_file=True)

    nes = ne_fdb[Ne][Ne]
    nss = ns_fdb[Ns][Ns]

    # Lengths and counts
    e_num = len(nes)                                                # number of hydro elements
    p_num = len(pts)                                                # number of elevation points

    init_taichi(use_gpu=True, profiler=False)

    # Coordinate fields
    ex_t = copy_to_taichi(nes.column.x,  ti.f32, None)
    ey_t = copy_to_taichi(nes.column.y,  ti.f32, None)
    sx_t = copy_to_taichi(nss.column.x,  ti.f32, None)
    sy_t = copy_to_taichi(nss.column.y,  ti.f32, None)

    # Compact side-index fields for bounding-box calculation
    isl_data_t  = copy_to_taichi(ne_fdb[IndexLike]['isl_data'].column.index,  ti.i32, None)
    isl_ptr_l_t = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_l'].column.index, ti.i32, None)
    isl_ptr_b_t = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_b'].column.index, ti.i32, None)

    # Points field and result flag field
    pts_t          = copy_to_taichi(pts, ti.f32, (p_num, 2))
    needs_elevate_t = ti.field(dtype=ti.i32, shape=e_num)

    @ti.kernel
    @no_type_check
    def find_elements_to_elevate():
        for ei in range(1, e_num):
            # Determine bounding box from leftmost and bottommost side positions
            lsi = isl_data_t[isl_ptr_l_t[ei]]
            bsi = isl_data_t[isl_ptr_b_t[ei]]
            
            half_width = ex_t[ei] - sx_t[lsi]
            half_height = ey_t[ei] - sy_t[bsi]
            
            x_min = ex_t[ei] - half_width
            x_max = ex_t[ei] + half_width
            y_min = ey_t[ei] - half_height
            y_max = ey_t[ei] + half_height
            for pi in range(p_num):
                px = pts_t[pi, 0]
                py = pts_t[pi, 1]
                if x_min <= px <= x_max and y_min <= py <= y_max:
                    needs_elevate_t[ei] = 1
                    break

    find_elements_to_elevate()
    
    # Check if ei 354070 in picked elements
    if needs_elevate_t[354070] == 1:
        print(f'Element 354070 needs elevation. Its center is at ({ex_t[354070]}, {ey_t[354070]}), z = {nes.column.z[354070]}')
    else:
        print('OK!')

    # Update z values where a point fell inside the element bbox and current z is below the target
    flags = needs_elevate_t.to_numpy()
    z_arr = nes.column.z
    mask = (flags > 0) & (z_arr < elevate_meter)
    z_arr[mask] = elevate_meter
    print(f'set_elevation: {mask.sum()} elements elevated to {elevate_meter} m')

    # Save updated ne.fdb back to disk
    ne_fdb.save(str(ne_fdb_fn))
    print(f'set_elevation: ne.fdb saved to {ne_fdb_fn}')


@ti.func
def face_discharge(qn: ti.template(), lengths: ti.template(), ids: ti.template(),
                   start: ti.i32, count: ti.i32):
    result = 0.
    if count == 1:
        # Keep single-edge faces bitwise identical to the original stencil.
        result = qn[ids[start]]
    elif count > 1:
        total, width = 0., 0.
        for offset in range(count):
            side = ids[start + offset]
            total += qn[side] * lengths[side]
            width += lengths[side]
        result = total / width
    return result


@ti.func
def boundary_neighbor_sum(q: ti.f32, left: ti.f32, right: ti.f32,
                          left_exterior: ti.f32, right_exterior: ti.f32):
    """Linearly continue missing theta neighbours at prescribed boundaries."""
    result = left + right
    if left_exterior > 0. or right_exterior > 0.:
        # L = left + fL * (2*q - R), R = right + fR * (2*q - L).
        # Only L+R enters theta; a fully missing face fixes this sum to 2*q.
        if left_exterior == 1. or right_exterior == 1.:
            result = 2. * q
        else:
            product = left_exterior * right_exterior
            result = ((1. - right_exterior) * left + (1. - left_exterior) * right
                      + 2. * q * (left_exterior + right_exterior - 2. * product)) / (1. - product)
    return result


def run_simulation(domain_cfg: DomainConfig, force_cfg: ForceConfig, pipe_cfg=None):
    """Preprocess and simulate, recording failures from either stage."""
    with RunRecord(domain_cfg, force_cfg, pipe_cfg) as record:
        record.hash_inputs(domain_cfg, force_cfg, pipe_cfg)
        preprocess(domain_cfg, force_cfg)
        if pipe_cfg is not None:
            from ..preprocess.pipe import prepare_pipe
            prepare_pipe(pipe_cfg, domain_cfg)
        return _solve(domain_cfg, force_cfg, record, pipe_cfg)


def solver(domain_cfg: DomainConfig, force_cfg: ForceConfig, pipe_cfg=None):
    """Simulate an already preprocessed domain, including configured warmup."""
    with RunRecord(domain_cfg, force_cfg, pipe_cfg) as record:
        record.hash_inputs(domain_cfg, force_cfg, pipe_cfg)
        return _solve(domain_cfg, force_cfg, record, pipe_cfg)


def _solve(domain_cfg, force_cfg, record, pipe_cfg=None, end_time=None):
    end_time = domain_cfg.duration if end_time is None else end_time
    if pipe_cfg is not None:
        from .coupled.runtime import PipeCoupling
        with PipeCoupling(domain_cfg, pipe_cfg) as coupling:
            return _evolve(domain_cfg, force_cfg, record, coupling, end_time)
    return _evolve(domain_cfg, force_cfg, record, None, end_time)


def _evolve(domain_cfg, force_cfg, record, coupling, end_time):
    init_taichi(use_gpu=True, profiler=False)
    record.device()
    ne_db = fdb.ORM.load(domain_cfg.ne_fdb, from_file=True)
    ns_db = fdb.ORM.load(domain_cfg.ns_fdb, from_file=True)
    bd_db = fdb.ORM.load(domain_cfg.boundary_fdb, from_file=True)
    nes, nss = ne_db[Ne][Ne], ns_db[Ns][Ns]
    mesh = MeshGeometry.from_fdb(ne_db, ns_db)
    forcing = Forcing.load(domain_cfg, force_cfg, mesh)
    e_num, s_num = len(nes), len(nss)
    b_num = len(forcing.boundaries)
    side_n, cell_f0, cell_fc, cell_k = map_land_parameters(
        domain_cfg.land_parameters, nes.column.type,
        ns_db[SideTopoInfo][SideTopoInfo].column.info.reshape(-1, 3))
    g, afa, sita, min_h = 9.81, domain_cfg.afa, domain_cfg.sita, domain_cfg.min_h
    ex_t = copy_to_taichi(nes.column.x, ti.f32, None)
    ey_t = copy_to_taichi(nes.column.y, ti.f32, None)
    ez_t = copy_to_taichi(nes.column.z, ti.f32, None)
    eu_t = copy_to_taichi(nes.column.type, ti.i32, None)
    area_t = copy_to_taichi(mesh.area.astype('float32'), ti.f32, None)
    eq_t = ti.field(ti.f32, (e_num, 4))
    increment_t = ti.field(ti.f32, e_num)
    rain_depth_t = ti.field(ti.f32, e_num)
    ndt_t = ti.field(ti.f32, ())
    dh_t = ti.field(ti.f32, s_num)
    sq_t = ti.field(ti.f32, s_num)
    donor_scale_t = ti.field(ti.f32, e_num)
    limited_cells_t = ti.field(ti.u32, ())
    limited_faces_t = ti.field(ti.u32, ())
    minimum_scale_t = ti.field(ti.f32, ())
    cfl_t = ti.field(ti.f32, s_num)
    flow_depth_t = ti.field(ti.f32, s_num)
    sdc_t = ti.field(ti.f32, s_num)
    sl_t = copy_to_taichi(nss.column.length, ti.f32, None)
    side_n_t = copy_to_taichi(side_n, ti.f32, None)
    infiltration_f0_t = copy_to_taichi(cell_f0, ti.f32, None)
    infiltration_fc_t = copy_to_taichi(cell_fc, ti.f32, None)
    infiltration_k_t = copy_to_taichi(cell_k, ti.f32, None)
    sbf_t = copy_to_taichi(bd_db[U8Value]['sbf'].column.value, ti.u8, None)
    sts_t = copy_to_taichi(ns_db[SideTopoInfo][SideTopoInfo].column.info, ti.i32, (s_num, 3))
    isl_data_t = copy_to_taichi(ne_db[IndexLike]['isl_data'].column.index, ti.i32, None)
    face_ptrs = np.column_stack([
        ne_db[IndexLike][f'isl_ptr_{direction}'].column.index
        for direction in ('l', 'r', 'b', 't')])
    face_counts = np.column_stack([
        getattr(nes.column, f'{direction}_side_num')
        for direction in ('l', 'r', 'b', 't')])
    face_ptr_t = copy_to_taichi(face_ptrs, ti.i32, None)
    face_count_t = copy_to_taichi(face_counts, ti.i32, None)
    face_qn_t = ti.field(ti.f32, (e_num, 4))
    u_t, v_t = (ti.field(ti.f32, e_num) for _ in range(2))
    depth_t = ti.field(ti.f32, e_num)
    pipe_q_t = ti.field(ti.f32, e_num)
    ssq_t = ti.field(ti.f32, e_num)
    cumulative_rain_time_t = ti.field(ti.f32, e_num)
    infiltration_t = ti.field(ti.f32, e_num)
    boundary_ids = np.full(e_num, -1, dtype=np.int32)
    weights = np.zeros(e_num, dtype=np.float32)
    kinds = np.zeros(max(1, b_num), dtype=np.int32)
    for i, b in enumerate(forcing.boundaries):
        boundary_ids[b.cells] = i
        weights[b.cells] = b.weights
        kinds[i] = int(b.kind == 'discharge')
    boundary_id_t = copy_to_taichi(boundary_ids, ti.i32, None)
    open_faces = np.zeros((1, 4), dtype=np.float32)
    selected_sides = np.zeros(s_num, dtype=bool)
    for b in forcing.boundaries:
        selected_sides[b.side_ids] = True
    if sita != 1.:
        open_faces = open_boundary_face_fractions(
            selected_sides, nss.column.length,
            ns_db[SideTopoInfo][SideTopoInfo].column.info.reshape(-1, 3),
            face_ptrs, face_counts, ne_db[IndexLike]['isl_data'].column.index)
    open_face_t = copy_to_taichi(open_faces, ti.f32, None)
    weight_t = copy_to_taichi(weights, ti.f32, None)
    kind_t = copy_to_taichi(kinds, ti.i32, None)
    values_t = ti.field(ti.f32, max(1, b_num))
    level_values_t = ti.field(ti.f32, max(1, b_num))
    level_initial_t = ti.field(ti.f32, e_num)
    rain_index_t = copy_to_taichi(forcing.rain.pixel_index, ti.i32, None)
    rain_rates_t = ti.field(ti.f32, forcing.rain.millimetres.shape[1])
    # Each forcing/coupling event can add one step shorter than the usual floor.
    interval = min(domain_cfg.yield_step, max(domain_cfg.warmup_duration, end_time))
    step_bound = int(np.ceil(interval / float(np.float32(_MIN_DT)))) + len(forcing.events) + 2
    if coupling is not None:
        step_bound += int(np.ceil(interval / coupling.pipe_cfg.coupling_interval)) + 2
    device_budget = DeviceBudget((e_num + 255) // 256, 4 + 6 * b_num, step_bound)
    diagnostic_t = device_budget.step
    record.data['budget_accumulator'].update(word_count=device_budget.word_count,
                                           interval_step_bound=step_bound)
    gate_open = ti.field(ti.i32, ())
    gate = force_cfg.gate
    coupled_enabled = coupling is not None
    gate_type = gate.land_type if gate else -1

    state = read_checkpoint(domain_cfg.restart_state) if domain_cfg.restart_state else None
    state_meta = json.loads(state.metadata_json) if state else {}
    continuation = bool(state_meta.get('run_state', False))
    if continuation and coupling is not None:
        raise ValueError('A surface checkpoint cannot restore the SWMM actor state')
    origin_before_start = (state_meta['clock_origin_before_start_s'] if continuation
                           else domain_cfg.warmup_duration)
    initial_clock = ((state.clock_hi * 2**32 + state.clock_lo) / 2**40
                     if state else 0.)
    if state:
        end_tick = _seconds_to_ticks(end_time + origin_before_start)
        state_tick = (state.clock_hi << 32) | state.clock_lo
        if state_tick > end_tick:
            raise ValueError('Checkpoint time is later than the requested simulation end')
        if continuation and state_tick == end_tick:
            record.data['completion_reason'] = 'checkpoint_at_end'
    clock = StepClock(initial_clock)
    raining_t = ti.field(ti.i32, ())
    batch_slot_t = ti.field(ti.i32, ())
    coefficients_t = ti.field(ti.f32, (max(1, b_num), 2))
    max_depth_t, max_speed_t = (ti.field(ti.f32, e_num) for _ in range(2))
    peak_hi_t, peak_lo_t = (ti.field(ti.u32, e_num) for _ in range(2))
    widths_t = copy_to_taichi((mesh.bounds[:, 2] - mesh.bounds[:, 0]).astype('float32'), ti.f32, None)
    heights_t = copy_to_taichi((mesh.bounds[:, 3] - mesh.bounds[:, 1]).astype('float32'), ti.f32, None)
    pipe_volume_t = ti.field(ti.f32, e_num if coupling else 1)
    pipe_correction_t = ti.field(ti.f32, e_num if coupling else 1)
    cfl_arg_t = ti.field(ti.i32, ())
    limit_value_t = ti.field(ti.f32, 5)  # candidate, two cell depths, face depth, velocity
    limit_ids_t = ti.field(ti.i32, 3)
    station_cells = resolve_station_cells(domain_cfg.stations, domain_cfg.ne_fdb, domain_cfg.ns_fdb)
    station_names = list(station_cells)
    station_count = len(station_names)
    station_id_t = copy_to_taichi(np.array(list(station_cells.values()) or [0], dtype=np.int32), ti.i32, None)
    station_sample_t = ti.field(ti.f32, (max(1, station_count), 4))

    @ti.func
    def velocity(ei):
        h = ti.max(depth_t[ei], 0.01)
        u, v = 0., 0.
        if depth_t[ei] >= min_h:
            u = (eq_t[ei, 0] + eq_t[ei, 1]) / heights_t[ei] / h / 2.
            v = (eq_t[ei, 2] + eq_t[ei, 3]) / widths_t[ei] / h / 2.
        return u, v

    @ti.kernel
    @no_type_check
    def prepare_step():
        clock.choose(ndt_t[None], batch_slot_t[None])
        if clock.active[None] == 1:
            limited_cells_t[None], limited_faces_t[None] = 0, 0
            minimum_scale_t[None] = 1.
            for bi in range(b_num):
                values_t[bi] = coefficients_t[bi, 0] + coefficients_t[bi, 1] * (clock.local[None] + clock.dt[None] / 2.)
                level_values_t[bi] = coefficients_t[bi, 0] + coefficients_t[bi, 1] * clock.end_local[None]
            if ndt_t[None] < limit_value_t[0]:
                si = cfl_arg_t[None]
                el, eh = sts_t[si, 1], sts_t[si, 2]
                h = flow_depth_t[si]
                limit_value_t[0] = ndt_t[None]
                limit_value_t[1], limit_value_t[2] = depth_t[el], depth_t[eh]
                limit_value_t[3] = h
                limit_value_t[4] = sq_t[si] / ti.max(h, 1e-30)
                limit_ids_t[0], limit_ids_t[1], limit_ids_t[2] = si, el, eh

    @ti.kernel
    def finish_step():
        if clock.active[None] == 1:
            clock.trace[batch_slot_t[None], 6] = limited_cells_t[None]
            clock.trace[batch_slot_t[None], 7] = limited_faces_t[None]
            clock.trace[batch_slot_t[None], 8] = ti.bit_cast(minimum_scale_t[None], ti.u32)
        clock.commit(batch_slot_t[None])
        batch_slot_t[None] += 1

    @ti.func
    def add_diagnostic(block, term, value):
        if value != 0.:
            ti.atomic_add(diagnostic_t[block, term], value)

    @ti.kernel
    @no_type_check
    def initialize():
        for ei in range(1, e_num):
            depth_t[ei] = 0.
            infiltration_t[ei] = infiltration_f0_t[ei]
        for si in range(1, s_num):
            dh_t[si] = -999.
            so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]
            if so == 2:
                sdc_t[si] = ti.max(ti.abs(ex_t[eh] - ex_t[el]), 0.01)
            else:
                sdc_t[si] = ti.max(ti.abs(ey_t[eh] - ey_t[el]), 0.01)

    @ti.kernel
    @no_type_check
    def apply_initial_levels():
        for ei in range(1, e_num):
            bi = boundary_id_t[ei]
            if bi >= 0 and kind_t[bi] == 0:
                previous = depth_t[ei]
                depth_t[ei] = level_initial_t[ei]
                add_diagnostic(ei // 256, 4 + 5 * bi, (depth_t[ei] - previous) * area_t[ei])

    @ti.kernel
    @no_type_check
    def apply_gate_event(impose_levels: ti.i32):
        for ei in range(1, e_num):
            if eu_t[ei] == gate_type:
                # Bed motion itself preserves depth. A coincident prescribed
                # level is a separate, measured zero-time boundary exchange.
                ez_t[ei] = 0.
                bi = boundary_id_t[ei]
                if impose_levels:
                    if bi >= 0:
                        if kind_t[bi] == 0:
                            previous = depth_t[ei]
                            target = ti.max(level_values_t[bi] - ez_t[ei], 0.)
                            depth_t[ei] = target
                            add_diagnostic(ei // 256, 4 + 5 * bi, (target-previous)*area_t[ei])
        gate_open[None] = 1

    @ti.kernel
    @no_type_check
    def tick():
        dt = clock.dt[None]
        raining = raining_t[None]
        for block, term in diagnostic_t:
            if clock.active[None] == 1:
                diagnostic_t[block, term] = 0.
        if ti.static(sita != 1.):
            for ei in range(e_num):
                if clock.active[None] == 1:
                    for direction in ti.static(range(4)):
                        face_qn_t[ei, direction] = face_discharge(
                            sq_t, sl_t, isl_data_t, face_ptr_t[ei, direction],
                            face_count_t[ei, direction])

            # Tick sides, reading only the completed previous-step face averages.
        for si in range(1, s_num):
            if clock.active[None] == 1:
                bf = ti.select(sbf_t[si] == 1, 0.0, 1.0)  # boundary factor
                so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]   # orient, lower element, higher element
                # Handle flow in direction x (Type 2: Vertical side, connects Left/Right)
                if so == 2:
                    eil, eir = el, eh
                    hl, hr = ez_t[eil] + depth_t[eil], ez_t[eir] + depth_t[eir]
                    dx = sdc_t[si]
                    xq = sq_t[si]
                    xwh = ti.max(hl, hr) - ti.max(ez_t[eil], ez_t[eir], dh_t[si]) # water height in direction x
                    # Calculate the unit discharge quantity caused by pressure gradient in direction x
                    xdq = (-g * (hr - hl) / dx) * ti.max(xwh, 0.0) * dt
                    # Calculate the friction term in direction x (Manning's formula)
                    xf = 1.0 + g * dt * (side_n_t[si] ** 2) * ti.abs(xq / (ti.max(xwh, 0.00001) ** (7.0 / 3.0)))
                    x_neighbors = face_qn_t[eil, 0] + face_qn_t[eir, 1]
                    if ti.static(sita != 1.):
                        x_neighbors = boundary_neighbor_sum(xq, face_qn_t[eil, 0], face_qn_t[eir, 1],
                                                            open_face_t[eil, 0], open_face_t[eir, 1])
                    new_xq = (sita * xq + (1.0 - sita) / 2.0 * x_neighbors + xdq) / xf
                    new_xq = ti.select(xwh < min_h, 0.0, new_xq)    # cutoff small flow
                    new_xq *= bf                                    # zero flow for boundary sides
                    sq_t[si] = new_xq                               # update current flow quantity in direction x
                # Handle flow in direction y (Type 1: Horizontal side, connects Bottom/Top)
                else:
                    eib, eit = el, eh
                    hb, ht = ez_t[eib] + depth_t[eib], ez_t[eit] + depth_t[eit]
                    dy = sdc_t[si]
                    yq = sq_t[si]
                    ywh = ti.max(hb, ht) - ti.max(ez_t[eib], ez_t[eit], dh_t[si]) # water height in direction y
                    # Calculate the unit discharge quantity caused by pressure gradient in direction y
                    ydq = (-g * (ht - hb) / dy) * ti.max(ywh, 0.0) * dt
                    # Calculate the friction term in direction y (Manning's formula)
                    yf = 1.0 + g * dt * (side_n_t[si] ** 2) * ti.abs(yq / (ti.max(ywh, 0.00001) ** (7.0 / 3.0)))
                    y_neighbors = face_qn_t[eib, 2] + face_qn_t[eit, 3]
                    if ti.static(sita != 1.):
                        y_neighbors = boundary_neighbor_sum(yq, face_qn_t[eib, 2], face_qn_t[eit, 3],
                                                            open_face_t[eib, 2], open_face_t[eit, 3])
                    new_yq = (sita * yq + (1.0 - sita) / 2.0 * y_neighbors + ydq) / yf
                    new_yq = ti.select(ywh < min_h, 0.0, new_yq)    # cutoff small flow
                    new_yq *= bf                                    # zero flow for boundary sides
                    sq_t[si] = new_yq                               # update current flow quantity in direction y


            # Freeze one supply factor per donor cell before changing any edge.
            # Incoming edge flux is deliberately excluded from available water.
        for ei in range(1, e_num):
            if clock.active[None] == 1:
                bi = boundary_id_t[ei]
                discharge = 0.
                if bi >= 0 and kind_t[bi] == 1:
                    discharge = values_t[bi] * weight_t[ei]
                rainq = 0.
                if raining:
                    rainq = rain_rates_t[rain_index_t[ei]]
                positive = (ti.max(pipe_q_t[ei], 0.) + ti.max(discharge, 0.)) * dt
                available = depth_t[ei] * area_t[ei] + positive + rainq * dt * area_t[ei]
                outgoing = (ti.max(-pipe_q_t[ei], 0.) + ti.max(-discharge, 0.)) * dt
                for direction in ti.static(range(4)):
                    for j in range(face_count_t[ei, direction]):
                        side = isl_data_t[face_ptr_t[ei, direction] + j]
                        signed_q = sq_t[side] * sl_t[side]
                        if ti.static(direction == 0 or direction == 2):
                            signed_q = -signed_q
                        outgoing += ti.max(signed_q, 0.) * dt
                donor_scale_t[ei] = 1.
                if outgoing > available and not (bi >= 0 and kind_t[bi] == 0):
                    donor_scale_t[ei] = available / outgoing
                    ti.atomic_add(limited_cells_t[None], 1)
                    ti.atomic_min(minimum_scale_t[None], donor_scale_t[ei])
            # Each shared subedge has a single writer. Both cells receive this flux.
        for si in range(1, s_num):
            if clock.active[None] == 1:
                so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]
                donor = ti.select(sq_t[si] >= 0., el, eh)
                if sq_t[si] != 0. and donor_scale_t[donor] < 1.:
                    ti.atomic_add(limited_faces_t[None], 1)
                sq_t[si] *= donor_scale_t[donor]
        for ei in range(1, e_num):
            if clock.active[None] == 1:
                bi = boundary_id_t[ei]
                pipe = ti.max(pipe_q_t[ei], 0.) + ti.min(pipe_q_t[ei], 0.) * donor_scale_t[ei]
                discharge = 0.
                if bi >= 0 and kind_t[bi] == 1:
                    requested = values_t[bi] * weight_t[ei]
                    discharge = ti.max(requested, 0.) + ti.min(requested, 0.) * donor_scale_t[ei]
                    add_diagnostic(ei // 256, 4 + 5 * b_num + bi, discharge * dt)
                ssq_t[ei] = pipe + discharge
                if ti.static(coupled_enabled):
                    increment = pipe * dt - pipe_correction_t[ei]
                    total = pipe_volume_t[ei] + increment
                    pipe_correction_t[ei] = (total - pipe_volume_t[ei]) - increment
                    pipe_volume_t[ei] = total
                rainq = 0.
                if raining:
                    rainq = rain_rates_t[rain_index_t[ei]]
                if rainq > 0.:
                    cumulative_rain_time_t[ei] += dt
                    infiltration_t[ei] = infiltration_fc_t[ei] + (
                        infiltration_f0_t[ei] - infiltration_fc_t[ei]) * ti.exp(
                            -infiltration_k_t[ei] * cumulative_rain_time_t[ei] / 3600.)
                ql, qr, qb, qt = 0., 0., 0., 0.
                for direction in ti.static(range(4)):
                    face_flux = 0.
                    for j in range(face_count_t[ei, direction]):
                        si = isl_data_t[face_ptr_t[ei, direction] + j]
                        face_flux += sq_t[si] * sl_t[si]
                    if ti.static(direction == 0): ql = face_flux
                    elif ti.static(direction == 1): qr = face_flux
                    elif ti.static(direction == 2): qb = face_flux
                    else: qt = face_flux
                tq = ql - qr + qb - qt + ssq_t[ei]
                eq_t[ei, 0], eq_t[ei, 1], eq_t[ei, 2], eq_t[ei, 3] = ql, qr, qb, qt
                ea, previous = area_t[ei], depth_t[ei]
                rain_depth = rainq * dt
                # Materialize the net increment before it meets the large
                # inventory. Separate offloads prevent compiler reassociation
                # into (previous + flux_increment) + rain_depth on Metal.
                increment_t[ei] = tq * dt / ea + rain_depth
                rain_depth_t[ei] = rain_depth
        for ei in range(1, e_num):
            if clock.active[None] == 1:
                bi = boundary_id_t[ei]
                ea, previous = area_t[ei], depth_t[ei]
                pipe = ti.max(pipe_q_t[ei], 0.) + ti.min(pipe_q_t[ei], 0.) * donor_scale_t[ei]
                ql, qr, qb, qt = eq_t[ei, 0], eq_t[ei, 1], eq_t[ei, 2], eq_t[ei, 3]
                rain_depth = rain_depth_t[ei]
                raw_depth = previous + increment_t[ei]
                removed = ti.min(infiltration_t[ei] * dt, ti.max(raw_depth, 0.))
                raw_depth -= removed
                floor = ti.max(-raw_depth, 0.)
                add_diagnostic(ei // 256, 0, removed * ea)
                add_diagnostic(ei // 256, 3, pipe * dt)
                if bi >= 0 and kind_t[bi] == 0:
                    add_diagnostic(ei // 256, 2, floor * ea)
                    target = ti.max(level_values_t[bi] - ez_t[ei], 0.)
                    # Prescribed cells need no intermediate rounded state. Keep the
                    # physical source components separate until host float64 reduction.
                    offset = 4 + 5 * bi
                    add_diagnostic(ei // 256, offset, (target - previous) * ea)
                    add_diagnostic(ei // 256, offset + 1, -(ql - qr + qb - qt) * dt)
                    add_diagnostic(ei // 256, offset + 2, -pipe * dt)
                    add_diagnostic(ei // 256, offset + 3, -rain_depth * ea)
                    add_diagnostic(ei // 256, offset + 4, removed * ea)
                    depth_t[ei] = target
                else:
                    add_diagnostic(ei // 256, 1, floor * ea)
                    depth_t[ei] = ti.max(raw_depth, 0.)
                u, v = velocity(ei)
                u_t[ei], v_t[ei] = u, v
                max_speed_t[ei] = ti.max(max_speed_t[ei], ti.sqrt(u*u + v*v))
                if depth_t[ei] > max_depth_t[ei]:
                    max_depth_t[ei] = depth_t[ei]
                    peak_hi_t[ei], peak_lo_t[ei] = clock.step_end[0], clock.step_end[1]
        for block, term in diagnostic_t:
            if clock.active[None] == 1:
                device_budget.accumulate(block, term)



    @ti.kernel
    @no_type_check
    def compute_cfl():
        if clock.active[None] == 1:
            ndt_t[None] = float('inf')
        for si in range(1, s_num):
            if clock.active[None] == 1:
                el, eh = sts_t[si, 1], sts_t[si, 2]
                cfl_t[si] = float('inf')
                # Exterior edges never enter the reduction, including dry ones.
                if el != 0 and eh != 0:
                    head = ti.max(ez_t[el] + depth_t[el], ez_t[eh] + depth_t[eh])
                    h = ti.max(head - ti.max(ez_t[el], ez_t[eh], dh_t[si]), 0.01)
                    flow_depth_t[si] = h
                    cfl_t[si] = afa * sl_t[si] / (ti.sqrt(g * h) + ti.abs(sq_t[si]) / h)
                    ti.atomic_min(ndt_t[None], cfl_t[si])
        if clock.active[None] == 1:
            cfl_arg_t[None] = s_num
        for si in range(1, s_num):
            if clock.active[None] == 1:
                if cfl_t[si] == ndt_t[None] and sts_t[si, 1] != 0 and sts_t[si, 2] != 0:
                    ti.atomic_min(cfl_arg_t[None], si)
        if clock.active[None] == 1:
            if cfl_arg_t[None] == s_num:
                cfl_arg_t[None] = 0


    def supported_dt():
        candidate = float(ndt_t[None])
        if candidate < _MIN_DT:
            si = int(cfl_arg_t[None])
            cells = ns_db[SideTopoInfo][SideTopoInfo].column.info.reshape(-1, 3)[si, 1:]
            depths = depth_t.to_numpy()[cells]
            h = float(flow_depth_t[si])
            q = float(sq_t[si])
            raise RuntimeError(f'CFL {candidate:.12g}s below supported minimum {_MIN_DT}s; '
                               f'edge={si}, cells={cells.tolist()}, depth_m={depths.tolist()}, '
                               f'face_depth_m={h}, unit_discharge_m2_s={q}, velocity_m_s={q/h if h else 0.}')
        return candidate

    @ti.kernel
    @no_type_check
    def initial_diagnostics():
        for ei in range(1, e_num):
            for direction in ti.static(range(4)):
                total = 0.
                for j in range(face_count_t[ei, direction]):
                    si = isl_data_t[face_ptr_t[ei, direction] + j]
                    total += sq_t[si] * sl_t[si]
                eq_t[ei, direction] = total
            u, v = velocity(ei)
            u_t[ei], v_t[ei] = u, v
            max_speed_t[ei] = ti.max(max_speed_t[ei], ti.sqrt(u*u + v*v))
            if depth_t[ei] > max_depth_t[ei]:
                max_depth_t[ei] = depth_t[ei]
                peak_hi_t[ei], peak_lo_t[ei] = clock.now[0], clock.now[1]

    @ti.kernel
    def reset_extrema():
        for ei in range(1, e_num):
            max_depth_t[ei] = depth_t[ei]
            max_speed_t[ei] = ti.sqrt(u_t[ei]*u_t[ei] + v_t[ei]*v_t[ei])
            peak_hi_t[ei], peak_lo_t[ei] = clock.now[0], clock.now[1]

    @ti.kernel
    def sample_stations():
        for i in range(station_count):
            ei = station_id_t[i]
            station_sample_t[i, 0] = depth_t[ei]
            station_sample_t[i, 1] = ez_t[ei] + depth_t[ei]
            station_sample_t[i, 2], station_sample_t[i, 3] = u_t[ei], v_t[ei]

    initialize()
    state_identity = {
        'ne_sha256': record.data['input_sha256'][str(Path(domain_cfg.ne).resolve())],
        'ns_sha256': record.data['input_sha256'][str(Path(domain_cfg.ns).resolve())],
        'gate_config': gate.model_dump_json() if gate else 'null',
    }
    boundary_identity = [dict(name=b.name, kind=b.kind,
        selected_edges_sha256=hashlib.sha256(np.sort(b.side_ids).astype('<i8').tobytes()).hexdigest())
        for b in forcing.boundaries]
    if state:
        if len(state.depth) != e_num or len(state.q) != s_num:
            raise ValueError('Checkpoint size differs from mesh')
        for key in state_identity:
            if state_meta[key] != state_identity[key]:
                raise ValueError(f'Checkpoint {key} differs from the current mesh/configuration')
        if continuation:
            if state_meta.get('boundary_identity') != boundary_identity:
                raise ValueError('Checkpoint boundary identity or order differs from configuration')
            expected = {'start': domain_cfg.start.isoformat(), 'afa': afa, 'sita': sita,
                        'min_h': min_h, 'land_parameters': domain_cfg.model_dump(mode='json')['land_parameters']}
            if state_meta['physics_config'] != expected:
                raise ValueError('Checkpoint clock or physical parameters differ from configuration')
        depth_t.from_numpy(state.depth)
        ez_t.from_numpy(state.bed)
        sq_t.from_numpy(state.q)
        cumulative_rain_time_t.from_numpy(state.rain_time)
        infiltration_t.from_numpy(state.infiltration)
        max_depth_t.from_numpy(state.max_depth)
        max_speed_t.from_numpy(state.max_speed)
        peak_hi_t.from_numpy(state.first_time_hi)
        peak_lo_t.from_numpy(state.first_time_lo)
        gate_open[None] = int(state.gate_open)
    bed_np = ez_t.to_numpy()
    root = Path(domain_cfg.output_dir)
    record.data.update(bed_events=[], output={
        'root': 'output', 'frame_interval_s': 900, 'station_budget_interval_s': domain_cfg.yield_step,
        'frames': 'F32 depth only; bed.fdb plus manifest bed_events reconstruct water level',
        'extrema_time_origin': (domain_cfg.start - timedelta(seconds=origin_before_start)).isoformat(),
        'clock_quantum_s': 2.**-40,
        'extrema_window': 'simulation' if end_time > 0 else 'warmup',
    })
    elapsed = clock.host_now_seconds - origin_before_start
    next_output = min((math.floor(elapsed / domain_cfg.yield_step) + 1) * domain_cfg.yield_step,
                      0. if elapsed < 0. else end_time)
    next_frame = min((math.floor(elapsed / 900) + 1) * 900, 0. if elapsed < 0. else end_time)
    steps = warmup_steps = 0
    previous_steps = state_meta['total_steps'] if continuation else 0
    previous_warmup_steps = state_meta['total_warmup_steps'] if continuation else 0
    min_dt, max_dt, integrated_dt = float('inf'), 0., 0.
    discharge_peaks = np.zeros(b_num, dtype=np.float64)
    gate_cells = np.flatnonzero(nes.column.type[1:] == gate_type) + 1 if gate else np.empty(0,dtype=int)
    record.data['gate_cell_count'] = len(gate_cells)
    record.data['station_cells'] = station_cells
    coefficients = np.zeros((max(1, b_num), 2), dtype=np.float32)
    previous_coefficients = np.full_like(coefficients, np.nan)
    rain_index = -1
    rain_volume_rate = np.float64(0)
    pipe_end, pipe_pending = elapsed, False
    event_samples = {_seconds_to_ticks(float(t) + origin_before_start): float(t)
                     for t in forcing.events if -origin_before_start <= t <= end_time}
    event_ticks = np.array(sorted(event_samples), dtype=np.uint64)
    stats_file = (Path(domain_cfg.domain_dir) / 'step_statistics.csv').open('w', newline='')
    stats_fields = ['time', 'elapsed_seconds', 'steps', 'standard_dt_min_s', 'standard_dt_max_s', 'standard_dt_mean_s',
                    'standard_steps', 'event_short_steps', 'event_dt_min_s', 'event_dt_sum_s', 'cfl_min_s',
                    'limited_cell_steps', 'limited_face_steps', 'minimum_donor_scale',
                    'limiting_edge', 'lower_cell', 'higher_cell', 'lower_depth_m',
                    'higher_depth_m', 'face_depth_m', 'face_velocity_m_s']
    stats_writer = csv.DictWriter(stats_file, fieldnames=stats_fields)
    stats_writer.writeheader()
    interval_dts, interval_event_dts = [], []
    interval_activity = [0, 0, 1.]
    total_activity = [0, 0, 1.]
    limit_value_t[0] = float('inf')
    gate_has_opened = bool(state.gate_open) if state else False
    gate_tick = (None if forcing.gate_opening_time is None else
                 _seconds_to_ticks(forcing.gate_opening_time + origin_before_start))
    gate_levels = np.zeros(max(1,b_num),dtype='float32')
    dt_hasher = hashlib.sha256()
    q_exact = np.zeros(b_num, dtype=np.float64)
    # At a batch boundary Python uploads the active linear segment only once.
    def upload_forcing(seconds, warming):
        nonlocal rain_index, rain_volume_rate
        # A quantized clock can be half a tick before the original knot.
        # Select the new segment using that knot's exact input time.
        sample = 0. if warming else event_samples.get(_seconds_to_ticks(clock.host_now_seconds), seconds)
        for bi, b in enumerate(forcing.boundaries):
            i = min(int(np.searchsorted(b.series.times, sample, side='right')) - 1,
                    len(b.series.times) - 2)
            coefficients[bi, 0] = b.series.at(sample)
            coefficients[bi, 1] = 0. if warming else ((b.series.values[i+1] - b.series.values[i]) /
                                                       (b.series.times[i+1] - b.series.times[i]))
        if not np.array_equal(previous_coefficients, coefficients):
            coefficients_t.from_numpy(coefficients)
            previous_coefficients[:] = coefficients
        raining_t[None] = int(not warming)
        if not warming:
            index = forcing.rain.index_at(sample)
            if index != rain_index:
                rates = forcing.rain.rates(index)
                rain_rates_t.from_numpy(rates.astype('float32'))
                rain_volume_rate = np.sum(rates[forcing.rain.pixel_index[1:]] * mesh.area[1:], dtype=np.float64)
                rain_index = index

    def process_gate_event(seconds, *, initial=False):
        nonlocal gate_has_opened
        if gate_tick is None or gate_has_opened:
            return False
        now_tick = _seconds_to_ticks(clock.host_now_seconds)
        if now_tick < gate_tick:
            return False
        if seconds == end_time and forcing.gate_opening_time == end_time:
            reference = next(b for b in forcing.boundaries if b.name == gate.level_boundary)
            if reference.series.at(max(0.,seconds)) <= gate.threshold:
                return False
        impose_levels = not initial or continuation
        if impose_levels:
            sample = 0. if seconds < 0. else event_samples.get(now_tick,seconds)
            for bi,b in enumerate(forcing.boundaries):
                if b.kind == 'level': gate_levels[bi] = b.series.at(sample)
            level_values_t.from_numpy(gate_levels)
        diagnostic_t.fill(0)
        apply_gate_event(int(impose_levels))
        device_budget.collect(); device_budget.step_finished()
        diagnostic_t.fill(0)
        bed_np[gate_cells] = 0.
        gate_has_opened = True
        record.data['bed_events'].append(dict(elapsed_seconds=seconds,
            time=(domain_cfg.start+timedelta(seconds=seconds)).isoformat(),
            cell_ids=gate_cells.tolist(), new_bed=np.zeros(len(gate_cells)).tolist()))
        record.data['gate_open'] = True
        record.write()
        if not initial:
            initial_diagnostics()
            clock.active[None] = 1
            compute_cfl()
            supported_dt()
        return True

    def checkpoint(path, depth_np):
        nonlocal last_checkpoint_wall
        metadata = dict(state_identity, run_state=True,
            boundary_identity=boundary_identity,
            clock_origin_before_start_s=origin_before_start,
            physics_config={'start': domain_cfg.start.isoformat(), 'afa': afa, 'sita': sita,
                            'min_h': min_h, 'land_parameters': domain_cfg.model_dump(mode='json')['land_parameters']},
            total_steps=previous_steps+steps, total_warmup_steps=previous_warmup_steps+warmup_steps,
            elapsed_seconds=elapsed,
            budget={'initial_storage_m3': budget.initial, 'rainfall_m3': float(budget.rain),
                    'requested_discharge_m3': budget.discharge.tolist(),
                    'device_units': device_budget.total_units})
        tick_count = _seconds_to_ticks(clock.host_now_seconds)
        write_checkpoint(path, depth=depth_np, bed=ez_t.to_numpy(), q=sq_t.to_numpy(),
            rain_time=cumulative_rain_time_t.to_numpy(), infiltration=infiltration_t.to_numpy(),
            clock_hi=tick_count >> 32, clock_lo=tick_count & 0xffffffff,
            gate_open=bool(gate_open[None]), metadata_json=json.dumps(metadata),
            max_depth=max_depth_t.to_numpy(), first_time_hi=peak_hi_t.to_numpy(),
            first_time_lo=peak_lo_t.to_numpy(), max_speed=max_speed_t.to_numpy())
        last_checkpoint_wall = perf_counter()

    def write_output(seconds, phase, *, frame=False, final=False):
        date = domain_cfg.start + timedelta(seconds=seconds)
        budget.set_device_terms(device_budget.flush())
        depth_np = depth_t.to_numpy()
        storage = float(np.sum(depth_np.astype(float) * mesh.area, dtype=np.float64))
        row = budget.output(date, seconds, storage, phase)
        if station_count:
            sample_stations()
            samples = station_sample_t.to_numpy()
            append_station_samples(root / 'stations.csv', elapsed_seconds=seconds, time=date.isoformat(),
                samples={name: dict(cell_id=station_cells[name], **dict(zip(('depth','level','u','v'), map(float,samples[i]))))
                         for i,name in enumerate(station_names)})
        if frame:
            write_depth_frame(Path(domain_cfg.frame_dir) / f'depth_{int(seconds)}.fdb', depth_np)
        if seconds == 0. or final:
            checkpoint(root / ('state_final.fdb' if final else 'state_start.fdb'), depth_np)
        elif frame and perf_counter() - last_checkpoint_wall >= _CHECKPOINT_INTERVAL_S:
            # write_checkpoint atomically replaces the file from a sibling temporary.
            checkpoint(root / 'state_latest.fdb', depth_np)
        limits, ids = limit_value_t.to_numpy(), limit_ids_t.to_numpy()
        regular, shortened = np.array(interval_dts), np.array(interval_event_dts)
        stats_writer.writerow(dict(time=date.isoformat(), elapsed_seconds=seconds,
            steps=len(regular)+len(shortened), standard_steps=len(regular), standard_dt_min_s=float(regular.min()) if len(regular) else '',
            standard_dt_max_s=float(regular.max()) if len(regular) else '', standard_dt_mean_s=float(regular.mean()) if len(regular) else '',
            event_short_steps=len(shortened), event_dt_min_s=float(shortened.min()) if len(shortened) else '',
            event_dt_sum_s=float(shortened.sum()) if len(shortened) else 0.,
            limited_cell_steps=interval_activity[0], limited_face_steps=interval_activity[1],
            minimum_donor_scale=interval_activity[2],
            cfl_min_s=float(limits[0]), limiting_edge=int(ids[0]), lower_cell=int(ids[1]), higher_cell=int(ids[2]),
            lower_depth_m=float(limits[1]), higher_depth_m=float(limits[2]), face_depth_m=float(limits[3]),
            face_velocity_m_s=float(limits[4])))
        stats_file.flush()
        interval_dts.clear(); interval_event_dts.clear()
        interval_activity[:] = [0, 0, 1.]
        limit_value_t[0] = float('inf')
        print(f't={seconds:.6f}s phase={phase} storage={storage:.9g} residual={row["residual_m3"]:.9g} m3', flush=True)

    process_gate_event(elapsed, initial=True)
    if state is None and domain_cfg.warmup_duration > 0:
        initial_depth, initialization = connected_still_water(
            bed_np, dh_t.to_numpy(), bd_db[U8Value]['sbf'].column.value,
            ns_db[SideTopoInfo][SideTopoInfo].column.info.reshape(-1, 3),
            forcing.boundaries, mesh.area)
        depth_t.from_numpy(initial_depth)
        record.data['initialization'] = dict(method='connected_still_water', **initialization)
    else:
        record.data['initialization'] = dict(method='checkpoint' if state else 'dry')
    write_bed(root / 'bed.fdb', state.bed if state else nes.column.z)
    initial_depth = depth_t.to_numpy()
    initial_storage = float(np.sum(initial_depth.astype(float) * mesh.area, dtype=np.float64))
    budget = WaterBudget(Path(domain_cfg.domain_dir) / 'mass_balance.csv', forcing.boundaries, initial_storage)
    if continuation:
        saved = state_meta['budget']
        budget.initial = saved['initial_storage_m3']
        budget.rain = np.float64(saved['rainfall_m3'])
        budget.discharge[:] = saved['requested_discharge_m3']
        device_budget.total_units = list(saved['device_units'])
    if not continuation:
        targets = np.zeros(e_num, dtype='float32')
        for bi, b in enumerate(forcing.boundaries):
            if b.kind == 'level':
                targets[b.cells] = np.maximum(b.series.at(0.) - bed_np[b.cells].astype(float), 0.)
        level_initial_t.from_numpy(targets)
        apply_initial_levels()
        device_budget.collect(); device_budget.step_finished()
    clock.active[None] = 1
    compute_cfl()
    initial_diagnostics()
    if elapsed == 0. and end_time > 0.:
        reset_extrema()
    # Graph compilation lowers the same ordered global passes as plain calls.
    builder = ti.graph.GraphBuilder()
    for _ in range(clock.batch_size):
        builder.dispatch(prepare_step)
        builder.dispatch(tick)
        builder.dispatch(compute_cfl)
        builder.dispatch(finish_step)
    graph = builder.compile()
    ti.sync()
    phase_started = perf_counter()
    last_checkpoint_wall = phase_started
    preprocess_seconds = phase_started - record.clock
    warmup_seconds = 0.
    formal_started = phase_started if elapsed >= 0. and end_time > elapsed else None
    write_output(elapsed, 'warmup' if elapsed < 0 or end_time == 0 else 'simulation', frame=True,
                 final=elapsed == end_time)
    try:
        while elapsed < end_time:
            warming = elapsed < 0.
            if coupling is not None and elapsed == pipe_end:
                if pipe_pending:
                    coupling.end_window(pipe_volume_t.to_numpy().astype(float))
                    pipe_volume_t.fill(0); pipe_correction_t.fill(0)
                window_end = min(elapsed + coupling.pipe_cfg.coupling_interval, 0. if warming else end_time)
                source = coupling.begin_window(elapsed, window_end-elapsed,
                    depth_t.to_numpy(), ez_t.to_numpy(), mesh.area, nes.column.type.copy())
                pipe_q_t.from_numpy(source)
                pipe_pending, pipe_end = True, window_end
            target = min(next_output, next_frame, 0. if warming else end_time)
            if not warming:
                now_tick = _seconds_to_ticks(clock.host_now_seconds)
                i = np.searchsorted(event_ticks, now_tick, side='right')
                if i < len(event_ticks):
                    target = min(target, int(event_ticks[i])/2**40-origin_before_start)
            if coupling is not None:
                target = min(target, pipe_end)
            upload_forcing(elapsed, warming)
            batch_slot_t[None] = 0
            clock.begin(target + origin_before_start, clock.host_now_seconds)
            graph.run({})
            rows = clock.read_trace()
            for bits, hi, lo, status, cfl_bits, event_short, limited_cells, limited_faces, scale_bits in rows:
                if int(status) == clock.DONE:
                    break
                if int(status) != clock.COMPLETED:
                    if int(status) == clock.BAD_CFL:
                        supported_dt()
                    raise RuntimeError(f'Device clock status {int(status)} at {elapsed}s')
                scale = float(np.array(scale_bits, dtype=np.uint32).view(np.float32))
                for activity in (interval_activity, total_activity):
                    activity[0] += int(limited_cells)
                    activity[1] += int(limited_faces)
                    activity[2] = min(activity[2], scale)
                dt = float(np.array(bits,dtype=np.uint32).view(np.float32))
                for bi, b in enumerate(forcing.boundaries):
                    q_exact[bi] = b.series.at(0. if warming else elapsed+dt/2.)
                q_values = q_exact
                budget.add_forcing(dt, 0. if warming else rain_volume_rate, q_values)
                elapsed = (int(hi)*2**32+int(lo))/2**40-origin_before_start
                steps += 1; warmup_steps += int(warming)
                min_dt, max_dt, integrated_dt = min(min_dt,dt), max(max_dt,dt), integrated_dt+dt
                np.maximum(discharge_peaks, q_values, out=discharge_peaks)
                (interval_event_dts if event_short else interval_dts).append(dt)

                device_budget.step_finished()
            completed_bits = rows[rows[:,3] == clock.COMPLETED,0].astype('uint32')
            dt_hasher.update(completed_bits.tobytes())
            process_gate_event(elapsed)
            is_output = abs(elapsed-next_output) < 1e-9 or elapsed == end_time
            is_frame = abs(elapsed-next_frame) < 1e-9 or elapsed == end_time
            if is_output or is_frame:
                if warming and elapsed == 0. and end_time > 0.:
                    reset_extrema()
                write_output(elapsed,'warmup' if elapsed < 0 or end_time == 0 else 'simulation',
                             frame=is_frame, final=elapsed == end_time)
                if warming and elapsed == 0.:
                    warmup_seconds = perf_counter()-phase_started
                    if end_time > 0.:
                        formal_started = perf_counter()
                if is_output:
                    next_output = min(next_output+domain_cfg.yield_step,0. if elapsed < 0 else end_time)
                if is_frame:
                    next_frame = min(next_frame+900,0. if elapsed < 0 else end_time)
        if coupling is not None and pipe_pending:
            coupling.end_window(pipe_volume_t.to_numpy().astype(float))
            record.data['coupling'] = coupling.metadata()
        write_extrema(root / 'extrema.fdb',max_depth_t.to_numpy(),peak_hi_t.to_numpy(),peak_lo_t.to_numpy(),max_speed_t.to_numpy())
        record.data.update(steps=steps,warmup_steps=warmup_steps,mass_balance_final=budget.last_row,
            dt_seconds={'min':min_dt if steps else 0.,'max':max_dt,'sum':integrated_dt},
            dt_sequence_sha256=dt_hasher.hexdigest(),
            limiter_activity=dict(limited_cell_steps=total_activity[0], limited_face_steps=total_activity[1],
                                  minimum_donor_scale=total_activity[2]),
            requested_discharge_peak_m3_s={b.name:float(discharge_peaks[i]) for i,b in enumerate(forcing.boundaries) if b.kind=='discharge'},
            gate_open=bool(gate_open[None]),storage_depth_one_ulp_volume_m3=float(np.sum(
                np.abs(np.spacing(depth_t.to_numpy()[1:]))*mesh.area[1:],dtype=np.float64)))
        phase_finished = perf_counter()
        if end_time == 0.:
            warmup_seconds = phase_finished-phase_started
        formal_seconds = phase_finished-formal_started if formal_started is not None else 0.
        record.data['timing_seconds'] = dict(preprocess_and_setup=preprocess_seconds,warmup=warmup_seconds,
            simulation=formal_seconds,warmup_per_step=warmup_seconds/warmup_steps if warmup_steps else None,
            simulation_per_step=formal_seconds/(steps-warmup_steps) if steps>warmup_steps else None)
        record.data['budget_accumulator']['output_transfers'] = device_budget.transfer_count
    finally:
        budget.close(); stats_file.close()
    return budget.last_row


def warmup_solver(domain_cfg: DomainConfig, force_cfg: ForceConfig, pipe_cfg=None):
    """Run an already preprocessed domain through warmup, stopping at start."""
    with RunRecord(domain_cfg, force_cfg, pipe_cfg) as record:
        record.data.update(run_kind='warmup', simulation_end=domain_cfg.start.isoformat())
        record.hash_inputs(domain_cfg, force_cfg, pipe_cfg)
        return _solve(domain_cfg, force_cfg, record, pipe_cfg, end_time=0.)
