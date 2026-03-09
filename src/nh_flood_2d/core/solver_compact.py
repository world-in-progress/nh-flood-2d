import gc
import shutil
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from datetime import datetime
from typing import no_type_check

from ..util import benchmark
from ..output.hydrograph import _find_ei
from ..input import DomainConfig, ForceConfig
from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import Ne, Ns, IndexLike, SideTopoInfo, Rainfall, Tide, Gate, U8Value, UVH

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

@benchmark(applied=True)
def solver(domain_cfg: DomainConfig, force_cfg: ForceConfig, start_time_step: int = 0):
    init_taichi(use_gpu=True, profiler=True)
    
    inflow_ei = _find_ei(domain_cfg.ne_fdb, domain_cfg.ns_fdb, 827040.3, 843912.8)
    # set_elevation(domain_cfg, elevate_meter=3.0)

    # Check fdbs
    ne_fdb_fn = Path(domain_cfg.ne_fdb)
    ns_fdb_fn = Path(domain_cfg.ns_fdb)
    rain_fdb_fn = Path(force_cfg.rain_fdb)
    tide_fdb_fn = Path(force_cfg.tide_fdb)
    gate_fdb_fn = Path(force_cfg.gate_fdb)
    boundary_fdb_fn = Path(domain_cfg.boundary_fdb)

    # Load fdbs
    ne_fdb = fdb.ORM.load(str(ne_fdb_fn), from_file=True)
    ns_fdb = fdb.ORM.load(str(ns_fdb_fn), from_file=True)
    tide_fdb = fdb.ORM.load(str(tide_fdb_fn), from_file=True)
    gate_fdb = fdb.ORM.load(str(gate_fdb_fn), from_file=True)
    rain_fdb = fdb.ORM.load(str(rain_fdb_fn), from_file=True)
    boundary_fdb = fdb.ORM.load(str(boundary_fdb_fn), from_file=True)

    # Load fdb feature tables
    nes = ne_fdb[Ne][Ne]
    nss = ns_fdb[Ns][Ns]
    tides = tide_fdb[Tide][Tide]
    gates = gate_fdb[Gate][Gate]
    sbfs = boundary_fdb[U8Value]['sbf']
    bdeis = boundary_fdb[IndexLike]['bdei']
    rainfalls = rain_fdb[Rainfall][Rainfall]
    sts = ns_fdb[SideTopoInfo][SideTopoInfo]

    # Extract tide and rainfall columns as owned copies for FDB independence
    tts: np.ndarray = tides.column.time.copy()
    tls: np.ndarray = tides.column.level.copy()
    rts: np.ndarray = rainfalls.column.time.copy()
    rqs: np.ndarray = rainfalls.column.quantity.copy()

    # Lengths and counts
    e_num = len(nes)                                                # number of hydro elements
    s_num = len(nss)                                                # number of hydro sides
    b_num = len(bdeis)                                              # number of boundary elements
    t_num = len(tides)                                              # number of tide records
    r_num = len(rainfalls)                                          # number of rainfall records
    g_num = gate_fdb[IndexLike]['gate_count'][0].index              # number of gates

    # Taichi physical parameters and constants
    n = 0.033                                                       # Manning's roughness coefficient
    g = 9.81                                                        # gravitational acceleration (m/s²)
    pi = 3.141592653589793                                          # value of pi
    afa = domain_cfg.afa                                            # Courant number (CFL condition)
    sita = domain_cfg.sita                                          # time weighting factor
    min_h = domain_cfg.min_h                                        # minimum water depth (m)

    # Taichi fields about hydro elements
    esl_t = ti.field(dtype=ti.f32, shape=e_num)                     # side length of each hydro element
    eq_t = ti.field(dtype=ti.f32, shape=(e_num, 4))                 # flow quantities from all four sides of each hydro element
    enq_t = ti.field(dtype=ti.f32, shape=(e_num, 4))                # next time step flow quantities from all four sides of each hydro element
    ez_t = copy_to_taichi(nes.column.z, ti.f32, None)
    eu_t = copy_to_taichi(nes.column.type, ti.u8, None)             # underlay type of each hydro element, this field will be transformed to type flag field in init_gpu()
    bdei_t = copy_to_taichi(bdeis.column.index, ti.i32, None)       # as index, taichi must use i32 as iterator index type

    # Compact side-index fields (CSR-like layout produced by create_ne_fdb_compact)
    #   isl_data_t   : flat array of all side indices for all elements
    #   isl_ptr_l/r/b/top_t : per-element start position in isl_data for each direction
    isl_data_t    = copy_to_taichi(ne_fdb[IndexLike]['isl_data'].column.index,  ti.i32, None)
    isl_ptr_l_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_l'].column.index, ti.i32, None)
    isl_ptr_r_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_r'].column.index, ti.i32, None)
    isl_ptr_b_t   = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_b'].column.index, ti.i32, None)
    isl_ptr_top_t = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_t'].column.index, ti.i32, None)

    # Taichi fields about hydro sides
    ndt_t = ti.field(dtype=ti.f32, shape=())                        # next global time step
    dh_t = ti.field(dtype=ti.f32, shape=s_num)                      # dike height at each hydro side
    sq_t = ti.field(dtype=ti.f32, shape=s_num)                      # storage quantity at each hydro side (both in direction x and y)
    sqn_t = ti.field(dtype=ti.f32, shape=s_num)                     # next time step storage quantity at each hydro side (both in direction x and y)
    sdc_t = ti.field(dtype=ti.f32, shape=s_num)                     # length between two hydro element centers at each hydro side
    sl_t = copy_to_taichi(nss.column.length, ti.f32, None)
    sbf_t = copy_to_taichi(sbfs.column.value, ti.u8, None)
    sts_t = copy_to_taichi(sts.column.info, ti.i32, [s_num, 3])

    # Taichi fields about gates
    gate_t = copy_to_taichi(gates.column.info[:g_num * 100], ti.i32, [g_num, 100])

    # Taichi fields about hydrodynamic model
    u_t = ti.field(dtype=ti.f32, shape=e_num)                       # horizontal velocity at current time step
    v_t = ti.field(dtype=ti.f32, shape=e_num)                       # vertical velocity at current time step
    h_t = ti.field(dtype=ti.f32, shape=e_num)                       # water depth at current time step
    depth_t = ti.field(dtype=ti.f32, shape=e_num)                   # water depth (h - z) at current time step
    ssq_t = ti.field(dtype=ti.f32, shape=e_num)                     # quantity of source / sink term at current time step

    # Taichi rainning time counters
    cumulative_rain_time_t = ti.field(dtype=ti.f32, shape=())

    # Infiltration rate
    fr1 = ti.field(dtype=ti.f32, shape=())   # building
    fr2 = ti.field(dtype=ti.f32, shape=())   # road
    fr3 = ti.field(dtype=ti.f32, shape=())   # agricultural land
    fr4 = ti.field(dtype=ti.f32, shape=())   # fish pond
    fr5 = ti.field(dtype=ti.f32, shape=())   # mountainous land
    fr6 = ti.field(dtype=ti.f32, shape=())   # water body
    fr7 = ti.field(dtype=ti.f32, shape=())   # catch basin

    # Time counters
    current_time = 0.0
    current_rain_idx = 0
    current_tide_idx = 0
    simulation_begin_time = 0.0

    @ti.kernel
    @no_type_check
    def init_gpu(ex_t: ti.template(), ey_t: ti.template(),
                 isl_data: ti.template(), isl_ptr_l: ti.template(),
                 sx_t: ti.template()):
        ndt_t[None] = 0.1
        cumulative_rain_time_t[None] = 0.0

        for ei in range(1, e_num):
            u_t[ei] = 0.0
            v_t[ei] = 0.0
            ssq_t[ei] = 0.0
            h_t[ei] = ez_t[ei] if ez_t[ei] > 0.0 else 0.0
            eq_t[ei, 0] = eq_t[ei, 1] = eq_t[ei, 2] = eq_t[ei, 3] = 0.0
            enq_t[ei, 0] = enq_t[ei, 1] = enq_t[ei, 2] = enq_t[ei, 3] = 0.0

            # Get side length (all four sides are the same for square elements)
            lsi0 = isl_data[isl_ptr_l[ei]]
            esl_t[ei] = ti.max((ex_t[ei] - sx_t[lsi0]) * 2.0, 0.0001)
            eu_t[ei] = 0b1 << (eu_t[ei] - 1)    # set type flag bit

        for si in range(1, s_num):
            sq_t[si] = 0.0
            sqn_t[si] = 0.0
            dh_t[si] = -999.0
            so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]   # orient, lower element, higher element
            if so == 2:
                eil, eir = el, eh
                sdc_t[si] = ti.max(ti.abs(ex_t[eir] - ex_t[eil]), 0.01)
            else:
                eib, eit = el, eh
                sdc_t[si] = ti.max(ti.abs(ey_t[eit] - ey_t[eib]), 0.01)

        fr1[None] = fr2[None] = fr3[None] = fr4[None] = fr5[None] = fr6[None] = fr7[None] = 0.0

    def init():
        nonlocal current_time, simulation_begin_time, current_rain_idx

        tide_step = round((tts[1] - tts[0]) / 60.0)  # time step in minutes
        begin_index = max(0, (int(start_time_step * 5 / tide_step) - 1))
        current_time = tts[begin_index]
        simulation_begin_time = current_time

        while rts[current_rain_idx] < simulation_begin_time:
            current_rain_idx += 1

        ex = copy_to_taichi(nes.column.x, ti.f32, None)
        ey = copy_to_taichi(nes.column.y, ti.f32, None)
        sx = copy_to_taichi(nss.column.x, ti.f32, None)
        init_gpu(ex, ey, isl_data_t, isl_ptr_l_t, sx)

    @ti.kernel
    @no_type_check
    def tick(tide: float, rainq: float) -> ti.f32:
        # Tick dt
        dt = ti.max(0.0001, ti.min(ndt_t[None], 1.0))    # clamp dt to avoid instability
        ndt_t[None] = 1000.0

        # Tick sides
        for si in range(1, s_num):
            bf = ti.select(sbf_t[si] == 1, 0.0, 1.0)  # boundary factor
            so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]   # orient, lower element, higher element
            # Handle flow in direction x (Type 2: Vertical side, connects Left/Right)
            if so == 2:
                eil, eir = el, eh
                side_l = isl_data_t[isl_ptr_l_t[eil]]      # first left side of the left element
                side_r = isl_data_t[isl_ptr_r_t[eir]]      # first right side of the right element
                hl, hr = h_t[eil], h_t[eir]
                dx = sdc_t[si]
                xq = sq_t[si]
                xwh = ti.max(hl, hr) - ti.max(ez_t[eil], ez_t[eir], dh_t[si]) # water height in direction x
                # Calculate the unit discharge quantity caused by pressure gradient in direction x
                xdq = (-g * (hr - hl) / dx) * ti.max(xwh, 0.0) * dt
                # Calculate the friction term in direction x (Manning's formula)
                xf = 1.0 + g * dt * (n ** 2) * ti.abs(xq / (ti.max(xwh, 0.00001) ** (7.0 / 3.0)))
                new_xq = (sita * xq + (1.0 - sita) / 2.0 * (sqn_t[side_l] + sqn_t[side_r]) + xdq) / xf
                new_xq = ti.select(xwh < min_h, 0.0, new_xq)    # cutoff small flow
                new_xq *= bf                                    # zero flow for boundary sides
                sq_t[si] = new_xq                               # update current flow quantity in direction x
                sqn_t[si] = new_xq                              # update next time step flow quantity in direction x (for use in the next tick)
                xwh = ti.max(xwh, 0.01)
                sdt = bf * afa * sl_t[si] / (ti.sqrt(g * xwh) + ti.abs(new_xq) / xwh) # CFL Condition: Ignore boundary sides (connected to element 0)
                sdt = ti.max((0.001 - sdt) * 100000.0, sdt)  # avoid too small time step
                ti.atomic_min(ndt_t[None], sdt)  # update global next time step
                flux = new_xq * sl_t[si]
                ti.atomic_add(enq_t[eil, 1], flux)
                ti.atomic_add(enq_t[eir, 0], flux)
            # Handle flow in direction y (Type 1: Horizontal side, connects Bottom/Top)
            else:
                eib, eit = el, eh
                side_b = isl_data_t[isl_ptr_b_t[eib]]      # first bottom side of the bottom element
                side_t = isl_data_t[isl_ptr_top_t[eit]]    # first top side of the top element
                hb, ht = h_t[eib], h_t[eit]
                dy = sdc_t[si]
                yq = sq_t[si]
                ywh = ti.max(hb, ht) - ti.max(ez_t[eib], ez_t[eit], dh_t[si]) # water height in direction y
                # Calculate the unit discharge quantity caused by pressure gradient in direction y
                ydq = (-g * (ht - hb) / dy) * ti.max(ywh, 0.0) * dt
                # Calculate the friction term in direction y (Manning's formula)
                yf = 1.0 + g * dt * (n ** 2) * ti.abs(yq / (ti.max(ywh, 0.00001) ** (7.0 / 3.0)))
                new_yq = (sita * yq + (1.0 - sita) / 2.0 * (sqn_t[side_b] + sqn_t[side_t]) + ydq) / yf
                new_yq = ti.select(ywh < min_h, 0.0, new_yq)    # cutoff small flow
                new_yq *= bf                                    # zero flow for boundary sides
                sq_t[si] = new_yq                               # update current flow quantity in direction y
                sqn_t[si] = new_yq                              # update next time step flow quantity in direction y (for use in the next tick)
                ywh = ti.max(ywh, 0.01)
                sdt = bf * afa * sl_t[si] / (ti.sqrt(g * ywh) + ti.abs(new_yq) / ywh) # CFL Condition: Ignore boundary sides (connected to element 0)
                sdt = ti.max((0.001 - sdt) * 100000.0, sdt)  # avoid too small time step
                ti.atomic_min(ndt_t[None], sdt)  # update global next time step
                flux = new_yq * sl_t[si]
                ti.atomic_add(enq_t[eib, 3], flux)
                ti.atomic_add(enq_t[eit, 2], flux)

        # Tick infiltration rate during rainfall (based on Horton model)
        if rainq > 0.0:
            cumulative_rain_time_t[None] += dt
            fr3[None] = fr5[None] = fr7[None] = horton_decay(3.0, 0.1, 2.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
            fr1[None] = fr2[None] = horton_decay(0.8, 0.02, 10.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0

        # Tick elements
        for ei in range(1, e_num):
            # # If is inflow element, add source/sink quantity to ssq_t
            # q_source = 0.0
            # if ei == inflow_ei:
            #     q_source = rainq * 1000 * 300    # inflow quantity (m³/s)
                
            # Calculate flow quantities
            ql = enq_t[ei, 0]
            qr = enq_t[ei, 1]
            qb = enq_t[ei, 2]
            qt = enq_t[ei, 3]
            # tq = ql - qr + qb - qt + ssq_t[ei] + q_source                       # total inflow quantity
            tq = ql - qr + qb - qt + ssq_t[ei]                                  # total inflow quantity
            eq_t[ei, 0], eq_t[ei, 1], eq_t[ei, 2], eq_t[ei, 3] = ql, qr, qb, qt # update current side flow quantities
            enq_t[ei, 0] = enq_t[ei, 1] = enq_t[ei, 2] = enq_t[ei, 3] = 0.0     # reset next time step side flow quantities

            # Calculate infiltration masks for all underlay types
            eu = eu_t[ei]       # underlay type flag
            f1 = (eu >> 0) & 1  # building
            f2 = (eu >> 1) & 1  # road
            f3 = (eu >> 2) & 1  # agricultural land
            f4 = (eu >> 3) & 1  # fish pond
            f5 = (eu >> 4) & 1  # mountainous land
            f6 = (eu >> 5) & 1  # water body
            f7 = (eu >> 6) & 1  # catch basin

            ea = esl_t[ei] ** 2                 # area of element
            next_h = h_t[ei] + (tq * dt) / ea   # update next water depth
            next_h += rainq * dt                # add rainfall effect
            next_h -= (fr1[None] * f1 + fr2[None] * f2 + fr3[None] * f3 +
                       fr4[None] * f4 + fr5[None] * f5 + fr6[None] * f6 +
                       fr7[None] * f7) * dt     # subtract infiltration effect
            next_h = ti.max(next_h, ez_t[ei])   # water depth cannot be lower than ground elevation
            h_t[ei] = next_h                    # update current water depth
            depth_t[ei] = ti.max(next_h - ez_t[ei], 0.00)     # update current water depth (h - z)

        # Tick boundaries
        for count in range(b_num):
            bdei = bdei_t[count]
            h_t[bdei] = tide

        # Output dt
        return ndt_t[None]

    @ti.kernel
    @no_type_check
    def update_velocities():
        for ei in range(1, e_num):
            esl = esl_t[ei]
            depth = ti.max(h_t[ei] - ez_t[ei], 0.01)
            u_t[ei] = ti.select(
                depth < min_h, 0.0,
                (eq_t[ei, 0] + eq_t[ei, 1]) / esl / depth / 2.0
            )
            v_t[ei] = ti.select(
                depth < min_h, 0.0,
                (eq_t[ei, 2] + eq_t[ei, 3]) / esl / depth / 2.0
            )

    # Main logic here ##################################################
    init()

    print(f'Initial time step: {ndt_t[None]} seconds')
    # Free FDB objects — all data has been copied to Taichi GPU fields
    # or independent numpy arrays (tts/tls/rts/rqs) by this point.
    del ne_fdb, ns_fdb, tide_fdb, gate_fdb, rain_fdb, boundary_fdb
    del nes, nss, tides, gates, sbfs, bdeis, rainfalls, sts
    gc.collect()
    print('FDB objects deleted, memory freed for simulation.')

    # Prepare for output
    last_output_time = current_time
    evolve_start_time = current_time
    output_uvh_fn = Path(domain_cfg.uvh_dir)
    if output_uvh_fn.exists():
        shutil.rmtree(output_uvh_fn)
    output_uvh_fn.mkdir(parents=True, exist_ok=True)

    # Main simulation loop
    while domain_cfg.duration == -1 or current_time - evolve_start_time < domain_cfg.duration:
        # Update tide by linear interpolation
        if current_time >= tts[current_tide_idx + 1]:
            if current_tide_idx + 2 >= t_num:
                break                   # end of simulation
            else:
                current_tide_idx += 1   # tick tide index
        tide = lerp(
            tls[current_tide_idx],
            tls[current_tide_idx + 1],
            (current_time - tts[current_tide_idx]) / (tts[current_tide_idx + 1] - tts[current_tide_idx])
        )

        # Update average rainfall quantity (m/s)
        rainq = 0.0
        if current_time <= rts[r_num - 1]: # still raining
            if current_time > rts[current_rain_idx + 1]:
                current_rain_idx += 1  # tick rainfall index
            rainq = rqs[current_rain_idx + 1] / (rts[current_rain_idx + 1] - rts[current_rain_idx]) * 0.001

        # Core solver
        dt = tick(tide, rainq)
        current_time += dt

        # Output cumulative time at each yield step
        if current_time - last_output_time >= domain_cfg.yield_step:
            last_output_time += domain_cfg.yield_step
            cumulative_time = current_time - simulation_begin_time
            print(f'Cumulative simulation time: {cumulative_time} seconds, current dt: {dt} seconds')

            # Update water velocities
            update_velocities()

            # Output uvh data
            uvh_db = fdb.ORM.truncate([
                fdb.TableDefn(UVH, e_num)
            ])
            us = uvh_db[UVH][UVH].column.u
            vs = uvh_db[UVH][UVH].column.v
            hs = uvh_db[UVH][UVH].column.h
            us[:] = u_t.to_numpy()
            vs[:] = v_t.to_numpy()
            hs[:] = h_t.to_numpy()

            # Save uvh fdb, name: uvh_{timestamp of last_output_time}.fdb
            time_str = datetime.fromtimestamp(last_output_time).strftime('%Y%m%d-%H%M%S')
            uvh_fn = output_uvh_fn / f'uvh_{time_str}.fdb'
            uvh_db.save(str(uvh_fn))
            uvh_db.unlink()

# Helpers ##################################################

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

@ti.func
def horton_decay(initial: float, final: float, k: float, t: float) -> float:
    return final + (initial - final) * ti.exp(-k * t)
