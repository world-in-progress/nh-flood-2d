import time
import shutil
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from typing import no_type_check

from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import IndexLike, Ne, SideTopoInfo, Ns, Rainfall, Tide, Gate, U8Value, UVH

def solver(fdb_dir: str, start_time_step: int = 0):
    init_taichi(use_gpu=True, profiler=True)
    
    # Check fdbs
    fdb_path = Path(fdb_dir)
    ne_fdb_fn = fdb_path / 'ne.fdb'
    ns_fdb_fn = fdb_path / 'ns.fdb'
    rain_fdb_fn = fdb_path / 'rain.fdb'
    tide_fdb_fn = fdb_path / 'tide.fdb'
    gate_fdb_fn = fdb_path / 'gate.fdb'
    boundary_fdb_fn = fdb_path / 'boundary.fdb'
    if not (ne_fdb_fn.exists() and ns_fdb_fn.exists() and 
            rain_fdb_fn.exists() and tide_fdb_fn.exists() and 
            gate_fdb_fn.exists() and boundary_fdb_fn.exists()):
        raise FileNotFoundError("One or more required FDB files are missing.")
    
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
    
    # Extract tide and rainfall columns from feature tables
    tts: np.ndarray = tides.column.time
    tls: np.ndarray = tides.column.level
    rts: np.ndarray = rainfalls.column.time
    rqs: np.ndarray = rainfalls.column.quantity
    
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
    afa = 0.5                                                       # Courant number (CFL condition)
    sita = 1.0                                                      # time weighting factor
    min_h = 0.02                                                    # minimum water depth (m)
    
    # Taichi fields about hydro elements
    esl_t = ti.field(dtype=ti.f32, shape=e_num)                     # side length of each hydro element
    eq_t = ti.field(dtype=ti.f32, shape=(e_num, 4))                 # flow quantities from all four sides of each hydro element
    enq_t = ti.field(dtype=ti.f32, shape=(e_num, 4))                # next time step flow quantities from all four sides of each hydro element  
    ez_t = copy_to_taichi(nes.column.z, ti.f32, None)
    eu_t = copy_to_taichi(nes.column.type, ti.u8, None)             # underlay type of each hydro element, this field will be transformed to type flag field in init_gpu()
    bdei_t = copy_to_taichi(bdeis.column.index, ti.i32, None)       # as index, taichi must use i32 as iterator index type
    
    # Taichi fields about hydro sides
    ndt_t = ti.field(dtype=ti.f32, shape=())                        # next global time step
    dh_t = ti.field(dtype=ti.f32, shape=s_num)                      # dike height at each hydro side
    sq_t = ti.field(dtype=ti.f32, shape=s_num)                      # storage quantity at each hydro side (both in direction x and y)
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
    # ssf_t = ti.field(dtype=ti.i32, shape=e_num)                   # flag of source / sink term at current time step
    
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
    def init_gpu(ex_t: ti.template(), ey_t: ti.template(), isl1: ti.template(), sx_t: ti.template()):
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
            lsi0 = isl1[ei, 0]
            esl_t[ei] = ti.max((ex_t[ei] - sx_t[lsi0]) * 2.0, 0.0001)
            eu_t[ei] = 0b1 << (eu_t[ei] - 1)    # set type flag bit
        
        for si in range(1, s_num):
            sq_t[si] = 0.0
            dh_t[si] = -999.0
            so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]   # orient, lower element, higher element
            if so == 2:
                eil, eir = el, eh
                sdc_t[si] = ti.max(ti.abs(ex_t[eir] - ex_t[eil]), 0.01)
            else:
                eib, eit = el, eh
                sdc_t[si] = ti.max(ti.abs(ey_t[eit] - ey_t[eib]), 0.01)
        
        # # Update infiltration with horton model
        # fr3[None] = fr5[None] = fr7[None] = horton_decay(3.0, 0.1, 2.0, 0.0) * 0.0254 / 3600.0
        # fr1[None] = fr2[None] = horton_decay(0.8, 0.02, 10.0, 0.0) * 0.0254 / 3600.0
        # fr4[None] = fr6[None] = 0.0
        fr1[None] = fr2[None] = fr3[None] = fr4[None] = fr5[None] = fr6[None] = fr7[None] = 0.0
            
    def init():
        nonlocal current_time, simulation_begin_time, current_rain_idx
        
        tide_step = round((tts[1] - tts[0]) / 60.0)  # time step in minutes
        begin_index = max(0, (int(start_time_step * 5 / tide_step) - 1))
        current_time = tts[begin_index]
        simulation_begin_time = current_time
        
        while rainfalls[current_rain_idx].time < simulation_begin_time:
            current_rain_idx += 1
            
        ex = copy_to_taichi(nes.column.x, ti.f32, None)
        ey = copy_to_taichi(nes.column.y, ti.f32, None)
        sx = copy_to_taichi(nss.column.x, ti.f32, None)
        isl1 = copy_to_taichi(ne_fdb[SideTopoInfo]['isl1'].column.info, ti.i32, [e_num, 10])
        init_gpu(ex, ey, isl1, sx)
    
    @ti.kernel
    @no_type_check
    def tick(tide: float, rainq: float) -> ti.f32:
        # Tick dt
        dt = ti.max(0.0001, ti.min(ndt_t[None], 1.0))    # clamp dt to avoid instability
        ndt_t[None] = 1000.0
        
        # Tick gates
        for gi in range(g_num):
            up, down, level = gate_t[gi, 0], gate_t[gi, 1], float(gate_t[gi, 2])
            should_open = h_t[up] + 0.1 > h_t[down]
            new_z = 0.0 if should_open else level
            for ei_count in range(3, 100):
                ei = gate_t[gi, ei_count]
                if ei == 0:
                    break
                ez_t[ei] = new_z
        
        # Tick sides
        for si in range(1, s_num):
            bf = ti.select(sbf_t[si] == 1, 0.0, 1.0)  # boundary factor
            so, el, eh = sts_t[si, 0], sts_t[si, 1], sts_t[si, 2]   # orient, lower element, higher element
            # Handle flow in direction x (Type 2: Vertical side, connects Left/Right)
            if so == 2:
                eil, eir = el, eh
                hl, hr = h_t[eil], h_t[eir]
                dx = sdc_t[si]
                xq = sq_t[si]
                xwh = ti.max(hl, hr) - ti.max(ez_t[eil], ez_t[eir], dh_t[si]) # water height in direction x
                # Calculate the unit discharge quantity caused by pressure gradient in direction x
                xdq = (-g * (hr - hl) / dx) * ti.max(xwh, 0.0) * dt
                # Calculate the friction term in direction x (Manning's formula)
                xf = 1.0 + g * dt * (n ** 2) * ti.abs(xq / (ti.max(xwh, 0.00001) ** (7.0 / 3.0)))
                new_xq = (sita * xq + (1.0 - sita) * (eq_t[eil, 0] / esl_t[eil] + eq_t[eir, 1] / esl_t[eir]) + xdq) / xf
                new_xq = ti.select(xwh < min_h, 0.0, new_xq)    # cutoff small flow
                new_xq *= bf                                    # zero flow for boundary sides
                sq_t[si] = new_xq                               # update current flow quantity in direction x
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
                hb, ht = h_t[eib], h_t[eit]
                dy = sdc_t[si]
                yq = sq_t[si]
                ywh = ti.max(hb, ht) - ti.max(ez_t[eib], ez_t[eit], dh_t[si]) # water height in direction y
                # Calculate the unit discharge quantity caused by pressure gradient in direction y
                ydq = (-g * (ht - hb) / dy) * ti.max(ywh, 0.0) * dt
                # Calculate the friction term in direction y (Manning's formula)
                yf = 1.0 + g * dt * (n ** 2) * ti.abs(yq / (ti.max(ywh, 0.00001) ** (7.0 / 3.0)))
                new_yq = (sita * yq + (1.0 - sita) * (eq_t[eib, 2] / esl_t[eib] + eq_t[eit, 3] / esl_t[eit]) + ydq) / yf
                new_yq = ti.select(ywh < min_h, 0.0, new_yq)    # cutoff small flow
                new_yq *= bf                                    # zero flow for boundary sides
                sq_t[si] = new_yq                               # update current flow quantity in direction y
                ywh = ti.max(ywh, 0.01)
                sdt = bf * afa * sl_t[si] / (ti.sqrt(g * ywh) + ti.abs(new_yq) / ywh) # CFL Condition: Ignore boundary sides (connected to element 0)
                sdt = ti.max((0.001 - sdt) * 100000.0, sdt)  # avoid too small time step
                ti.atomic_min(ndt_t[None], sdt)  # update global next time step
                flux = new_yq * sl_t[si]
                ti.atomic_add(enq_t[eib, 3], flux)
                ti.atomic_add(enq_t[eit, 2], flux)
        
        # Tick infiltration rate during rainfall (based on Horton model)
        # Green space / grassland infiltration rate (Types 3, 5, 7), initial 3 inches/hr, final 0.1 inches/hr
        # Infiltration rate of impermeable pavements (Types 1, 2), initial 0.8 inches/hr, final 0.02 inches/hr
        # Infiltration rate of impermeable water bodies (Types 4, 6), always 0.0
        # fr1 = fr2 = fr3 = fr4 = fr5 = fr6 = fr7 = 0.0
        if rainq > 0.0:
            cumulative_rain_time_t[None] += dt
            fr3[None] = fr5[None] = fr7[None] = horton_decay(3.0, 0.1, 2.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
            fr1[None] = fr2[None] = horton_decay(0.8, 0.02, 10.0, cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
        
        # Tick elements
        for ei in range(1, e_num):
            # Calculate flow quantities
            ql = enq_t[ei, 0]
            qr = enq_t[ei, 1]
            qb = enq_t[ei, 2]
            qt = enq_t[ei, 3]
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
    start_time = time.time()
    init()
    
    # Prepare for output
    last_output_count = 0
    last_output_time = current_time
    output_uvh_fn = fdb_path / 'uvh'
    if output_uvh_fn.exists():
        shutil.rmtree(output_uvh_fn)
    output_uvh_fn.mkdir(parents=True, exist_ok=True)
    
    # Main simulation loop
    while True:
        if last_output_count >= 144:    # 12 hours simulation
            break
        
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
        
        # Output cumulative time every 5 minutes
        if current_time - last_output_time >= 300.0:
            last_output_time += 300.0
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
            uvh_fn = output_uvh_fn / f'uvh_{last_output_count}.fdb'
            uvh_db.save(str(uvh_fn))
            uvh_db.unlink()
            last_output_count += 1
    print(f'Time profiling results: {time.time() - start_time} seconds')
            
# Helpers ##################################################

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

@ti.func
def horton_decay(initial: float, final: float, k: float, t: float) -> float:
    return final + (initial - final) * ti.exp(-k * t)

if __name__ == '__main__':
    solver('./fdb', start_time_step=0)