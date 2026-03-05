import shutil
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from typing import no_type_check, Iterator

from ..input import DomainConfig
from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import Ne, Ns, IndexLike, SideTopoInfo, Rainfall, Tide, Gate, U8Value, UVH

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

@ti.func
def horton_decay(initial: float, final: float, k: float, t: float) -> float:
    return final + (initial - final) * ti.exp(-k * t)

@ti.data_oriented
class Domain:
    def __init__(self, cfg: DomainConfig, start_time_step: int = 0):
        self.cfg = cfg
        self.start_time_step = start_time_step
        
        init_taichi(use_gpu=True, profiler=True) # Initialize taichi environment

        # Check fdbs
        ne_fdb_fn = Path(cfg.ne_fdb)
        ns_fdb_fn = Path(cfg.ns_fdb)
        rain_fdb_fn = Path(cfg.rain_fdb)
        tide_fdb_fn = Path(cfg.tide_fdb)
        gate_fdb_fn = Path(cfg.gate_fdb)
        boundary_fdb_fn = Path(cfg.boundary_fdb)
        
        if not (ne_fdb_fn.exists() and ns_fdb_fn.exists() and 
                rain_fdb_fn.exists() and tide_fdb_fn.exists() and 
                gate_fdb_fn.exists() and boundary_fdb_fn.exists()):
            raise FileNotFoundError('One or more required FDB files are missing.')
        
        # Load fdbs
        self.ne_fdb = fdb.ORM.load(str(ne_fdb_fn), from_file=True)
        self.ns_fdb = fdb.ORM.load(str(ns_fdb_fn), from_file=True)
        self.tide_fdb = fdb.ORM.load(str(tide_fdb_fn), from_file=True)
        self.gate_fdb = fdb.ORM.load(str(gate_fdb_fn), from_file=True)
        self.rain_fdb = fdb.ORM.load(str(rain_fdb_fn), from_file=True)
        self.boundary_fdb = fdb.ORM.load(str(boundary_fdb_fn), from_file=True)
        
        # Load fdb feature tables
        self.nes = self.ne_fdb[Ne][Ne]
        self.nss = self.ns_fdb[Ns][Ns]
        self.tides = self.tide_fdb[Tide][Tide]
        self.gates = self.gate_fdb[Gate][Gate]
        self.sbfs = self.boundary_fdb[U8Value]['sbf']
        self.bdeis = self.boundary_fdb[IndexLike]['bdei']
        self.rainfalls = self.rain_fdb[Rainfall][Rainfall]
        self.sts = self.ns_fdb[SideTopoInfo][SideTopoInfo]
        
        # Extract tide and rainfall columns
        self.tts : np.ndarray = self.tides.column.time
        self.tls : np.ndarray = self.tides.column.level
        self.rts : np.ndarray = self.rainfalls.column.time
        self.rqs : np.ndarray = self.rainfalls.column.quantity
        
        # Lengths and counts
        self.e_num = len(self.nes)                                                # number of hydro elements
        self.s_num = len(self.nss)                                                # number of hydro sides
        self.b_num = len(self.bdeis)                                              # number of boundary elements
        self.t_num = len(self.tides)                                              # number of tide records
        self.r_num = len(self.rainfalls)                                          # number of rainfall records
        self.g_num = self.gate_fdb[IndexLike]['gate_count'][0].index              # number of gates
        
        # Taichi physical parameters and constants
        self.n = 0.033                                                       # Manning's roughness coefficient
        # self.g = 9.81                                                      # gravitational acceleration (m/s²) - Used directly in kernel
        # self.pi = 3.141592653589793                                        # value of pi
        self.afa = 0.5                                                       # Courant number (CFL condition)
        self.sita = 1.0                                                      # time weighting factor
        self.min_h = 0.02                                                    # minimum water depth (m)
        
        # Taichi fields about hydro elements
        self.esl_t = ti.field(dtype=ti.f32, shape=self.e_num)                     # side length of each hydro element
        self.eq_t = ti.field(dtype=ti.f32, shape=(self.e_num, 4))                 # flow quantities from all four sides of each hydro element
        self.enq_t = ti.field(dtype=ti.f32, shape=(self.e_num, 4))                # next time step flow quantities from all four sides of each hydro element  
        self.ez_t = copy_to_taichi(self.nes.column.z, ti.f32, None)
        self.eu_t = copy_to_taichi(self.nes.column.type, ti.u8, None)             # underlay type of each hydro element
        self.bdei_t = copy_to_taichi(self.bdeis.column.index, ti.i32, None)       # as index, taichi must use i32 as iterator index type
        
        # Taichi fields about hydro sides
        self.ndt_t = ti.field(dtype=ti.f32, shape=())                        # next global time step
        self.dh_t = ti.field(dtype=ti.f32, shape=self.s_num)                      # dike height at each hydro side
        self.sq_t = ti.field(dtype=ti.f32, shape=self.s_num)                      # storage quantity at each hydro side (both in direction x and y)
        self.sdc_t = ti.field(dtype=ti.f32, shape=self.s_num)                     # length between two hydro element centers at each hydro side
        self.sl_t = copy_to_taichi(self.nss.column.length, ti.f32, None)
        self.sbf_t = copy_to_taichi(self.sbfs.column.value, ti.u8, None)
        self.sts_t = copy_to_taichi(self.sts.column.info, ti.i32, [self.s_num, 3])
        
        # Taichi fields about gates
        self.gate_t = copy_to_taichi(self.gates.column.info[:self.g_num * 100], ti.i32, [self.g_num, 100])
        
        # Taichi fields about hydrodynamic model
        self.u_t = ti.field(dtype=ti.f32, shape=self.e_num)                       # horizontal velocity at current time step
        self.v_t = ti.field(dtype=ti.f32, shape=self.e_num)                       # vertical velocity at current time step
        self.h_t = ti.field(dtype=ti.f32, shape=self.e_num)                       # water depth at current time step
        self.depth_t = ti.field(dtype=ti.f32, shape=self.e_num)                   # water depth (h - z) at current time step
        self.ssq_t = ti.field(dtype=ti.f32, shape=self.e_num)                     # quantity of source / sink term at current time step
        
        # Taichi rainning time counters
        self.cumulative_rain_time_t = ti.field(dtype=ti.f32, shape=())
        
        # Infiltration rate
        self.fr1 = ti.field(dtype=ti.f32, shape=())   # building
        self.fr2 = ti.field(dtype=ti.f32, shape=())   # road
        self.fr3 = ti.field(dtype=ti.f32, shape=())   # agricultural land
        self.fr4 = ti.field(dtype=ti.f32, shape=())   # fish pond
        self.fr5 = ti.field(dtype=ti.f32, shape=())   # mountainous land
        self.fr6 = ti.field(dtype=ti.f32, shape=())   # water body
        self.fr7 = ti.field(dtype=ti.f32, shape=())   # catch basin
        
        # Time counters
        self.current_time = 0.0
        self.current_rain_idx = 0
        self.current_tide_idx = 0
        self.evolve_start_time = 0.0
        
        # Initialization
        self._init_simulation()

    def _init_simulation(self):
        tide_step = round((self.tts[1] - self.tts[0]) / 60.0)  # time step in minutes
        begin_index = max(0, (int(self.start_time_step * 5 / tide_step) - 1))
        self.current_time = self.tts[begin_index]
        self.evolve_start_time = self.current_time
        
        while self.rainfalls[self.current_rain_idx].time < self.evolve_start_time:
            self.current_rain_idx += 1
            if self.current_rain_idx >= len(self.rainfalls):
                self.current_rain_idx = len(self.rainfalls) - 1
                break
            
        ex = copy_to_taichi(self.nes.column.x, ti.f32, None)
        ey = copy_to_taichi(self.nes.column.y, ti.f32, None)
        sx = copy_to_taichi(self.nss.column.x, ti.f32, None)
        isl1 = copy_to_taichi(self.ne_fdb[SideTopoInfo]['isl1'].column.info, ti.i32, [self.e_num, 10])
        
        self.init_gpu(ex, ey, isl1, sx)

    @ti.kernel
    @no_type_check
    def init_gpu(self, ex_t: ti.template(), ey_t: ti.template(), isl1: ti.template(), sx_t: ti.template()):
        self.ndt_t[None] = 0.1
        self.cumulative_rain_time_t[None] = 0.0
        
        for ei in range(1, self.e_num):
            self.u_t[ei] = 0.0
            self.v_t[ei] = 0.0
            self.ssq_t[ei] = 0.0
            self.h_t[ei] = self.ez_t[ei] if self.ez_t[ei] > 0.0 else 0.0
            self.eq_t[ei, 0] = self.eq_t[ei, 1] = self.eq_t[ei, 2] = self.eq_t[ei, 3] = 0.0
            self.enq_t[ei, 0] = self.enq_t[ei, 1] = self.enq_t[ei, 2] = self.enq_t[ei, 3] = 0.0
            
            # Get side length (all four sides are the same for square elements)
            lsi0 = isl1[ei, 0]
            self.esl_t[ei] = ti.max((ex_t[ei] - sx_t[lsi0]) * 2.0, 0.0001)
            self.eu_t[ei] = 0b1 << (self.eu_t[ei] - 1)    # set type flag bit
        
        for si in range(1, self.s_num):
            self.sq_t[si] = 0.0
            self.dh_t[si] = -999.0
            so, el, eh = self.sts_t[si, 0], self.sts_t[si, 1], self.sts_t[si, 2]   # orient, lower element, higher element
            if so == 2:
                eil, eir = el, eh
                self.sdc_t[si] = ti.max(ti.abs(ex_t[eir] - ex_t[eil]), 0.01)
            else:
                eib, eit = el, eh
                self.sdc_t[si] = ti.max(ti.abs(ey_t[eit] - ey_t[eib]), 0.01)
        
        self.fr1[None] = self.fr2[None] = self.fr3[None] = self.fr4[None] = self.fr5[None] = self.fr6[None] = self.fr7[None] = 0.0

    @ti.kernel
    @no_type_check
    def tick(self, tide: float, rainq: float) -> ti.f32:
        g = 9.81
        # Tick dt
        dt = ti.max(0.0001, ti.min(self.ndt_t[None], 1.0))    # clamp dt to avoid instability
        self.ndt_t[None] = 1000.0
        
        # Tick gates
        for gi in range(self.g_num):
            up, down, level = self.gate_t[gi, 0], self.gate_t[gi, 1], float(self.gate_t[gi, 2])
            should_open = self.h_t[up] + 0.1 > self.h_t[down]
            new_z = 0.0 if should_open else level
            for ei_count in range(3, 100):
                ei = self.gate_t[gi, ei_count]
                if ei == 0:
                    break
                self.ez_t[ei] = new_z
        
        # Tick sides
        for si in range(1, self.s_num):
            bf = ti.select(self.sbf_t[si] == 1, 0.0, 1.0)  # boundary factor
            so, el, eh = self.sts_t[si, 0], self.sts_t[si, 1], self.sts_t[si, 2]   # orient, lower element, higher element
            # Handle flow in direction x (Type 2: Vertical side, connects Left/Right)
            if so == 2:
                eil, eir = el, eh
                hl, hr = self.h_t[eil], self.h_t[eir]
                dx = self.sdc_t[si]
                xq = self.sq_t[si]
                xwh = ti.max(hl, hr) - ti.max(self.ez_t[eil], self.ez_t[eir], self.dh_t[si]) # water height in direction x
                # Calculate the unit discharge quantity caused by pressure gradient in direction x
                xdq = (-g * (hr - hl) / dx) * ti.max(xwh, 0.0) * dt
                # Calculate the friction term in direction x (Manning's formula)
                xf = 1.0 + g * dt * (self.n ** 2) * ti.abs(xq / (ti.max(xwh, 0.00001) ** (7.0 / 3.0)))
                new_xq = (self.sita * xq + (1.0 - self.sita) * (self.eq_t[eil, 0] / self.esl_t[eil] + self.eq_t[eir, 1] / self.esl_t[eir]) + xdq) / xf
                new_xq = ti.select(xwh < self.min_h, 0.0, new_xq)    # cutoff small flow
                new_xq *= bf                                    # zero flow for boundary sides
                self.sq_t[si] = new_xq                               # update current flow quantity in direction x
                xwh = ti.max(xwh, 0.01)
                sdt = bf * self.afa * self.sl_t[si] / (ti.sqrt(g * xwh) + ti.abs(new_xq) / xwh) # CFL Condition: Ignore boundary sides (connected to element 0)
                sdt = ti.max((0.001 - sdt) * 100000.0, sdt)  # avoid too small time step
                ti.atomic_min(self.ndt_t[None], sdt)  # update global next time step
                flux = new_xq * self.sl_t[si]
                ti.atomic_add(self.enq_t[eil, 1], flux)
                ti.atomic_add(self.enq_t[eir, 0], flux)
            # Handle flow in direction y (Type 1: Horizontal side, connects Bottom/Top)
            else:
                eib, eit = el, eh
                hb, ht = self.h_t[eib], self.h_t[eit]
                dy = self.sdc_t[si]
                yq = self.sq_t[si]
                ywh = ti.max(hb, ht) - ti.max(self.ez_t[eib], self.ez_t[eit], self.dh_t[si]) # water height in direction y
                # Calculate the unit discharge quantity caused by pressure gradient in direction y
                ydq = (-g * (ht - hb) / dy) * ti.max(ywh, 0.0) * dt
                # Calculate the friction term in direction y (Manning's formula)
                yf = 1.0 + g * dt * (self.n ** 2) * ti.abs(yq / (ti.max(ywh, 0.00001) ** (7.0 / 3.0)))
                new_yq = (self.sita * yq + (1.0 - self.sita) * (self.eq_t[eib, 2] / self.esl_t[eib] + self.eq_t[eit, 3] / self.esl_t[eit]) + ydq) / yf
                new_yq = ti.select(ywh < self.min_h, 0.0, new_yq)    # cutoff small flow
                new_yq *= bf                                    # zero flow for boundary sides
                self.sq_t[si] = new_yq                               # update current flow quantity in direction y
                ywh = ti.max(ywh, 0.01)
                sdt = bf * self.afa * self.sl_t[si] / (ti.sqrt(g * ywh) + ti.abs(new_yq) / ywh) # CFL Condition: Ignore boundary sides (connected to element 0)
                sdt = ti.max((0.001 - sdt) * 100000.0, sdt)  # avoid too small time step
                ti.atomic_min(self.ndt_t[None], sdt)  # update global next time step
                flux = new_yq * self.sl_t[si]
                ti.atomic_add(self.enq_t[eib, 3], flux)
                ti.atomic_add(self.enq_t[eit, 2], flux)
        
        # Tick infiltration rate during rainfall (based on Horton model)
        # Green space / grassland infiltration rate (Types 3, 5, 7), initial 3 inches/hr, final 0.1 inches/hr
        # Infiltration rate of impermeable pavements (Types 1, 2), initial 0.8 inches/hr, final 0.02 inches/hr
        # Infiltration rate of impermeable water bodies (Types 4, 6), always 0.0
        # fr1 = fr2 = fr3 = fr4 = fr5 = fr6 = fr7 = 0.0
        if rainq > 0.0:
            self.cumulative_rain_time_t[None] += dt
            self.fr3[None] = self.fr5[None] = self.fr7[None] = horton_decay(3.0, 0.1, 2.0, self.cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
            self.fr1[None] = self.fr2[None] = horton_decay(0.8, 0.02, 10.0, self.cumulative_rain_time_t[None] / 3600.0) * 0.0254 / 3600.0
        
        # Tick elements
        for ei in range(1, self.e_num):
            # Calculate flow quantities
            ql = self.enq_t[ei, 0]
            qr = self.enq_t[ei, 1]
            qb = self.enq_t[ei, 2]
            qt = self.enq_t[ei, 3]
            tq = ql - qr + qb - qt + self.ssq_t[ei]                                  # total inflow quantity
            self.eq_t[ei, 0], self.eq_t[ei, 1], self.eq_t[ei, 2], self.eq_t[ei, 3] = ql, qr, qb, qt # update current side flow quantities
            self.enq_t[ei, 0] = self.enq_t[ei, 1] = self.enq_t[ei, 2] = self.enq_t[ei, 3] = 0.0     # reset next time step side flow quantities
        
            # Calculate infiltration masks for all underlay types
            eu = self.eu_t[ei]       # underlay type flag
            f1 = (eu >> 0) & 1  # building
            f2 = (eu >> 1) & 1  # road
            f3 = (eu >> 2) & 1  # agricultural land
            f4 = (eu >> 3) & 1  # fish pond
            f5 = (eu >> 4) & 1  # mountainous land
            f6 = (eu >> 5) & 1  # water body
            f7 = (eu >> 6) & 1  # catch basin
            
            ea = self.esl_t[ei] ** 2                 # area of element
            next_h = self.h_t[ei] + (tq * dt) / ea   # update next water depth
            next_h += rainq * dt                # add rainfall effect
            next_h -= (self.fr1[None] * f1 + self.fr2[None] * f2 + self.fr3[None] * f3 +
                       self.fr4[None] * f4 + self.fr5[None] * f5 + self.fr6[None] * f6 +
                       self.fr7[None] * f7) * dt     # subtract infiltration effect
            next_h = ti.max(next_h, self.ez_t[ei])   # water depth cannot be lower than ground elevation
            self.h_t[ei] = next_h                    # update current water depth
            self.depth_t[ei] = ti.max(next_h - self.ez_t[ei], 0.00)     # update current water depth (h - z)
            
        # Tick boundaries
        for count in range(self.b_num):
            bdei = self.bdei_t[count]
            self.h_t[bdei] = tide
        
        # Output dt
        return self.ndt_t[None]

    @ti.kernel
    @no_type_check
    def update_velocities(self):
        for ei in range(1, self.e_num):
            esl = self.esl_t[ei]
            depth = ti.max(self.h_t[ei] - self.ez_t[ei], 0.01)
            self.u_t[ei] = ti.select(
                depth < self.min_h, 0.0,
                (self.eq_t[ei, 0] + self.eq_t[ei, 1]) / esl / depth / 2.0
            )
            self.v_t[ei] = ti.select(
                depth < self.min_h, 0.0,
                (self.eq_t[ei, 2] + self.eq_t[ei, 3]) / esl / depth / 2.0
            )

    def evolve(self) -> Iterator[float]:
        output_uvh_fn = Path(self.cfg.uvh_dir)
        if output_uvh_fn.exists():
            shutil.rmtree(output_uvh_fn)
        output_uvh_fn.mkdir(parents=True, exist_ok=True)
        
        last_output_count = 0
        last_output_time = self.current_time

        while self.cfg.duration == -1 or self.current_time - self.evolve_start_time < self.cfg.duration:
            # Update tide by linear interpolation
            if self.current_time >= self.tts[self.current_tide_idx + 1]:
                if self.current_tide_idx + 2 >= self.t_num:
                    break                   # end of simulation
                else:
                    self.current_tide_idx += 1   # tick tide index
            
            tide_val = lerp(
                self.tls[self.current_tide_idx],
                self.tls[self.current_tide_idx + 1],
                (self.current_time - self.tts[self.current_tide_idx]) / (self.tts[self.current_tide_idx + 1] - self.tts[self.current_tide_idx])
            )
            
            # Update average rainfall quantity (m/s)
            rainq = 0.0
            if self.current_time <= self.rts[self.r_num - 1]: # still raining
                if self.current_time > self.rts[self.current_rain_idx + 1]:
                    self.current_rain_idx += 1  # tick rainfall index
                rainq = self.rqs[self.current_rain_idx + 1] / (self.rts[self.current_rain_idx + 1] - self.rts[self.current_rain_idx]) * 0.001
            
            # Core solver
            dt = self.tick(tide_val, rainq)
            self.current_time += dt
            
            # Output cumulative time every 5 minutes (or as configured)
            if self.current_time - last_output_time >= self.cfg.yield_step:
                last_output_time += self.cfg.yield_step
                
                # Update water velocities before output (kept from original logic)
                self.update_velocities()
                
                # Output uvh data
                uvh_db = fdb.ORM.truncate([
                    fdb.TableDefn(UVH, self.e_num)
                ])
                us = uvh_db[UVH][UVH].column.u
                vs = uvh_db[UVH][UVH].column.v
                hs = uvh_db[UVH][UVH].column.h
                us[:] = self.u_t.to_numpy()
                vs[:] = self.v_t.to_numpy()
                hs[:] = self.h_t.to_numpy() # Note: h_t is total height (water elevation), not depth.
                uvh_fn = output_uvh_fn / f'uvh_{last_output_count}.fdb'
                uvh_db.save(str(uvh_fn))
                uvh_db.unlink()
                last_output_count += 1
                
                yield self.current_time - self.evolve_start_time