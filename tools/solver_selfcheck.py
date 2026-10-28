#!/usr/bin/env python3
"""Run portable high-bed rainfall and volume-preserving gate checks.

Example: python tools/solver_selfcheck.py --arch cuda --output PATH_TO_NEW_RUN
An unavailable requested backend fails; the tool does not fall back to CPU.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nh_flood_2d.core.solver_compact import run_simulation
from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint
from src.nh_flood_2d.core.run_record import sha256
from src.nh_flood_2d.core.device_budget import DeviceBudget
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.util.ti import init_taichi

START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


def write_seed(path, dc, fc, depth, bed, gate_open=False):
    depth = np.asarray(depth, dtype='float32')
    bed = np.asarray(bed, dtype='float32')
    zeros = np.zeros_like(depth)
    side_count = len(Path(dc.ns).read_text().splitlines()) + 1
    write_checkpoint(path, depth=depth, bed=bed, q=np.zeros(side_count,dtype='float32'),
        rain_time=zeros, infiltration=zeros, max_depth=depth,
        first_time_hi=np.zeros(len(depth),dtype='uint32'),
        first_time_lo=np.zeros(len(depth),dtype='uint32'), max_speed=zeros,
        clock_hi=0, clock_lo=0, gate_open=gate_open,
        metadata_json=json.dumps({'run_state': False,
            'ne_sha256': sha256(dc.ne), 'ns_sha256': sha256(dc.ns),
            'gate_config': fc.gate.model_dump_json() if fc.gate else 'null'}))


def case_inputs(path, bed, rain_mm, gate=False, restart_depth=None):
    path.mkdir(parents=True)
    # Isolated square cells: level/gate signals cannot move water between them.
    (path/'ne.txt').write_text(
        f'1,1,1,1,1,1,2,3,4,5,5,{bed},11\n'
        f'2,1,1,1,1,5,6,7,8,15,5,{bed},10\n')
    (path/'ns.txt').write_text(''.join([
        f'1,2,0,1,0,0,10,0,5,{bed},0\n', f'2,2,1,0,0,0,10,10,5,{bed},0\n',
        f'3,1,0,0,0,1,10,5,0,{bed},0\n', f'4,1,0,0,1,0,10,5,10,{bed},0\n',
        f'5,2,0,2,0,0,10,10,5,{bed},0\n', f'6,2,2,0,0,0,10,20,5,{bed},0\n',
        f'7,1,0,0,0,2,10,15,0,{bed},0\n', f'8,1,0,0,2,0,10,15,10,{bed},0\n']))
    duration = 60 if not gate else 2
    with rasterio.open(path/'rain.tif','w',driver='GTiff',width=1,height=1,count=1,
                       dtype='float32',crs='EPSG:2326',transform=from_origin(0,10,20,10)) as dst:
        dst.write(np.array([[rain_mm]],dtype=np.float32),1)
    (path/'intervals.csv').write_text('interval_start,interval_end\n'+
        f'{START.isoformat()},{(START+timedelta(seconds=duration)).isoformat()}\n')
    boundaries=[]
    if gate:
        (path/'level.geojson').write_text(json.dumps({'type':'LineString','coordinates':[[20,1],[20,9]],
            'crs':{'type':'name','properties':{'name':'EPSG:2326'}}}))
        (path/'level.csv').write_text('time,value\n'+
            '\n'.join(f'{(START+timedelta(seconds=t)).isoformat()},{eta}'
                      for t,eta in [(0,bed),(1,bed+2),(2,bed+2)])+'\n')
        boundaries=[dict(name='sea',kind='level',line=str(path/'level.geojson'),series=str(path/'level.csv'))]
    restart=str(path/'restart.fdb') if restart_depth is not None else ''
    dc=DomainConfig(ne=str(path/'ne.txt'),ns=str(path/'ns.txt'),epsg_code=2326,
        domain_dir=str(path/'out'),start=START,duration=duration,warmup_duration=0,
        land_parameters={key:dict(manning_n=.03,infiltration='none') for key in (10,11)},
        yield_step=duration,restart_state=restart)
    fc=ForceConfig(boundaries=boundaries,rainfall=dict(raster=str(path/'rain.tif'),intervals=str(path/'intervals.csv')),
        gate=dict(threshold=bed+1,land_type=11,level_boundary='sea') if gate else None)
    if restart_depth is not None:
        write_seed(restart, dc, fc, [0.,restart_depth,0.], [0.,bed,bed])
    (path/'domain.json').write_text(dc.model_dump_json(indent=2))
    (path/'force.json').write_text(fc.model_dump_json(indent=2))
    return dc,fc


def inventory_rounding_check(path):
    dc, fc = case_inputs(path, bed=0., rain_mm=0.)
    # Convert the two isolated cells to a connected pair.
    ne = (path/'ne.txt').read_text().replace('2,1,1,1,1,5,6,7,8,', '2,1,1,1,1,2,6,7,8,')
    ns = [line for line in (path/'ns.txt').read_text().splitlines() if not line.startswith('5,')]
    ns[1] = '2,2,1,2,0,0,10,10,5,0,0'
    (path/'ne.txt').write_text(ne)
    (path/'ns.txt').write_text('\n'.join(ns)+'\n')
    mm = np.float32(1.4911491234670393e-5)
    with rasterio.open(path/'rain.tif','r+') as dst:
        dst.write(np.array([[mm]],dtype='float32'),1)
    (path/'intervals.csv').write_text('interval_start,interval_end\n'+
        f'{START.isoformat()},{(START+timedelta(seconds=1)).isoformat()}\n')
    seed=path/'seed.fdb'
    write_seed(seed,dc,fc,[0.,.5,.4],[0.,0.,0.])
    dc=dc.model_copy(update={'duration':1,'yield_step':1,'restart_state':str(seed)})
    (path/'domain.json').write_text(dc.model_dump_json(indent=2))
    budget=run_simulation(dc,fc)
    state=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
    ideal=.5-float(state.q[2])*.1+float(np.float32(float(mm)*.001))
    ulp=float(np.spacing(np.float32(ideal)))
    error=abs(float(state.depth[1])-ideal)/ulp
    if error >= .55:
        raise AssertionError(f'Inventory/rain addition order lost precision: {error} ULP')
    return dict(error_ulp=error,ideal_depth_m=ideal,depth_m=float(state.depth[1]),budget=budget)


def run_checks(output, arch_name):
    arch=getattr(ti,arch_name)
    init_taichi(arch=arch)
    if ti.lang.impl.current_cfg().arch != arch:
        raise RuntimeError('The requested Taichi backend was not selected')
    # Cross-block and cross-output cancellation must retain the small terms.
    accumulator=DeviceBudget(2,3,2)
    terms=np.array([[2.**100,2.**25,0.],[1.,1.,0.]],dtype='float32')
    terms.view('uint32')[0,2]=1
    accumulator.step.from_numpy(terms);accumulator.collect();accumulator.step_finished()
    accumulator.flush()
    accumulator.step.from_numpy(np.array([[-2.**100,-2.**25,0.],[0.,0.,0.]],dtype='float32'))
    accumulator.collect();accumulator.step_finished()
    exact_budget=accumulator.flush()
    expected_budget=np.array([1.,1.,math.ldexp(1.,-149)],dtype='float64')
    if not np.array_equal(exact_budget,expected_budget):
        raise AssertionError(f'Device diagnostic accumulation lost small terms: {exact_budget}')
    dc,fc=case_inputs(output/'high_bed_rain',bed=10000.,rain_mm=.01)
    rain=run_simulation(dc,fc)
    expected=.01*.001*200
    if abs(rain['storage_m3']-expected) > expected*1e-5:
        raise AssertionError(f'High-bed rain was lost: {rain["storage_m3"]} vs {expected}')
    # A water-depth state must survive its checkpoint even when WSE rounds to bed.
    rain_state=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
    resumed, resumed_force=case_inputs(output/'high_bed_resume',bed=10000.,rain_mm=0.)
    seed=output/'high_bed_resume/seed.fdb'
    write_seed(seed, resumed, resumed_force, rain_state.depth, rain_state.bed)
    resumed=resumed.model_copy(update={'restart_state':str(seed)})
    resume_budget=run_simulation(resumed,resumed_force)
    if abs(resume_budget['storage_m3']-rain['storage_m3'])>expected*1e-6:
        raise AssertionError('High-bed checkpoint lost water on restart')
    gate_results={}
    for name,initial in [('gate_dry',None),('gate_wet',.25)]:
        dc,fc=case_inputs(output/name,bed=10.,rain_mm=0.,gate=True,restart_depth=initial)
        budget=run_simulation(dc,fc)
        final=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
        gate_surface=float(final.bed[1]+final.depth[1])
        manifest=json.loads((Path(dc.domain_dir)/'manifest.json').read_text())
        target=0. if initial is None else initial
        if not manifest['gate_open'] or abs(gate_surface-target)>1e-6 or budget['gate_bed_m3']!=0.:
            raise AssertionError(f'Gate changed its water volume: {gate_surface}, expected {target}')
        gate_results[name]={'final_gate_depth_m':gate_surface,'expected_depth_m':target,'budget':budget}
        if name=='gate_wet':
            resumed,resumed_force=case_inputs(output/'gate_wet_resume',bed=10.,rain_mm=0.,gate=True)
            seed=output/'gate_wet_resume/seed.fdb'
            write_seed(seed, resumed, resumed_force, final.depth, final.bed, gate_open=True)
            resumed=resumed.model_copy(update={'restart_state':str(seed)})
            gate_resume=run_simulation(resumed,resumed_force)
            state=read_checkpoint(Path(resumed.output_dir)/'state_final.fdb')
            if abs(float(state.depth[1])-.25)>1e-6 or state.bed[1]!=0 or not state.gate_open:
                raise AssertionError('Gate state failed to preserve working bed/depth/status')
            gate_results['gate_wet_resume']={'depth_m':float(state.depth[1]),'bed_m':float(state.bed[1]),'budget':gate_resume}
    # Independent groups avoid exhausting Taichi's process-wide field limit.
    ti.reset()
    import src.nh_flood_2d.util.ti as ti_helpers
    ti_helpers._TI_INITIALIZED = False
    init_taichi(arch=arch)
    rounding = inventory_rounding_check(output/'inventory_rounding')
    return {'status':'passed','requested_arch':arch_name,'actual_arch':ti.lang.impl.current_cfg().arch.name,
            'device_budget_cancellation':{'actual':exact_budget.tolist(),'expected':expected_budget.tolist(),
                'word_count':accumulator.word_count,'transfers':accumulator.transfer_count},
            'inventory_rounding':rounding,
            'high_bed_rain':{'bed_m':10000.,'expected_storage_m3':expected,'budget':rain,'resumed_budget':resume_budget},**gate_results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch',choices=['cpu','cuda','metal','vulkan','opengl','dx11','amdgpu'],required=True)
    parser.add_argument('--output',type=Path,required=True,help='New directory for input and output evidence')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result=run_checks(args.output,args.arch)
    (args.output/'selfcheck.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
