"""Opt-in production-path gate smoke, replay, continuation and timing evidence.

All device work is isolated in child processes. Metal is process/time gated by
gate_gpu_tools; --backend cpu can calibrate the harness without touching a GPU.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_case(path, case):
    from test_gate_solver import setup, level_boundary, START
    if case == 'small':
        boundaries = level_boundary(path, [(0,.5),(1,.5),(2,2),(3600,2)])
        domain, force, *_ = setup(path, boundaries=boundaries, bed=[0,3.125,0],
            depth=[.5,0,.3], gate=dict(trigger_level=1.5, opening_seconds=2.875,
                                     open_elevation=.13))
        return domain, force, 7., (1.,3.,6.)
    path.mkdir(parents=True, exist_ok=True)
    cells, types, bed, depth, gates, segments = [], [], [], [], [], []
    for group in range(7):
        y = 100*group
        for row in range(22):
            for col in range(25):
                x, bottom = 4*col, y+4*row
                cells.append((x,x+4,bottom,bottom+4))
                kind = 1 if row == 0 else 10 if row == 21 else 11
                types.append(kind)
                bed.append(3.125 if kind == 11 else 0.)
                depth.append(.5 if kind == 1 else .3 if kind == 10 else 0.)
        gates.append(dict(name=f'G{group}',inside_point=(2,y+6),control_point=(2,y+2)))
        segments.append([[0,y],[100,y]])
    line, series = path/'rivers.geojson', path/'rivers.csv'
    line.write_text(json.dumps(dict(type='MultiLineString',coordinates=segments,
        crs=dict(type='name',properties=dict(name='EPSG:2326')))))
    series.write_text('time,value\n'+''.join(
        f'{(START+timedelta(seconds=t)).isoformat()},{v}\n'
        for t,v in [(0,.5),(1,.5),(2,2),(3600,2)]))
    domain, force, *_ = setup(path, cells=cells, types=types, bed=bed, depth=depth,
        boundaries=[dict(name='rivers',kind='level',line=str(line),series=str(series))],
        gate=dict(gates=gates,trigger_level=1.5,opening_seconds=2.875,open_elevation=.13))
    return domain, force, 40., (1.,3.,6.)


class CpuMemory:
    def sample(self):
        return 0
    def report(self):
        return dict(process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def initialize(args):
    import taichi as ti
    from src.nh_flood_2d.util import ti as runtime
    if args.backend == 'metal':
        from gate_gpu_tools import require_gpu_window, MetalMemory
        window = require_gpu_window()
        ti.init(arch=ti.metal, offline_cache=False,log_level=ti.ERROR)
        memory = MetalMemory()
    else:
        window = None
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False,log_level=ti.ERROR)
        memory = CpuMemory()
    runtime._TI_INITIALIZED = True
    return ti,memory,window


def worker(args):
    import numpy as np
    from test_gate_solver import run
    from src.nh_flood_2d.core.output import read_bed_at, read_checkpoint, write_checkpoint
    from src.nh_flood_2d.core.gate import GateController
    from gate_gpu_tools import GraphProfile
    ti,memory,window = initialize(args)
    path = Path(args.output).resolve()
    domain,force,end,splits = build_case(path,args.case)
    if args.disabled:
        force.gate.enabled = False
        seed = read_checkpoint(domain.restart_state)
        metadata = json.loads(seed.metadata_json)
        metadata['gate_config'] = force.gate.model_dump_json()
        seed.metadata_json = json.dumps(metadata)
        write_checkpoint(domain.restart_state, **vars(seed))
    if args.performance:
        domain = domain.model_copy(update={'yield_step':300})
        end = 120.
    profiler = GraphProfile(ti,memory)
    profiler.install()
    original_init, controller_sizes = GateController.__init__, []
    def measure_controller_fields(controller,*values,**keywords):
        original_init(controller,*values,**keywords)
        names=('cell_gate','targets','control','closed','pond','commanded','command',
               'flow_seen','first_flow','opened','opened_at','previous','inventory_delta','sources')
        # These owned fields are F32/I32/U32. Shared physical bed/depth/area and
        # the clock are excluded; Metal's resource pool can hide small deltas.
        size=sum(int(np.prod(getattr(controller,name).shape))*4 for name in names)
        diagnostic=sum(int(np.prod(getattr(controller.budget,name).shape))*4
                       for name in ('step','digits','invalid'))
        controller_sizes.append(dict(owned_logical_field_bytes=size+diagnostic,
            diagnostic_accumulator_bytes=diagnostic,
            diagnostic_terms=controller.budget.step.shape[1],
            accumulator_word_count=controller.budget.word_count))
    GateController.__init__ = measure_controller_fields
    full,manifest = run(domain,force,end)
    GateController.__init__ = original_init
    full_profile = profiler.report()
    profiler.uninstall()
    full_memory = memory.report()
    full_memory['gate_controller_logical_fields']=(controller_sizes[0] if controller_sizes
        else dict(owned_logical_field_bytes=0,diagnostic_accumulator_bytes=0))
    result = dict(case=args.case,backend=manifest['backend'],device=manifest['device_name'],
        simulated_end_seconds=end,configuration_duration_seconds=domain.duration,
        enabled=not args.disabled, source_sha256=manifest['source_sha256'],
        input_sha256=manifest['input_sha256'],platform=platform.platform(),
        harness_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('gate_backend_validation.py','gate_gpu_tools.py','test_gate_solver.py')},
        dependencies={package:importlib.metadata.version(package)
            for package in ('taichi','fastdb4py','numpy','pydantic','rasterio')},
        python=sys.version.split()[0], gpu_window=window,
        steps=manifest['steps'], profile=full_profile, memory=full_memory,
        hashes={name:hashlib.sha256(getattr(full,name).tobytes()).hexdigest()
                for name in ('depth','q','bed')},
        global_budget=manifest['mass_balance_final'],
        gate_control=manifest.get('gate_control'), restart=[])
    if not args.disabled and not args.performance and not args.no_restarts:
        for split,phase in zip(splits,('before','during','after')):
            part,_ = run(domain,force,split,output=path/f'part-{phase}')
            saved_units = json.loads(part.metadata_json)['gate_controller']['budget_units']
            original_report, first_restored_units = GateController.report, []
            def capture_first_report(controller, depth, area, clock_seconds):
                report = original_report(controller,depth,area,clock_seconds)
                if not first_restored_units:
                    first_restored_units.append(list(controller.budget.total_units))
                return report
            GateController.report = capture_first_report
            actual,resumed_manifest = run(domain,force,end,output=path/f'resumed-{phase}',
                restart=path/f'part-{phase}/output/state_final.fdb')
            GateController.report = original_report
            assert first_restored_units[0] == saved_units
            equal = {name:getattr(full,name).tobytes()==getattr(actual,name).tobytes()
                for name in ('depth','q','bed','rain_time','infiltration',
                             'max_depth','max_speed','first_time_hi','first_time_lo')}
            replay = read_bed_at(path/'out/output',split)
            error = np.abs(replay.astype(float)-part.bed.astype(float))
            bound = float(4*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(part.bed)))))
            control = json.loads(part.metadata_json)['gate_controller']
            if phase == 'before':
                assert not any(control['commanded'])
            elif phase == 'during':
                assert all(control['commanded']) and not any(control['opened'])
            else:
                assert all(control['opened'])
            budget_a = json.loads(actual.metadata_json)['gate_controller']
            budget_f = json.loads(full.metadata_json)['gate_controller']
            units_a,units_f = budget_a.pop('budget_units'),budget_f.pop('budget_units')
            units_differences = [float(a-b)*2.**-149 for a,b in zip(units_a,units_f)]
            result['restart'].append(dict(phase=phase,split_seconds=split,
                physical_state_bitwise_equal=equal,
                control_state_equal_without_budget=(budget_a==budget_f),
                initial_restored_budget_units_exact=True,
                cumulative_budget_units_equal=(units_a==units_f),
                max_budget_term_difference_m3=max(map(abs,units_differences)),
                budget_term_differences_m3=units_differences,
                resumed_pond_residual_m3=resumed_manifest['gate_control']['pond_budget']['residual_m3'],
                resumed_channel_residuals_m3={gate['name']:gate['channel_budget']['residual_m3']
                    for gate in resumed_manifest['gate_control']['gates']},
                replay_bitwise_equal=(replay.tobytes()==part.bed.tobytes()),
                replay_max_abs_error_m=float(error.max()),replay_absolute_bound_m=bound))
            assert all(equal.values()), result['restart'][-1]
            assert budget_a == budget_f
            if args.backend == 'cpu':
                assert units_a == units_f
            assert float(error.max()) <= bound
            for gate in resumed_manifest['gate_control']['gates']:
                channel = gate['channel_budget']
                scale = max(1.,channel['initial_storage_m3'],channel['storage_m3'],
                    abs(channel['river_net_m3'])+abs(channel['pond_net_m3']))
                assert abs(channel['residual_m3']) <= max(2e-5,2e-6*scale)
        if args.backend == 'metal' and any(not row['cumulative_budget_units_equal'] for row in result['restart']):
            repeat,_ = run(domain,force,end,output=path/'full-repeat')
            assert all(getattr(repeat,name).tobytes()==getattr(full,name).tobytes()
                       for name in ('depth','q','bed'))
            repeat_units=json.loads(repeat.metadata_json)['gate_controller']['budget_units']
            full_units=json.loads(full.metadata_json)['gate_controller']['budget_units']
            spread=[abs(float(a-b)*2.**-149) for a,b in zip(repeat_units,full_units)]
            result['enabled_full_repeat_budget_spread_m3']=spread
            for row in result['restart']:
                row['budget_terms_outside_measured_repeat_spread']=[index for index,difference
                    in enumerate(row['budget_term_differences_m3']) if abs(difference)>spread[index]]
        assert np.all(full.depth >= 0)
        scale=max(1.,manifest['mass_balance_final']['storage_m3'])
        assert abs(manifest['mass_balance_final']['residual_m3']) <= 2e-6*scale
        pond = manifest['gate_control']['pond_budget']
        pond_scale=max(1.,pond['initial_storage_m3'],pond['storage_m3'],
            sum(abs(gate['pond_net_m3']) for gate in manifest['gate_control']['gates']))
        assert abs(pond['residual_m3']) <= 2e-6*pond_scale
        result['closure_tolerances_m3'] = dict(global_volume=2e-6*scale,
                                             pond_volume=2e-6*pond_scale,channels={})
        for gate in manifest['gate_control']['gates']:
            channel = gate['channel_budget']
            channel_scale=max(1.,channel['initial_storage_m3'],channel['storage_m3'],
                abs(channel['river_net_m3'])+abs(channel['pond_net_m3']),
                sum(abs(channel[key]) for key in ('rain_m3','infiltration_m3',
                    'positivity_m3','pipe_m3','boundary_m3','rounding_m3')))
            tolerance = max(2e-5,2e-6*channel_scale)
            result['closure_tolerances_m3']['channels'][gate['name']] = tolerance
            assert abs(channel['residual_m3']) <= tolerance
    (path/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend',choices=['cpu','metal'],default='cpu')
    parser.add_argument('--case',choices=['small','medium'],default='small')
    parser.add_argument('--output',required=True)
    parser.add_argument('--disabled',action='store_true')
    parser.add_argument('--performance',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--no-restarts',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.backend=='metal':
            from gate_gpu_tools import require_gpu_window
            require_gpu_window()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',
            '--backend',args.backend,'--case',args.case,'--output',args.output]
        if args.disabled: command.append('--disabled')
        if args.performance: command.append('--performance')
        if args.no_restarts: command.append('--no-restarts')
        path=Path(args.output)
        path.mkdir(parents=True,exist_ok=True)
        with (path/'run.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(path/'result.json',flush=True)


if __name__=='__main__':
    main()
