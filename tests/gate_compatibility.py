"""Backend-local fingerprints against 7648185 (no branch changes).

Run with --case small, or --case hk --domain FILE --force FILE. All generated
files, including the exported baseline source, stay under --output.
Metal requires the authorized process/time window; use --reference-repeats 2
or more to measure diagnostic-reduction variability before comparing candidates.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import importlib.metadata
import platform


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    import numpy as np
    import taichi as ti
    import fastdb4py as fdb
    from contextlib import closing
    from src.nh_flood_2d.util import ti as runtime
    from src.nh_flood_2d.input import DomainConfig, ForceConfig
    from src.nh_flood_2d.preprocess import preprocess
    from src.nh_flood_2d.preprocess.force import MeshGeometry, Forcing
    from src.nh_flood_2d.schema.feature import Ne, SideTopoInfo
    from src.nh_flood_2d.core.solver_compact import solver
    from src.nh_flood_2d.core.output import read_checkpoint
    from src.nh_flood_2d.core.clock import StepClock

    output = Path(args.output).resolve()
    domain_data = json.loads(Path(args.domain).read_text())
    force_data = json.loads(Path(args.force).read_text())
    domain_data['domain_dir'] = str(output/'run')
    domain = DomainConfig.model_validate_json(json.dumps(domain_data))
    memory = profile = None
    if args.backend == 'metal':
        from gate_gpu_tools import require_gpu_window
        require_gpu_window()
    ti.init(arch=ti.metal if args.backend == 'metal' else ti.cpu,
            cpu_max_num_threads=1, offline_cache=False, log_level=ti.ERROR)
    runtime._TI_INITIALIZED = True
    preprocess(domain, ForceConfig.model_validate(force_data))
    if args.disabled:
        from networkx.utils import UnionFind
        with closing(fdb.RecordEngine.load(domain.ne_fdb, from_file=True)) as ne, \
                closing(fdb.RecordEngine.load(domain.ns_fdb, from_file=True)) as ns:
            mesh = MeshGeometry.from_fdb(ne, ns)
            types = ne.table(Ne).column.type.copy()
            bed = ne.table(Ne).column.z.copy()
            topo = ns.table(SideTopoInfo).column.info.copy().reshape(-1,3)
        selected = types == 11
        selected[0] = False
        both = selected[topo[:,1]] & selected[topo[:,2]]
        components = UnionFind(np.flatnonzero(selected))
        for _,left,right in topo[both]:
            components.union(int(left),int(right))
        seeds = sorted(min(group) for group in components.to_sets())
        loaded = Forcing.load(domain, ForceConfig.model_validate(force_data), mesh)
        excluded = selected.copy()
        for boundary in loaded.boundaries:
            if boundary.kind == 'level':
                excluded[boundary.cells] = True
        control = np.flatnonzero(~excluded & (np.arange(len(types)) > 0))[0]
        force_data['gate'] = dict(land_type=11, pond_land_type=10, enabled=False, trigger_level=3.07,
            opening_seconds=1800, open_elevation=min(1.3, float(np.min(bed[selected]))),
            gates=[dict(name=f'G{i}', inside_point=[mesh.x[c],mesh.y[c]],
                        control_point=[mesh.x[control],mesh.y[control]]) for i,c in enumerate(seeds)])
    force = ForceConfig.model_validate(force_data)
    if args.backend == 'metal':
        from gate_gpu_tools import MetalMemory, GraphProfile
        memory = MetalMemory()
        if args.profile:
            profile = GraphProfile(ti, memory)
            profile.install()
    original, trace_hash = StepClock.read_trace, hashlib.sha256()
    def capture(clock):
        rows = original(clock)
        trace_hash.update(rows[rows[:,3] == clock.COMPLETED].astype('<u4').tobytes())
        if memory is not None:
            memory.sample()
        return rows
    StepClock.read_trace = capture
    budget = solver(domain, force)
    state = read_checkpoint(Path(domain.output_dir)/'state_final.fdb')
    manifest = json.loads((Path(domain.domain_dir)/'manifest.json').read_text())
    hashes = {name: hashlib.sha256(getattr(state,name).tobytes()).hexdigest()
              for name in ('depth','q','bed')}
    hashes.update(dt_sequence=manifest['dt_sequence_sha256'], trace=trace_hash.hexdigest(),
        budget=hashlib.sha256(json.dumps(budget,sort_keys=True).encode()).hexdigest(),
        mass_balance_csv=hashlib.sha256((Path(domain.domain_dir)/'mass_balance.csv').read_bytes()).hexdigest(),
        step_statistics_csv=hashlib.sha256((Path(domain.domain_dir)/'step_statistics.csv').read_bytes()).hexdigest())
    result = dict(hashes=hashes, steps=manifest['steps'],
        backend=manifest['backend'], device=manifest['device_name'], duration=domain.duration,
        warmup_duration=domain.warmup_duration, clock=[state.clock_hi,state.clock_lo],
        nonzero_discharge_edges=int(np.count_nonzero(state.q)),
        source_sha256=manifest['source_sha256'], input_sha256=manifest['input_sha256'],
        platform=platform.platform(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version.split()[0], dependencies={package:importlib.metadata.version(package)
            for package in ('taichi','fastdb4py','numpy','pydantic','rasterio')})
    if memory is not None:
        result['memory'] = memory.report()
    if profile is not None:
        result['profile'] = profile.report()
    (output/'fingerprint.json').write_text(json.dumps(result, indent=2)+'\n')


def budget_differences(output, names):
    """Report absolute CSV-field spread against the first original run."""
    import csv
    import numpy as np
    tables = {}
    for name in names:
        with (output/name/'run/mass_balance.csv').open() as stream:
            tables[name] = list(csv.DictReader(stream))
    reference = tables[names[0]]
    differences = {}
    for name in names[1:]:
        rows = tables[name]
        assert len(rows) == len(reference)
        fields = {}
        for key in reference[0]:
            if key in ('time','phase'):
                assert [row[key] for row in rows] == [row[key] for row in reference]
                continue
            actual = [float(row[key]) if row[key] else 0. for row in rows]
            expected = [float(row[key]) if row[key] else 0. for row in reference]
            fields[key] = float(np.max(np.abs(np.array(actual)-expected)))
        differences[name] = fields
    originals = [name for name in names[1:] if name.startswith('7648185')]
    envelope = {key:max((differences[name][key] for name in originals),default=0.)
                for key in next(iter(differences.values()))}
    outside = {name:[key for key,value in fields.items() if value > envelope[key]]
               for name,fields in differences.items() if not name.startswith('7648185')}
    return dict(reference_repeat_max_abs_by_field=envelope,
        max_abs_difference_by_field=differences,
        fields_outside_measured_reference_spread=outside,
        note='Measured finite-sample spread, not a bound on all GPU reduction schedules. Any fields outside the observed spread are listed explicitly.')


def prepare(args, output):
    output.mkdir(parents=True, exist_ok=True)
    if args.case == 'hk':
        domain_path, force_path = Path(args.domain).resolve(), Path(args.force).resolve()
        domain, force = json.loads(domain_path.read_text()), json.loads(force_path.read_text())
        for key in ('ne','ns'):
            domain[key] = str((domain_path.parent/domain[key]).resolve())
        for boundary in force['boundaries']:
            for key in ('line','series'):
                boundary[key] = str((force_path.parent/boundary[key]).resolve())
        for key in ('raster','intervals'):
            force['rainfall'][key] = str((force_path.parent/force['rainfall'][key]).resolve())
        domain.update(duration=3600, warmup_duration=0, restart_state='')
        force['gate'] = None
    else:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        ne, ns = output/'ne.txt', output/'ns.txt'
        ne.write_text('1,1,1,1,1,1,2,3,4,5,5,0,1\n'
            '2,1,1,1,1,2,5,6,7,15,5,0.2,11\n'
            '3,1,1,1,1,5,8,9,10,25,5,0,10\n')
        ns.write_text('1,2,0,1,0,0,10,0,5,0,0\n2,2,1,2,0,0,10,10,5,0,0\n'
            '3,1,0,0,0,1,10,5,0,0,0\n4,1,0,0,1,0,10,5,10,0,0\n'
            '5,2,2,3,0,0,10,20,5,0,0\n6,1,0,0,0,2,10,15,0,0,0\n'
            '7,1,0,0,2,0,10,15,10,0,0\n8,2,3,0,0,0,10,30,5,0,0\n'
            '9,1,0,0,0,3,10,25,0,0,0\n10,1,0,0,3,0,10,25,10,0,0\n')
        raster, intervals = output/'rain.tif', output/'intervals.csv'
        with rasterio.open(raster,'w',driver='GTiff',width=1,height=1,count=1,
            dtype='float32',crs='EPSG:2326',transform=from_origin(0,10,30,10)) as dst:
            dst.write(np.array([[100]],'float32'),1)
        intervals.write_text('interval_start,interval_end\n2023-09-07T08:00:00Z,2023-09-07T09:00:00Z\n')
        line, series = output/'river.geojson', output/'river.csv'
        line.write_text(json.dumps(dict(type='LineString',coordinates=[[0,0],[0,10]],
            crs=dict(type='name',properties=dict(name='EPSG:2326')))))
        series.write_text('time,value\n2023-09-07T08:00:00Z,4\n2023-09-07T09:00:00Z,4\n')
        domain=dict(ne=str(ne),ns=str(ns),epsg_code=2326,start='2023-09-07T08:00:00Z',
            duration=12,warmup_duration=2,yield_step=3,land_parameters={
                t:dict(manning_n=.03,infiltration='none') for t in (1,10,11)})
        force=dict(boundaries=[dict(name='river',kind='discharge',line=str(line),series=str(series))],
            rainfall=dict(raster=str(raster),intervals=str(intervals)),gate=None)
    (output/'domain.json').write_text(json.dumps(domain,indent=2)+'\n')
    (output/'force.json').write_text(json.dumps(force,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--case',choices=['small','hk'],default='small')
    parser.add_argument('--output',required=True)
    parser.add_argument('--domain')
    parser.add_argument('--force')
    parser.add_argument('--source')
    parser.add_argument('--disabled',action='store_true')
    parser.add_argument('--backend',choices=['cpu','metal'],default='cpu')
    parser.add_argument('--reference-repeats',type=int,default=1)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.reference_repeats < 1:
        parser.error('--reference-repeats must be positive')
    if args.source:
        worker(args)
        return
    if args.backend == 'metal':
        from gate_gpu_tools import require_gpu_window
        gate_window = require_gpu_window()
    output=Path(args.output).resolve()
    prepare(args,output)
    baseline=output/'baseline-source'
    baseline.mkdir()
    archive=subprocess.check_output(['git','archive','7648185','src'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(baseline)
    results={}
    runs = [('7648185' if i == 0 else f'7648185-repeat{i+1}',baseline,False)
            for i in range(args.reference_repeats)]
    runs += [('candidate-no-gate',Path.cwd(),False),('candidate-disabled',Path.cwd(),True)]
    for name,source,disabled in runs:
        target=output/name
        target.mkdir()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--source',str(source),
            '--domain',str(output/'domain.json'),'--force',str(output/'force.json'),'--output',str(target),
            '--backend',args.backend]
        if disabled:
            command.append('--disabled')
        if args.profile:
            command.append('--profile')
        with (target/'run.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        results[name]=json.loads((target/'fingerprint.json').read_text())
        print(f'{name}: {results[name]["steps"]} {args.backend} steps completed',flush=True)
    reference=results['7648185']
    required = ('depth','q','bed','dt_sequence','trace')
    physics_equal = all(all(result['hashes'][key] == reference['hashes'][key] for key in required)
                        for result in results.values())
    all_equal = all(result['hashes'] == reference['hashes'] for result in results.values())
    data = dict(case=args.case,backend=args.backend,cpu_threads=1,
        bitwise_equal=all_equal,physics_bitwise_equal=physics_equal,results=results)
    if args.backend == 'metal':
        data.update(gpu_window=gate_window, budget_variability=budget_differences(output,list(results)))
    (output/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    assert physics_equal and (args.backend == 'metal' or all_equal), data
    print(('BITWISE_EQUAL ' if all_equal else 'PHYSICS_BITWISE_EQUAL_WITH_BUDGET_VARIATION ')
          +str(output/'comparison.json'),flush=True)


if __name__=='__main__':
    main()
