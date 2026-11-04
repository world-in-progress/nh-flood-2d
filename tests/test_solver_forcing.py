"""Small physical experiments exercise the actual Taichi evolution path."""
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint


pytestmark = pytest.mark.usefixtures('taichi_cpu')

START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


def test_next_cfl_uses_depth_after_the_level_update(tmp_path, monkeypatch):
    from src.nh_flood_2d.core.clock import StepClock
    original = StepClock.read_trace
    traces = []

    def capture(clock):
        rows = original(clock)
        traces.extend(rows[rows[:, 3] == clock.COMPLETED].copy())
        return rows

    monkeypatch.setattr(StepClock, 'read_trace', capture)
    experiment(tmp_path, [[0, 0]], 1, connected=True, level=0,
               level_rows=[(0, 0), (.000002, 100), (1, 100)])
    candidates = np.array(traces, dtype=np.uint32)[:, 4].copy().view(np.float32)
    # The first event step raises cell 2 from dry to 100 m; q is still zero.
    assert candidates[1] == pytest.approx(.5 * 10 / np.sqrt(9.81 * 100), rel=2e-7)
    assert candidates[1] < 1
    rows = list(csv.DictReader((tmp_path / 'out/step_statistics.csv').open()))
    last = rows[-1]
    assert float(last['face_depth_m']) == max(float(last['lower_depth_m']),
                                             float(last['higher_depth_m']))
    expected = .5 * 10 / (np.sqrt(9.81 * float(last['face_depth_m']))
                          + abs(float(last['face_velocity_m_s'])))
    assert float(last['cfl_min_s']) == pytest.approx(expected, rel=2e-7)


def test_all_event_steps_have_explicit_standard_statistics(tmp_path):
    experiment(tmp_path, [[0, 0]], 1, q=.1,
               q_rows=[(0, .1), (.000002, .1), (1, .1)])
    with (tmp_path / 'out/step_statistics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    last = rows[-1]
    assert last['steps'] == last['event_short_steps'] == '2'
    assert last['standard_steps'] == '0'
    for name in ['standard_dt_min_s', 'standard_dt_max_s', 'standard_dt_mean_s']:
        assert last[name] == ''
    assert 'dt_min_s' not in last and 'dt_max_s' not in last and 'dt_mean_s' not in last
    assert float(last['event_dt_min_s']) == pytest.approx(.000002)
    assert float(last['event_dt_sum_s']) == pytest.approx(1., abs=1e-7)


def experiment(path, bands, duration, land_type=10, q=0., warmup=0., level=None,
               bed=0., connected=False, gate=None, level_rows=None, q_rows=None,
               yield_step=None, warmup_only=False, infiltration='none', stations=None):
    """Two isolated 10 x 10 m cells make spatial infiltration directly observable."""
    path.mkdir(parents=True, exist_ok=True)
    ne, ns = [], []
    for ei, x in enumerate([5., 15.], start=1):
        first = 4 * (ei - 1) + 1
        ne.append(f'{ei},1,1,1,1,{first},{first+1},{first+2},{first+3},{x},5,{bed},{land_type}\n')
        ns += [f'{first},2,0,{ei},0,0,10,{x-5},5,0,0\n',
               f'{first+1},2,{ei},0,0,0,10,{x+5},5,0,0\n',
               f'{first+2},1,0,0,0,{ei},10,{x},0,0,0\n',
               f'{first+3},1,0,0,{ei},0,10,{x},10,0,0\n']
    if connected:
        ne[1] = ne[1].replace('2,1,1,1,1,5,6,7,8,', '2,1,1,1,1,2,6,7,8,')
        ns[1] = '2,2,1,2,0,0,10,10,5,0,0\n'
        ns = [row for row in ns if not row.startswith('5,')]
    (path/'ne.txt').write_text(''.join(ne))
    (path/'ns.txt').write_text(''.join(ns))
    values = np.array(bands, dtype='float32').reshape(-1, 1, 2)
    with rasterio.open(path/'rain.tif', 'w', driver='GTiff', width=2, height=1,
                       count=len(values), dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 10, 10, 10)) as out:
        out.write(values)
    with open(path/'intervals.csv', 'w', newline='') as out:
        w=csv.writer(out);w.writerow(['interval_start','interval_end'])
        for i in range(len(values)):
            a = i * duration / len(values)
            b = (i+1)*duration/len(values)
            w.writerow([(START+timedelta(seconds=a)).isoformat(),(START+timedelta(seconds=b)).isoformat()])
    boundaries=[]
    for name, val, kind, x in [('river',q,'discharge',0),('sea',level,'level',20)]:
        if val is None or (kind=='discharge' and q==0): continue
        (path/f'{name}.geojson').write_text(json.dumps({'type':'LineString','coordinates':[[x,1],[x,9]],
             'crs':{'type':'name','properties':{'name':'EPSG:2326'}}}))
        (path/f'{name}.csv').write_text('time,value\n'+
            f'{START.isoformat()},{val}\n'+
            f'{(START+timedelta(seconds=duration)).isoformat()},{val}\n')
        if name == 'river' and q_rows:
            (path/f'{name}.csv').write_text('time,value\n' + ''.join(
                f'{(START+timedelta(seconds=t)).isoformat()},{v}\n' for t,v in q_rows))
        if name == 'sea' and level_rows:
            (path/f'{name}.csv').write_text('time,value\n' + ''.join(
                f'{(START+timedelta(seconds=t)).isoformat()},{v}\n' for t,v in level_rows))
        boundaries.append(dict(name=name,kind=kind,line=str(path/f'{name}.geojson'),series=str(path/f'{name}.csv')))
    dc=DomainConfig(ne=str(path/'ne.txt'),ns=str(path/'ns.txt'),epsg_code=2326,domain_dir=str(path/'out'),
                    start=START,duration=duration,warmup_duration=warmup,
                    land_parameters={land_type:dict(manning_n=.03,infiltration=infiltration)},
                    yield_step=yield_step or duration,stations=stations or {})
    fc=ForceConfig(boundaries=boundaries,gate=gate,rainfall=dict(raster=str(path/'rain.tif'),intervals=str(path/'intervals.csv')))
    preprocess(dc,fc)
    if warmup_only:
        from src.nh_flood_2d.core import warmup_solver
        result=warmup_solver(dc,fc)
    else:
        result=solver(dc,fc)
    state=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
    h=state.bed+state.depth
    return result,h,json.loads((path/'out/manifest.json').read_text())


def test_closed_basin_rain_and_discharge_balance(tmp_path):
    result, h, manifest = experiment(tmp_path, [[6,6]], 60, q=.1, connected=True)
    assert result['rainfall_m3'] == pytest.approx(1.2, abs=1e-8)
    assert result['discharge:river_m3'] == pytest.approx(6., abs=1e-7)
    assert abs(result['residual_m3']) / 7.2 < 1e-4
    assert result['storage_m3'] == pytest.approx(7.2, rel=1e-4)
    assert np.all(np.isfinite(h))
    assert manifest['status']=='completed' and manifest['steps'] >= 60
    assert len(manifest['input_sha256']) == 6
    print(f'BASIN residual={result["residual_m3"]:.12g} m3 relative={result["relative_residual"]:.12g}')


def test_horton_clocks_are_local(tmp_path):
    horton = dict(f0_mm_h=76.2, fc_mm_h=2.54, k_per_h=2)
    _, h_late, _ = experiment(tmp_path/'late', [[100,0],[0,100]], 120,
                              land_type=1, infiltration=horton)
    _, h_reference, _ = experiment(tmp_path/'reference', [[0,100]], 60,
                                   land_type=1, infiltration=horton)
    # A cell that was dry while its neighbour rained must retain its initial Horton capacity.
    assert h_late[2] == pytest.approx(h_reference[2], abs=1e-6)


def test_actual_infiltration_is_bounded_by_water(tmp_path):
    result, h, _ = experiment(tmp_path, [[.001,.001]], 60, land_type=6,
                              infiltration=dict(f0_mm_h=101.6, fc_mm_h=7.62, k_per_h=2))
    assert result['infiltration_m3'] == pytest.approx(result['rainfall_m3'], rel=2e-6)
    assert result['bed_clamp_nonlevel_m3'] == 0
    assert result['bed_clamp_level_m3'] == 0
    assert np.max(h) == 0


def test_configured_constant_horton_capacity_changes_actual_infiltration(tmp_path):
    high, _, _ = experiment(tmp_path/'high', [[100,100]], 60,
                            infiltration=dict(f0_mm_h=36, fc_mm_h=36, k_per_h=0))
    low, _, _ = experiment(tmp_path/'low', [[100,100]], 60,
                           infiltration=dict(f0_mm_h=18, fc_mm_h=18, k_per_h=0))
    none, _, _ = experiment(tmp_path/'none', [[100,100]], 60)
    assert high['infiltration_m3'] == pytest.approx(.12, rel=1e-4)
    assert low['infiltration_m3'] == pytest.approx(.06, rel=1e-4)
    assert none['infiltration_m3'] == 0.


def test_warmup_holds_boundaries_and_has_no_rain(tmp_path):
    result, _, _ = experiment(tmp_path, [[6,6]], 60, q=.01, warmup=20, level=.1,
        level_rows=[(0,.1),(60,.2)], q_rows=[(0,.01),(60,.03)], yield_step=10)
    rows=list(csv.DictReader((tmp_path/'out/mass_balance.csv').open()))
    at_start=next(r for r in rows if float(r['elapsed_seconds'])==0)
    assert float(at_start['rainfall_m3']) == 0
    assert float(at_start['discharge:river_m3']) == pytest.approx(.2)
    assert result['rainfall_m3'] == pytest.approx(1.2)
    assert result['discharge:river_m3'] == pytest.approx(1.4)
    assert [float(r['elapsed_seconds']) for r in rows if r['phase']=='warmup']==[-20.,-10.]
    state=read_checkpoint(tmp_path/'out/output/state_start.fdb')
    assert state.depth[2] == pytest.approx(.1)
    assert abs(result['relative_residual']) < 1e-4


@pytest.mark.parametrize('bed,land_type', [(-5.,10),(1.,8)])
def test_cold_start_is_dry_without_boundaries(tmp_path, bed, land_type):
    result, h, _ = experiment(tmp_path, [[0,0]], 2, bed=bed, land_type=land_type)
    assert result['initial_storage_m3'] == 0.
    assert result['storage_m3'] == 0.
    np.testing.assert_array_equal(h[1:], [bed,bed])


def test_warmup_only_stops_at_start_and_outputs_each_interval(tmp_path):
    result, _, manifest = experiment(tmp_path, [[6,6]], 60, q=.01, warmup=20,
        level=.1, yield_step=10, warmup_only=True)
    rows=list(csv.DictReader((tmp_path/'out/mass_balance.csv').open()))
    assert [float(r['elapsed_seconds']) for r in rows]==[-20.,-10.,0.]
    assert all(r['phase']=='warmup' for r in rows)
    assert result['rainfall_m3']==0.
    assert result['discharge:river_m3']==pytest.approx(.2)
    assert manifest['run_kind']=='warmup'
    assert manifest['dt_seconds']['sum']==pytest.approx(20.)
    assert manifest['simulation_end']==START.isoformat()
    assert sorted(p.stem for p in (tmp_path/'out/output/frames').glob('*.fdb')) == ['depth_-20', 'depth_0']
    assert (tmp_path/'out/output/state_final.fdb').exists()


def test_small_rain_is_retained_above_large_bed(tmp_path):
    result, _, _ = experiment(tmp_path, [[.01,.01]], 60, bed=100.)
    # Depth stores small increments independently of the 100 m bed.
    assert result['storage_m3'] == pytest.approx(.002, rel=2e-6)
    assert abs(result['residual_m3']) < 4e-9


def test_existing_output_is_not_overwritten(tmp_path):
    from src.nh_flood_2d.core.run_record import RunRecord
    result, _, _ = experiment(tmp_path, [[0,0]], 1)
    manifest = tmp_path/'out/manifest.json'
    balance = tmp_path/'out/mass_balance.csv'
    before = (manifest.read_bytes(),balance.read_bytes())
    dc = DomainConfig(ne=str(tmp_path/'ne.txt'),ns=str(tmp_path/'ns.txt'),epsg_code=2326,
                     domain_dir=str(tmp_path/'out'),start=START,duration=1,warmup_duration=0,
                     land_parameters={10:dict(manning_n=.03,infiltration='none')})
    fc = ForceConfig(boundaries=[],rainfall=dict(raster='missing',intervals='missing'))
    with pytest.raises(FileExistsError):
        RunRecord(dc,fc)
    assert before==(manifest.read_bytes(),balance.read_bytes())


def test_gate_detects_short_threshold_crossing(tmp_path):
    result, h, manifest = experiment(tmp_path, [[0,0]], 2, bed=1., land_type=1,
        level=1.4, level_rows=[(0,1.4),(1,1.6),(2,1.4)],
        gate=dict(threshold=1.5,land_type=1,level_boundary='sea'))
    assert manifest['gate_open']
    assert manifest['gate_cell_count'] == 2
    assert result['gate_bed_m3'] == 0.
    assert h[1] == 0.


def test_input_failure_is_recorded(tmp_path):
    from src.nh_flood_2d.core.run_record import RunRecord
    (tmp_path/'ne').write_text('mesh')
    (tmp_path/'ns').write_text('mesh')
    dc=DomainConfig(ne=str(tmp_path/'ne'),ns=str(tmp_path/'ns'),epsg_code=2326,
        domain_dir=str(tmp_path/'out'),start=START,duration=1,warmup_duration=0,
        land_parameters={10:dict(manning_n=.03,infiltration='none')})
    fc=ForceConfig(boundaries=[],rainfall=dict(raster='missing',intervals='missing'))
    with pytest.raises(FileNotFoundError):
        solver(dc,fc)
    assert json.loads((tmp_path/'out/manifest.json').read_text())['status']=='failed'


def test_fractional_output_interval_is_rejected(tmp_path):
    from pydantic import ValidationError
    (tmp_path/'ne').write_text('mesh')
    (tmp_path/'ns').write_text('mesh')
    with pytest.raises(ValidationError):
        DomainConfig(ne=str(tmp_path/'ne'),ns=str(tmp_path/'ns'),epsg_code=2326,
            domain_dir=str(tmp_path/'out'),start=START,duration=1,warmup_duration=0,
            land_parameters={10:dict(manning_n=.03,infiltration='none')},yield_step=.5)


def test_level_accounts_for_small_rain_at_high_bed(tmp_path):
    result, _, _ = experiment(tmp_path, [[0,.01]], 60, bed=99., level=100.)
    assert abs(result['residual_m3']) < 1e-8
    assert result['level:sea_m3'] == pytest.approx(100.-.001, abs=1e-8)


def test_level_below_bed_is_zero_depth_and_closes_budget(tmp_path):
    result, _, _ = experiment(tmp_path, [[0,.01]], 60, bed=100., level=50.)
    assert result['storage_m3'] == 0.
    assert result['bed_clamp_nonlevel_m3'] == 0.
    assert result['bed_clamp_level_m3'] == 0.
    assert result['rainfall_m3'] == pytest.approx(.001, abs=1e-10)
    assert abs(result['residual_m3']) < 1e-7


def test_cuda_manifest_records_gpu_instead_of_host_cpu(monkeypatch):
    from types import SimpleNamespace
    import taichi as ti
    from src.nh_flood_2d.core import run_record
    monkeypatch.setattr(ti.lang.impl, 'current_cfg', lambda: SimpleNamespace(arch=ti.cuda))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1,0')
    queries=[]
    def query(command, **kwargs):
        queries.append(command)
        return 'Test NVIDIA GPU\n'
    monkeypatch.setattr(run_record.subprocess,'check_output',query)
    record=run_record.RunRecord.__new__(run_record.RunRecord)
    record.data={}
    record.device()
    assert record.data=={'backend':'cuda','device_name':'Test NVIDIA GPU'}
    assert queries==[['nvidia-smi','--id=1','--query-gpu=name','--format=csv,noheader']]


def cell(row: int, col: int) -> int:
    return row * 3 + col + 1

def vertical(row: int, col: int) -> int:
    return row * 4 + col + 1

def horizontal(row: int, col: int) -> int:
    return 13 + row * 3 + col

def connected_grid(path: Path) -> None:
    ne, ns = [], []
    for row in range(3):
        for col in range(3):
            ei = cell(row, col)
            ne.append(f'{ei},1,1,1,1,{vertical(row,col)},{vertical(row,col+1)},'
                      f'{horizontal(row,col)},{horizontal(row+1,col)},'
                      f'{5+10*col},{5+10*row},0,10\n')
    for row in range(3):
        for col in range(4):
            left = cell(row, col-1) if col else 0
            right = cell(row, col) if col < 3 else 0
            ns.append(f'{vertical(row,col)},2,{left},{right},0,0,10,'
                      f'{10*col},{5+10*row},0,0\n')
    for row in range(4):
        for col in range(3):
            bottom = cell(row-1, col) if row else 0
            top = cell(row, col) if row < 3 else 0
            ns.append(f'{horizontal(row,col)},1,0,0,{bottom},{top},10,'
                      f'{5+10*col},{10*row},0,0\n')
    (path/'ne.txt').write_text(''.join(ne))
    (path/'ns.txt').write_text(''.join(ns))


def test_theta_less_than_one_keeps_previous_side_flows_stable(tmp_path):
    from src.nh_flood_2d.core.run_record import sha256
    frames=[]
    for name in ('first','repeat'):
        path=tmp_path/name
        path.mkdir()
        connected_grid(path)
        rows=[r.split(',') for r in (path/'ne.txt').read_text().splitlines()]
        xs=[float(r[-4]) for r in rows];ys=[float(r[-3]) for r in rows]
        with rasterio.open(path/'rain.tif','w',driver='GTiff',width=1,height=1,count=1,
            dtype='float32',crs='EPSG:2326',transform=from_origin(min(xs)-5,max(ys)+5,
            max(xs)-min(xs)+10,max(ys)-min(ys)+10)) as dst:
            dst.write(np.zeros((1,1),dtype='float32'),1)
        (path/'times.csv').write_text('interval_start,interval_end\n'+
            f'{START.isoformat()},{(START+timedelta(seconds=20)).isoformat()}\n')
        depth=np.zeros(10,dtype='float32');depth[5]=.2
        write_checkpoint(path/'restart.fdb',depth=depth,bed=np.zeros(10,dtype='float32'),
            q=np.zeros(len((path/'ns.txt').read_text().splitlines())+1,dtype='float32'),
            rain_time=np.zeros(10,dtype='float32'),
            infiltration=np.zeros(10,dtype='float32'),max_depth=depth,
            first_time_hi=np.zeros(10,dtype='uint32'),first_time_lo=np.zeros(10,dtype='uint32'),
            max_speed=np.zeros(10,dtype='float32'),clock_hi=0,clock_lo=0,
            gate_open=False,metadata_json=json.dumps(dict(run_state=False,
                ne_sha256=sha256(path/'ne.txt'),ns_sha256=sha256(path/'ns.txt'),gate_config='null')))
        dc=DomainConfig(ne=str(path/'ne.txt'),ns=str(path/'ns.txt'),domain_dir=str(path/'out'),
            epsg_code=2326,start=START,duration=20,warmup_duration=0,
            land_parameters={10:dict(manning_n=.033,infiltration='none')},
            sita=.7,yield_step=20,restart_state=str(path/'restart.fdb'))
        fc=ForceConfig(boundaries=[],rainfall=dict(raster=str(path/'rain.tif'),intervals=str(path/'times.csv')))
        preprocess(dc,fc);budget=solver(dc,fc)
        assert abs(budget['residual_m3'])/20 < 1e-4
        frames.append(read_checkpoint(Path(dc.output_dir)/'state_final.fdb').depth)
    np.testing.assert_array_equal(*frames)
    assert np.all(np.isfinite(frames[0]))


def test_negative_discharge_shares_available_water_without_creating_water(tmp_path):
    result, h, _ = experiment(tmp_path, [[6,6]], 60, q=-10.)
    assert result['discharge:river_m3'] == pytest.approx(-.6, abs=2e-6)
    assert result['requested_discharge:river_m3'] == pytest.approx(-600.)
    assert result['storage_m3'] == pytest.approx(.6, abs=2e-6)
    assert result['bed_clamp_nonlevel_m3'] < 1e-6
    assert abs(result['residual_m3']) < 2e-6


def test_horton_uses_f0_on_wet_cell_before_any_rain(tmp_path):
    result, _, _ = experiment(tmp_path, [[0,0]], 60, level=.1,
        infiltration=dict(f0_mm_h=76.2,fc_mm_h=2.54,k_per_h=2))
    assert result['infiltration_m3'] == pytest.approx(76.2/1000/3600*100*60,rel=2e-6)


def test_true_small_cfl_stops_with_limiting_edge_diagnostics(tmp_path):
    with pytest.raises(RuntimeError, match=r'CFL.*edge=2, cells=\[1, 2\].*depth_m=.*velocity_m_s='):
        experiment(tmp_path, [[0,0]], 1, level=1e9, connected=True)


def test_subsecond_rain_knot_selects_the_next_band(tmp_path):
    path=tmp_path/'rain_knot'
    # Start with the standard two-pixel input, then run a fresh configuration
    # whose first rainfall band ends at an ISO-representable 2 microseconds.
    experiment(path, [[1,0],[0,2]], 2)
    (path/'intervals.csv').write_text('interval_start,interval_end\n'+
        f'{START.isoformat()},{(START+timedelta(microseconds=2)).isoformat()}\n'+
        f'{(START+timedelta(microseconds=2)).isoformat()},{(START+timedelta(seconds=1)).isoformat()}\n')
    dc=DomainConfig(ne=str(path/'ne.txt'),ns=str(path/'ns.txt'),epsg_code=2326,
        domain_dir=str(path/'knot_out'),start=START,duration=1,warmup_duration=0,
        land_parameters={10:dict(manning_n=.03,infiltration='none')},yield_step=1)
    fc=ForceConfig(boundaries=[],rainfall=dict(raster=str(path/'rain.tif'),intervals=str(path/'intervals.csv')))
    preprocess(dc,fc);result=solver(dc,fc)
    state=read_checkpoint(Path(dc.output_dir)/'state_final.fdb')
    np.testing.assert_allclose(state.depth[1:],[.001,.002],rtol=2e-6,atol=0)
    assert result['storage_m3']==pytest.approx(.3,rel=2e-6)


def test_subsecond_level_knot_and_between_frame_peak(tmp_path):
    from src.nh_flood_2d.core.output import read_extrema,read_depth_frame,list_depth_frames
    _,h,_=experiment(tmp_path,[[0,0]],1,level=0.,
        level_rows=[(0,0.),(.000002,1.),(1,0.)])
    assert abs(h[2])<2e-6
    root=tmp_path/'out/output'
    extrema=read_extrema(root/'extrema.fdb')
    assert extrema.max_depth[2]==pytest.approx(1.,abs=2e-6)
    assert all(read_depth_frame(p)[2]<2e-6 for _,p in list_depth_frames(root))
    peak=(int(extrema.first_time_hi[2])*2**32+int(extrema.first_time_lo[2]))/2**40
    assert peak==pytest.approx(.000002,abs=5e-13)


@pytest.mark.parametrize('tail', [ [(1,1.6)], [(.000008,1.4),(1,1.4)] ])
def test_gate_uses_exact_upcrossing_without_lookahead_or_missing_pulse(tmp_path,tail):
    _,_,manifest=experiment(tmp_path,[[0,0]],1,bed=1.,land_type=1,level=1.4,
        level_rows=[(0,1.4),(.000004,1.6),*tail],
        gate=dict(threshold=1.5,land_type=1,level_boundary='sea'))
    assert manifest['gate_open']
    assert len(manifest['bed_events'])==1
    event=manifest['bed_events'][0]
    assert event['elapsed_seconds']==pytest.approx(.000002,abs=5e-13)
    assert event['cell_ids']==[1,2]
    from src.nh_flood_2d.core.output import read_bed_at
    root=tmp_path/'out/output'
    assert read_bed_at(root,0)[1]==1.
    assert read_bed_at(root,1)[1]==0.


def test_initial_gate_event_precedes_frame_and_level_application(tmp_path):
    result,_,manifest=experiment(tmp_path,[[0,0]],1,bed=1.,land_type=1,level=1.6,
        gate=dict(threshold=1.5,land_type=1,level_boundary='sea'))
    from src.nh_flood_2d.core.output import read_bed_at,read_depth_frame
    root=tmp_path/'out/output'
    bed=read_bed_at(root,0)
    depth=read_depth_frame(root/'frames/depth_0.fdb')
    assert manifest['bed_events'][0]['elapsed_seconds']==0.
    assert bed[2]==0.
    assert float(bed[2]+depth[2])==pytest.approx(1.6,abs=2e-6)
    assert abs(result['residual_m3'])<1e-3


def test_gate_at_output_time_is_reflected_in_station_and_budget(tmp_path):
    result,_,manifest=experiment(tmp_path,[[0,0]],2,bed=1.,land_type=1,level=1.4,
        level_rows=[(0,1.4),(1,1.5),(2,1.6)],yield_step=1,
        stations={'sea':(15.,5.)},gate=dict(threshold=1.5,land_type=1,level_boundary='sea'))
    rows=list(csv.DictReader((tmp_path/'out/output/stations.csv').open()))
    at_event=next(r for r in rows if float(r['elapsed_seconds'])==1.)
    assert float(at_event['depth'])==pytest.approx(1.5,abs=2e-6)
    assert float(at_event['level'])==pytest.approx(1.5,abs=2e-6)
    assert manifest['bed_events'][0]['elapsed_seconds']==1.
    assert result['gate_bed_m3']==0.
    assert abs(result['residual_m3'])<1e-3
