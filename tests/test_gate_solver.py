"""Mechanism tests use the production graph, limiter, clock and checkpoint I/O."""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint, read_bed_at
from src.nh_flood_2d.core.run_record import RunRecord, sha256
from src.nh_flood_2d.core.solver_compact import _solve
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess
from test_pressure_gradient import square_mesh
from test_solver_forcing import START

pytestmark = pytest.mark.usefixtures('taichi_cpu')


def setup(path, *, cells=None, types=None, bed=None, depth=None, gate=None, rain=0,
          warmup=0, yield_step=1, boundaries=None):
    path.mkdir(parents=True, exist_ok=True)
    cells = cells or [(i*10, (i+1)*10, 0, 10) for i in range(3)]
    types = types or [1, 11, 10]
    bed = np.r_[0, bed if bed is not None else [0]*len(cells)].astype('float32')
    depth = np.r_[0, depth if depth is not None else [2]*len(cells)].astype('float32')
    xy, _, topo, _ = square_mesh(path, cells, lambda x, y, i: bed[i])
    lines = (path/'ne.txt').read_text().splitlines()
    (path/'ne.txt').write_text('\n'.join(line.rsplit(',', 1)[0]+f',{kind}'
        for line, kind in zip(lines, types))+'\n')
    config = dict(land_type=11, pond_land_type=10, enabled=True, trigger_level=1,
        opening_seconds=1, open_elevation=-.1, gates=[dict(name='A',
            inside_point=xy[2].tolist(), control_point=xy[1].tolist())])
    if gate is not None:
        config.update(gate)
    x1, y1 = max(c[1] for c in cells), max(c[3] for c in cells)
    with rasterio.open(path/'rain.tif', 'w', driver='GTiff', width=1, height=1,
        count=1, dtype='float32', crs='EPSG:2326', transform=from_origin(0, y1, x1, y1)) as dst:
        dst.write(np.array([[rain]], dtype='float32'), 1)
    (path/'intervals.csv').write_text('interval_start,interval_end\n'
        '2023-09-07T08:00:00Z,2023-09-07T09:00:00Z\n')
    force = ForceConfig(boundaries=boundaries or [], gate=config,
        rainfall=dict(raster=str(path/'rain.tif'), intervals=str(path/'intervals.csv')))
    zeros, words = np.zeros(len(depth), 'float32'), np.zeros(len(depth), 'uint32')
    seed = path/'seed.fdb'
    write_checkpoint(seed, depth=depth, bed=bed, q=np.zeros(len(topo), 'float32'),
        rain_time=zeros, infiltration=zeros, max_depth=depth, max_speed=zeros,
        first_time_hi=words, first_time_lo=words, clock_hi=0, clock_lo=0,
        gate_open=False, metadata_json=json.dumps(dict(run_state=False,
            ne_sha256=sha256(path/'ne.txt'), ns_sha256=sha256(path/'ns.txt'),
            gate_config=force.gate.model_dump_json())))
    domain = DomainConfig(ne=str(path/'ne.txt'), ns=str(path/'ns.txt'), epsg_code=2326,
        domain_dir=str(path/'out'), start=START, duration=3600,
        warmup_duration=warmup, yield_step=yield_step, restart_state=str(seed),
        land_parameters={t: dict(manning_n=0, infiltration='none') for t in set(types)})
    return domain, force, topo, depth, bed


def run(domain, force, seconds, *, output=None, restart=None):
    # Public durations are whole seconds; mechanism tests request exact clock ticks.
    seconds = round(seconds * 2**40) / 2**40
    changes = {}
    if output is not None:
        changes['domain_dir'] = str(output)
    if restart is not None:
        changes['restart_state'] = str(restart)
    domain = domain.model_copy(update=changes)
    Path(domain.domain_dir).mkdir(parents=True, exist_ok=True)
    with RunRecord(domain, force) as record:
        record.hash_inputs(domain, force)
        preprocess(domain, force)
        _solve(domain, force, record, end_time=seconds)
    root = Path(domain.domain_dir)
    state = read_checkpoint(root/'output/state_final.fdb')
    return state, json.loads((root/'manifest.json').read_text())


def test_local_bed_motion_preserves_water_and_drives_expected_exchange(tmp_path):
    domain, force, topo, depth, bed = setup(tmp_path, gate=dict(opening_seconds=.1))
    # First step moves the bed after the hydraulic substep; the next uses it.
    run(domain, force, .1)
    state, manifest = run(domain, force, .2, output=tmp_path/'next',
                         restart=tmp_path/'out/output/state_final.fdb')
    internal = np.flatnonzero((topo[:,1]>0) & (topo[:,2]>0))
    assert state.q[internal[0]]*10 == pytest.approx(.1962, rel=2e-5)
    assert np.sum(state.depth)*100 == pytest.approx(600, abs=3e-5)
    assert state.depth[2] > state.depth[1]
    assert manifest['gate_control']['gates'][0]['command_seconds'] == 0
    assert manifest['bed_events'] == []


@pytest.mark.parametrize('control_depth,threshold,expected', [(2, 1, 0), (2, 2, None),
    (0, -1, None), (.001, -1, None), (2, 3, None)])
def test_strict_threshold_and_dry_control(tmp_path, control_depth, threshold, expected):
    domain, force, *_ = setup(tmp_path, depth=[control_depth, 0, 0],
        bed=[0, 10, 0], gate=dict(trigger_level=threshold))
    _, manifest = run(domain, force, .1)
    assert manifest['gate_control']['gates'][0]['command_seconds'] == expected


def test_end_bed_and_pond_exchange_accounting(tmp_path):
    domain, force, topo, initial, bed = setup(tmp_path, depth=[2,2,1])
    state, manifest = run(domain, force, 3)
    gate = manifest['gate_control']['gates'][0]
    pond = manifest['gate_control']['pond_budget']
    assert state.bed[2] == pytest.approx(-.1)
    assert gate['pond_in_m3'] > 0
    assert gate['first_effective_flow_seconds'] is not None
    assert gate['fully_open_seconds'] == 1
    assert abs(pond['residual_m3']) < 3e-5
    assert pond['storage_m3'] - pond['initial_storage_m3'] == pytest.approx(
        gate['pond_net_m3'] + pond['rounding_m3'], abs=3e-5)
    replayed = read_bed_at(tmp_path/'out/output', 3)
    np.testing.assert_array_equal(replayed, state.bed)


def level_boundary(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    line, series = path/'level.geojson', path/'level.csv'
    line.write_text(json.dumps(dict(type='LineString', coordinates=[[0,0],[0,10]],
        crs=dict(type='name', properties=dict(name='EPSG:2326')))))
    series.write_text('time,value\n'+''.join(
        f'{(START+timedelta(seconds=t)).isoformat()},{h}\n' for t,h in rows))
    return [dict(name='river', kind='level', line=str(line), series=str(series))]


@pytest.mark.parametrize('batch_size,yield_step', [(1,1), (32,1), (32,2)])
def test_short_pulse_latches_each_step_and_never_recloses(tmp_path, monkeypatch, batch_size, yield_step):
    from src.nh_flood_2d.core.clock import StepClock
    original = StepClock.__init__
    def init(clock, initial_seconds, **kwargs):
        original(clock, initial_seconds, batch_size=batch_size)
    monkeypatch.setattr(StepClock, '__init__', init)
    boundaries = level_boundary(tmp_path, [(0,1), (.125,2), (.25,1), (3600,1)])
    domain, force, *_ = setup(tmp_path, bed=[0,10,0], depth=[1,0,0],
        gate=dict(trigger_level=1.5, opening_seconds=1.875, open_elevation=0),
        boundaries=boundaries, yield_step=yield_step)
    state, manifest = run(domain, force, 2)
    gate = manifest['gate_control']['gates'][0]
    assert gate['command_seconds'] == .125
    assert gate['fully_open_seconds'] == 2
    assert gate['first_fully_open_sample_seconds'] == 2
    assert state.bed[2] == 0
    assert abs(manifest['gate_control']['pond_budget']['residual_m3']) < 1e-5


def test_gates_trigger_independently(tmp_path):
    domain, force, *_ = setup(tmp_path,
        cells=[(i*10,(i+1)*10,0,10) for i in range(6)],
        types=[1,11,10,1,11,10], depth=[2,0,0,0,0,0], bed=[0,10,0,0,10,0],
        gate=dict(gates=[dict(name='A', inside_point=(15,5), control_point=(5,5)),
                         dict(name='B', inside_point=(45,5), control_point=(35,5))]))
    state, manifest = run(domain, force, 1)
    gates = manifest['gate_control']['gates']
    assert [g['command_seconds'] for g in gates] == [0, None]
    assert state.bed[2] < 10 and state.bed[5] == 10


def test_warmup_disables_control_until_formal_zero(tmp_path):
    domain, force, *_ = setup(tmp_path, warmup=2, depth=[2,0,0], bed=[0,10,0])
    _, warming = run(domain, force, 0)
    assert warming['gate_control']['gates'][0]['command_seconds'] is None
    state, formal = run(domain, force, 1, output=tmp_path/'formal',
                         restart=tmp_path/'out/output/state_final.fdb')
    assert formal['gate_control']['gates'][0]['command_seconds'] == 0
    assert state.bed[2] == np.float32(-.1)


@pytest.mark.parametrize('moving', [False, True])
def test_fixed_still_water_and_uniform_motion_have_zero_flux(tmp_path, moving):
    domain, force, _, depth, bed = setup(tmp_path,
        cells=[(100,110,0,10), (10,20,0,10), (20,30,0,10), (30,40,0,10), (0,10,0,10)],
        types=[1,11,11,10,1], bed=[0,0,0,10,10], depth=[2,2,2,0,0],
        gate=dict(trigger_level=1 if moving else 3))
    state, _ = run(domain, force, 2)
    np.testing.assert_array_equal(state.depth, depth)
    assert np.all(state.q == 0)
    np.testing.assert_array_equal(state.bed[2:4], np.full(2, -.1 if moving else 0, dtype='float32'))


@pytest.mark.parametrize('split_at', [1,3,5])
def test_restart_before_during_after_opening_is_bitwise(tmp_path, split_at):
    boundaries = level_boundary(tmp_path, [(0,1), (1,1), (2,2), (3600,2)])
    domain, force, *_ = setup(tmp_path, boundaries=boundaries,
        depth=[1,0,0], bed=[0,3,0], gate=dict(trigger_level=1.5, opening_seconds=2))
    full, full_manifest = run(domain, force, 6)
    partial, _ = run(domain, force, split_at, output=tmp_path/'part')
    resumed, resumed_manifest = run(domain, force, 6, output=tmp_path/'resumed',
        restart=tmp_path/'part/output/state_final.fdb')
    for name in ('depth','bed','q','rain_time','infiltration','max_depth','max_speed',
                 'first_time_hi','first_time_lo'):
        assert getattr(full,name).tobytes() == getattr(resumed,name).tobytes(), name
    assert full.metadata_json == resumed.metadata_json
    assert full_manifest['gate_control'] == resumed_manifest['gate_control']
    for t in (split_at,6):
        np.testing.assert_array_equal(read_bed_at(tmp_path/'out/output',t),
                                      read_bed_at(tmp_path/'resumed/output',t))
    np.testing.assert_array_equal(read_bed_at(tmp_path/'out/output',split_at),partial.bed)
    import csv
    def rows(path):
        with path.open() as stream:
            return [r for r in csv.DictReader(stream) if float(r['elapsed_seconds']) > split_at]
    assert rows(tmp_path/'out/gate_balance.csv') == rows(tmp_path/'resumed/gate_balance.csv')


@pytest.mark.parametrize('pond_depth,gate_depth', [(0,2), (2,.021), (1,1.00001), (2,1)])
def test_dry_near_dry_small_and_reverse_heads(tmp_path, pond_depth, gate_depth):
    domain, force, *_ = setup(tmp_path, depth=[gate_depth,gate_depth,pond_depth],
        gate=dict(trigger_level=-1, open_elevation=0))
    state, manifest = run(domain, force, 2)
    gate = manifest['gate_control']['gates'][0]
    assert np.all(state.depth >= 0)
    assert abs(manifest['gate_control']['pond_budget']['residual_m3']) < 4e-5
    if gate_depth > pond_depth:
        assert gate['pond_in_m3'] > 0
    else:
        assert gate['pond_out_m3'] > 0


@pytest.mark.parametrize('pond_depth', [0,2])
def test_hanging_subedges_have_one_shared_flux_and_new_bed_pressure(tmp_path, pond_depth, monkeypatch):
    import taichi as ti
    from src.nh_flood_2d.core.pressure import PressureReconstruction
    original = PressureReconstruction.update_cell
    observed_bed = ti.field(ti.f32, ())
    @ti.func
    def capture(reconstruction, slot):
        ti.atomic_min(observed_bed[None], reconstruction.bed[3])
        original(reconstruction, slot)
    monkeypatch.setattr(PressureReconstruction, 'update_cell', capture)
    cells=[(100,110,0,10),(0,10,0,10),(10,15,0,5),(10,15,5,10),(15,25,0,10)]
    domain, force, topo, depth, bed = setup(tmp_path, cells=cells, types=[1,10,11,11,1],
        depth=[2,pond_depth,2,2,2], gate=dict(gates=[dict(name='A', inside_point=(12,2), control_point=(105,5))]))
    state, manifest = run(domain, force, 2)
    shared=(topo[:,1]==2)&np.isin(topo[:,2],[3,4])
    assert np.count_nonzero(shared)==2
    assert np.all(state.q[shared] != 0)
    gate=manifest['gate_control']['gates'][0]
    pond=manifest['gate_control']['pond_budget']
    assert pond['storage_m3']-pond['initial_storage_m3'] == pytest.approx(
        gate['pond_net_m3']+pond['rounding_m3'], abs=2e-5)
    assert abs(pond['residual_m3']) < 2e-5
    assert observed_bed[None] == np.float32(-.1)


def event_steps(path, force, step, end):
    count = round(end/step)
    with rasterio.open(path/'rain.tif', 'w', driver='GTiff', width=1,height=1,
        count=count+1,dtype='float32',crs='EPSG:2326',
        transform=from_origin(0,10,30,10)) as dst:
        dst.write(np.zeros((count+1,1,1),'float32'))
    times=[i*step for i in range(count+1)]+[3600]
    (path/'intervals.csv').write_text('interval_start,interval_end\n'+''.join(
        f'{(START+timedelta(seconds=a)).isoformat()},'
        f'{(START+timedelta(seconds=b)).isoformat()}\n' for a,b in zip(times,times[1:])))


def test_local_movement_response_converges_with_step_refinement(tmp_path):
    results=[]
    for i,step in enumerate((.125,.0625,.03125)):
        path=tmp_path/str(i)
        domain,force,*_=setup(path, gate=dict(opening_seconds=.5))
        event_steps(path,force,step,1)
        state,manifest=run(domain,force,1)
        results.append(state.depth)
        assert np.sum(state.depth)*100 == pytest.approx(600,abs=8e-5)
        assert abs(manifest['gate_control']['pond_budget']['residual_m3']) < 2e-5
    assert np.linalg.norm(results[2]-results[1]) < np.linalg.norm(results[1]-results[0])


def test_multicell_dry_channel_wets_and_can_reverse(tmp_path):
    cells=[(i*10,(i+1)*10,0,10) for i in range(5)]
    boundaries=level_boundary(tmp_path,[(0,4),(30,4),(31,0),(3600,0)])
    domain,force,*_=setup(tmp_path,cells=cells,types=[1,11,11,11,10],
        bed=[0,4,4,4,0],depth=[4,0,0,0,0],boundaries=boundaries,
        gate=dict(open_elevation=0,opening_seconds=1))
    state,manifest=run(domain,force,90)
    gate=manifest['gate_control']['gates'][0]
    assert gate['pond_in_m3'] > 0 and gate['pond_out_m3'] > 0
    assert gate['river_in_m3'] > 0 and gate['river_out_m3'] > 0
    assert gate['first_effective_flow_seconds'] > gate['command_seconds']
    assert np.all(state.depth >= 0)
    assert abs(manifest['gate_control']['pond_budget']['residual_m3']) < .002


def test_pond_rain_infiltration_overtopping_and_roundoff_close(tmp_path):
    cells=[(i*10,(i+1)*10,0,10) for i in range(4)]
    domain,force,*_=setup(tmp_path,cells=cells,types=[1,11,10,1],depth=[2,2,1,.5],
        rain=100,gate=dict(open_elevation=0))
    data=domain.model_dump()
    data['land_parameters'][10]['infiltration']=dict(f0_mm_h=36,fc_mm_h=18,k_per_h=2)
    domain=DomainConfig.model_validate(data)
    state,manifest=run(domain,force,4)
    pond=manifest['gate_control']['pond_budget']
    assert pond['rain_m3'] > 0 and pond['infiltration_m3'] > 0
    assert pond['overtopping_out_m3'] > 0
    assert abs(pond['residual_m3']) < 2e-5


def test_limiter_records_actual_shared_flux_not_requested_flux(tmp_path):
    domain,force,*_=setup(tmp_path,depth=[.021,.021,0],
        gate=dict(trigger_level=-1,open_elevation=0))
    # Supply a large outgoing momentum to force donor limiting on the first step.
    seed=read_checkpoint(domain.restart_state)
    from contextlib import closing
    import fastdb4py as fdb
    from src.nh_flood_2d.schema.feature import SideTopoInfo
    preprocess(domain,force)
    with closing(fdb.RecordEngine.load(domain.ns_fdb,from_file=True)) as db:
        topo=db.table(SideTopoInfo).column.info.copy().reshape(-1,3)
    seed.q[:]=0
    side=np.flatnonzero((topo[:,1]==2)&(topo[:,2]==3))[0]
    seed.q[side]=100
    seed.q[np.flatnonzero((topo[:,1]==1)&(topo[:,2]==2))[0]]=-100
    domain=domain.model_copy(update={'afa':1.})
    write_checkpoint(domain.restart_state,**vars(seed))
    state,manifest=run(domain,force,1)
    gate=manifest['gate_control']['gates'][0]
    assert manifest['limiter_activity']['limited_face_steps'] > 0
    assert gate['pond_in_m3'] <= .021*100+1e-6
    assert state.depth[3]*100 == pytest.approx(gate['pond_net_m3'],abs=1e-6)
    assert abs(manifest['gate_control']['pond_budget']['residual_m3']) < 1e-6


def test_fractional_opening_replays_online_bed_bitwise(tmp_path):
    domain,force,*_=setup(tmp_path,bed=[0,10,0],depth=[2,0,0],
        gate=dict(opening_seconds=7,open_elevation=1.3))
    state,_=run(domain,force,1.25)
    assert read_bed_at(tmp_path/'out/output',1.25).tobytes() == state.bed.tobytes()


def test_pond_level_reservoir_accounts_for_overwritten_negative_depth(tmp_path):
    boundaries=level_boundary(tmp_path,[(0,.01),(3600,.01)])
    line=Path(boundaries[0]['line'])
    geometry=json.loads(line.read_text())
    geometry['coordinates']=[[30,0],[30,10]]
    line.write_text(json.dumps(geometry))
    domain,force,topo,*_=setup(tmp_path,boundaries=boundaries,depth=[2,2,.01])
    seed=read_checkpoint(domain.restart_state)
    side=np.flatnonzero((topo[:,1]==2)&(topo[:,2]==3))[0]
    seed.q[side]=-100
    write_checkpoint(domain.restart_state,**vars(seed))
    state,manifest=run(domain,force,.125)
    pond=manifest['gate_control']['pond_budget']
    assert state.depth[3] == np.float32(.01)
    assert manifest['gate_control']['gates'][0]['pond_out_m3'] > 1
    assert pond['boundary_m3'] > 1
    assert pond['positivity_m3'] == 0
    assert abs(pond['residual_m3']) < 3e-5


def test_cold_warmup_initialization_preserves_closed_gate_barrier(tmp_path):
    boundaries=level_boundary(tmp_path,[(0,2),(3600,2)])
    domain,force,*_=setup(tmp_path,boundaries=boundaries,warmup=2,
        bed=[0,10,0],depth=[0,0,0])
    domain=domain.model_copy(update={'restart_state':''})
    state,manifest=run(domain,force,0)
    assert manifest['initialization']['method'] == 'connected_still_water'
    assert manifest['gate_control']['gates'][0]['command_seconds'] is None
    assert state.bed[2] == 10
    assert state.depth[1] == 2 and np.all(state.depth[2:] == 0)


@pytest.mark.parametrize('pond_type', [10,7])
def test_configurable_pond_and_channel_source_budgets_close(tmp_path, pond_type):
    boundaries=level_boundary(tmp_path,[(0,.01),(3600,.01)])
    line=Path(boundaries[0]['line'])
    geometry=json.loads(line.read_text())
    geometry['coordinates']=[[10,10],[20,10]]
    line.write_text(json.dumps(geometry))
    boundaries[0].update(name='gate_supply',kind='discharge')
    domain,force,*_=setup(tmp_path,boundaries=boundaries,types=[1,11,pond_type],
        depth=[2,2,1],rain=100,gate=dict(pond_land_type=pond_type,open_elevation=0))
    data=domain.model_dump()
    data['land_parameters'][11]['infiltration']=dict(f0_mm_h=36,fc_mm_h=18,k_per_h=2)
    domain=DomainConfig.model_validate(data)
    state,manifest=run(domain,force,4)
    control=manifest['gate_control']
    channel=control['gates'][0]['channel_budget']
    assert control['pond_land_type'] == pond_type
    assert channel['rain_m3'] > 0 and channel['infiltration_m3'] > 0
    assert channel['boundary_m3'] == pytest.approx(.04,abs=3e-8)
    assert channel['initial_storage_m3'] == 200
    assert channel['storage_m3'] == float(state.depth[2])*100
    assert abs(channel['residual_m3']) < 3e-5
    assert abs(control['pond_budget']['residual_m3']) < 3e-5
    import csv
    rows=list(csv.DictReader((tmp_path/'out/gate_balance.csv').open()))
    final=next(row for row in reversed(rows) if row['region']=='A')
    assert float(final['residual_m3']) == channel['residual_m3']
    assert float(final['river_net_m3']) == channel['river_net_m3']
