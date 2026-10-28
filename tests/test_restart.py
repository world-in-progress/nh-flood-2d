"""A saved physical state resumes the same rain, Horton, and flow evolution."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.core.output import read_checkpoint
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess


pytestmark = pytest.mark.usefixtures('taichi_cpu')
START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


def _inputs(path):
    path.mkdir()
    # Two connected 10 x 10 m cells; side 2 is shared.
    (path / 'ne.txt').write_text(
        '1,1,1,1,1,1,2,3,4,5,5,0,10\n'
        '2,1,1,1,1,2,6,7,8,15,5,0,10\n')
    (path / 'ns.txt').write_text(
        '1,2,0,1,0,0,10,0,5,0,0\n'
        '2,2,1,2,0,0,10,10,5,0,0\n'
        '3,1,0,0,0,1,10,5,0,0,0\n'
        '4,1,0,0,1,0,10,5,10,0,0\n'
        '6,2,2,0,0,0,10,20,5,0,0\n'
        '7,1,0,0,0,2,10,15,0,0,0\n'
        '8,1,0,0,2,0,10,15,10,0,0\n')
    with rasterio.open(path / 'rain.tif', 'w', driver='GTiff', width=2, height=1,
                       count=2, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 10, 10, 10)) as dst:
        dst.write(np.array([[100, 0]], dtype='float32'), 1)
        dst.write(np.array([[0, 100]], dtype='float32'), 2)
    (path / 'intervals.csv').write_text(
        'interval_start,interval_end\n'
        f'{START.isoformat()},{(START + timedelta(seconds=5)).isoformat()}\n'
        f'{(START + timedelta(seconds=5)).isoformat()},'
        f'{(START + timedelta(seconds=10)).isoformat()}\n')


def _run(inputs, output, duration, restart='', boundaries=None):
    domain = DomainConfig(
        ne=str(inputs / 'ne.txt'), ns=str(inputs / 'ns.txt'),
        epsg_code=2326, domain_dir=str(output), start=START,
        duration=duration, warmup_duration=0, yield_step=1,
        land_parameters={10: dict(manning_n=.03, infiltration=dict(
            f0_mm_h=36, fc_mm_h=18, k_per_h=2))},
        restart_state=str(restart) if restart else '')
    forcing = ForceConfig(boundaries=boundaries or [], rainfall=dict(
        raster=str(inputs / 'rain.tif'), intervals=str(inputs / 'intervals.csv')))
    preprocess(domain, forcing)
    budget = solver(domain, forcing)
    state = read_checkpoint(Path(domain.output_dir) / 'state_final.fdb')
    manifest = json.loads((output / 'manifest.json').read_text())
    return budget, state, manifest


def test_restart_after_requested_end_rejects_before_results(tmp_path):
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    _run(inputs, tmp_path / 'first', 10)
    path = tmp_path / 'first/output/state_final.fdb'
    target = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='Checkpoint time.*simulation end'):
        _run(inputs, target, 5, restart=path)
    assert not (target / 'output').exists()
    assert not (target / 'mass_balance.csv').exists()
    assert not (target / 'step_statistics.csv').exists()


def test_restart_at_requested_end_completes_without_steps(tmp_path):
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    expected, first, _ = _run(inputs, tmp_path / 'first', 10)
    actual, final, manifest = _run(inputs, tmp_path / 'same', 10,
                                   restart=tmp_path / 'first/output/state_final.fdb')
    assert manifest['steps'] == 0
    assert manifest['completion_reason'] == 'checkpoint_at_end'
    assert actual == expected
    np.testing.assert_array_equal(final.depth, first.depth)


def test_restart_first_cfl_and_controlling_edge_share_current_state(tmp_path):
    import csv
    from test_positivity import _grid, _vertical
    from src.nh_flood_2d.core.output import write_checkpoint
    from src.nh_flood_2d.core.run_record import sha256
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    _grid(inputs)
    with rasterio.open(inputs / 'rain.tif', 'w', driver='GTiff', width=1, height=1,
                       count=2, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 30, 30, 30)) as rain:
        rain.write(np.zeros((2, 1, 1), dtype='float32'))
    line = inputs / 'sea.geojson'
    line.write_text(json.dumps({'type': 'LineString', 'coordinates': [[30, 20], [30, 30]],
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:2326'}}}))
    series = inputs / 'sea.csv'
    series.write_text('time,value\n' + ''.join(
        f'{(START + timedelta(seconds=t)).isoformat()},{v}\n' for t, v in [(0, .1), (1, 1), (10, 1)]))
    depth = np.full(10, .1, dtype='float32')
    depth[0] = 0
    zeros = np.zeros(10, dtype='float32')
    words = np.zeros(10, dtype='uint32')
    q = np.zeros(25, dtype='float32')
    q[_vertical(0, 1)] = .2
    seed = inputs / 'seed.fdb'
    write_checkpoint(seed, depth=depth, bed=zeros, q=q, rain_time=zeros,
        infiltration=zeros, max_depth=depth, max_speed=zeros,
        first_time_hi=words, first_time_lo=words, clock_hi=0, clock_lo=0,
        gate_open=False, metadata_json=json.dumps(dict(run_state=False,
            gate_config='null', ne_sha256=sha256(inputs / 'ne.txt'), ns_sha256=sha256(inputs / 'ns.txt'))))
    cfg = DomainConfig(ne=str(inputs / 'ne.txt'), ns=str(inputs / 'ns.txt'),
        epsg_code=2326, domain_dir=str(tmp_path / 'first'), start=START,
        duration=1, warmup_duration=0, yield_step=1, restart_state=str(seed),
        land_parameters={10: dict(manning_n=.03, infiltration='none')})
    force = ForceConfig(boundaries=[dict(name='sea', kind='level', line=str(line), series=str(series))],
        rainfall=dict(raster=str(inputs / 'rain.tif'), intervals=str(inputs / 'intervals.csv')))
    preprocess(cfg, force)
    solver(cfg, force)
    checkpoint = Path(cfg.output_dir) / 'state_final.fdb'
    state = read_checkpoint(checkpoint)
    resumed = cfg.model_copy(update={'duration': 2, 'domain_dir': str(tmp_path / 'resumed'),
                                      'restart_state': str(checkpoint)})
    preprocess(resumed, force)
    solver(resumed, force)
    with (tmp_path / 'resumed/step_statistics.csv').open() as stream:
        row = list(csv.DictReader(stream))[-1]
    assert row['steps'] == '1'  # Both candidates exceed the 1 s cap.
    side = _vertical(2, 2)  # New high level in cell 9, not the old moving edge 2.
    assert int(row['limiting_edge']) == side
    assert (int(row['lower_cell']), int(row['higher_cell'])) == (8, 9)
    face_depth = max(float(state.depth[8]), float(state.depth[9]), .01)
    velocity = float(state.q[side]) / face_depth
    candidate = .5 * 10 / (np.sqrt(9.81 * face_depth) + abs(velocity))
    assert float(row['cfl_min_s']) == pytest.approx(candidate, rel=2e-7)
    assert float(row['face_depth_m']) == face_depth
    assert float(row['face_velocity_m_s']) == velocity
    assert float(row['lower_depth_m']) == float(state.depth[8])
    assert float(row['higher_depth_m']) == float(state.depth[9])


@pytest.mark.parametrize('change', ['order', 'name', 'kind', 'edges'])
def test_restart_rejects_changed_boundary_identity(tmp_path, change):
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    boundaries = []
    for name, kind, x, value in [('A', 'level', 0, .2), ('B', 'discharge', 20, .1)]:
        line = inputs / f'{name}.geojson'
        line.write_text(json.dumps({'type': 'LineString', 'coordinates': [[x, 0], [x, 10]],
            'crs': {'type': 'name', 'properties': {'name': 'EPSG:2326'}}}))
        series = inputs / f'{name}.csv'
        series.write_text('time,value\n' + ''.join(
            f'{(START + timedelta(seconds=t)).isoformat()},{value}\n' for t in [0, 10]))
        boundaries.append(dict(name=name, kind=kind, line=str(line), series=str(series)))
    _run(inputs, tmp_path / 'first', 5, boundaries=boundaries)
    if change == 'order':
        boundaries.reverse()
    elif change in ('name', 'kind'):
        boundaries[0][change] = 'renamed' if change == 'name' else 'discharge'
    else:
        p = inputs / 'A.geojson'
        data = json.loads(p.read_text())
        data['coordinates'] = [[0, 0], [10, 0]]
        p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Checkpoint boundary identity'):
        _run(inputs, tmp_path / 'changed', 10,
             restart=tmp_path / 'first/output/state_final.fdb', boundaries=boundaries)


def test_restart_matches_uninterrupted_rain_and_horton_evolution(tmp_path, monkeypatch):
    from src.nh_flood_2d.core.clock import StepClock
    traces = []
    original = StepClock.read_trace

    def capture(clock):
        rows = original(clock)
        traces.extend(rows[rows[:, 3] == clock.COMPLETED].copy())
        return rows

    monkeypatch.setattr(StepClock, 'read_trace', capture)
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    full_budget, full, full_manifest = _run(inputs, tmp_path / 'full', 10)
    full_trace = np.array(traces, dtype=np.uint32)
    traces.clear()
    first_budget, split, first_manifest = _run(inputs, tmp_path / 'first', 5)
    split_path = tmp_path / 'first' / 'output' / 'state_final.fdb'
    traces.clear()
    resumed_budget, resumed, resumed_manifest = _run(
        inputs, tmp_path / 'resumed', 10, restart=split_path)
    resumed_trace = np.array(traces, dtype=np.uint32)
    after_five = ((full_trace[:, 1].astype(np.uint64) << 32)
                  | full_trace[:, 2].astype(np.uint64)) > 5 * 2**40
    np.testing.assert_array_equal(full_trace[after_five], resumed_trace)
    assert not hasattr(split, 'next_dt')
    print('RESTART_CFL_BITWISE: derived first candidate and complete trace match uninterrupted evolution')

    assert first_budget['rainfall_m3'] > 0
    assert first_budget['infiltration_m3'] > 0
    assert np.any(split.rain_time[1:] > 0)
    assert np.any(split.q[1:] != 0)
    for name in ('depth', 'bed', 'q', 'rain_time', 'infiltration',
                 'max_depth', 'max_speed', 'first_time_hi', 'first_time_lo'):
        np.testing.assert_array_equal(getattr(full, name), getattr(resumed, name),
                                      err_msg=f'{name} differs after restart')
    assert (full.clock_hi, full.clock_lo, full.gate_open) == (
        resumed.clock_hi, resumed.clock_lo, resumed.gate_open)
    assert full_budget == resumed_budget
    assert full_budget['rainfall_m3'] > first_budget['rainfall_m3']
    assert full_budget['infiltration_m3'] > first_budget['infiltration_m3']
    assert full_manifest['steps'] == first_manifest['steps'] + resumed_manifest['steps']
    assert json.loads(resumed.metadata_json)['total_steps'] == full_manifest['steps']
    assert json.loads(split.metadata_json)['elapsed_seconds'] == 5


def test_formal_restart_discards_warmup_extrema(tmp_path):
    from src.nh_flood_2d.core.output import write_checkpoint
    from src.nh_flood_2d.core.run_record import sha256
    from src.nh_flood_2d.core.solver_compact import warmup_solver
    inputs=tmp_path/'inputs';_inputs(inputs)
    with rasterio.open(inputs/'rain.tif','r+') as dst:
        dst.write(np.zeros((2,1,2),dtype='float32'))
    depth=np.array([0,.5,0],dtype='float32');zeros=np.zeros_like(depth)
    words=np.zeros(3,dtype='uint32')
    seed=tmp_path/'seed.fdb'
    write_checkpoint(seed,depth=depth,bed=zeros,q=np.zeros(8,dtype='float32'),
        rain_time=zeros,infiltration=np.array([0,.1,.1],dtype='float32'),
        clock_hi=0,clock_lo=0,gate_open=False,
        max_depth=depth,first_time_hi=words,first_time_lo=words,max_speed=zeros,
        metadata_json=json.dumps(dict(run_state=False,ne_sha256=sha256(inputs/'ne.txt'),
            ns_sha256=sha256(inputs/'ns.txt'),gate_config='null')))
    domain=DomainConfig(ne=str(inputs/'ne.txt'),ns=str(inputs/'ns.txt'),epsg_code=2326,
        domain_dir=str(tmp_path/'warm'),start=START,duration=2,warmup_duration=2,
        yield_step=1,restart_state=str(seed),land_parameters={10:dict(manning_n=.03,
            infiltration=dict(f0_mm_h=360000,fc_mm_h=360000,k_per_h=0))})
    forcing=ForceConfig(boundaries=[],rainfall=dict(raster=str(inputs/'rain.tif'),
        intervals=str(inputs/'intervals.csv')))
    preprocess(domain,forcing);warmup_solver(domain,forcing)
    path=Path(domain.output_dir)/'state_final.fdb';warm=read_checkpoint(path)
    assert warm.max_depth[1]==.5 and warm.depth[1]<.5
    resumed=domain.model_copy(update={'domain_dir':str(tmp_path/'formal'),
        'warmup_duration':0,'restart_state':str(path)})
    preprocess(resumed,forcing);solver(resumed,forcing)
    final=read_checkpoint(Path(resumed.output_dir)/'state_final.fdb')
    assert final.max_depth[1]==warm.depth[1]
    first_tick=int(final.first_time_hi[1])*2**32+int(final.first_time_lo[1])
    assert first_tick==2*2**40  # formal t=0, with the original warmup clock origin
