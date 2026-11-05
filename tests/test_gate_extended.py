"""CPU integration cases for large gate components, paired faces and clock age."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from src.nh_flood_2d.core.clock import StepClock, _seconds_to_ticks, _words
from src.nh_flood_2d.core.gate import GateController
from src.nh_flood_2d.input.force import GateConfig
from src.nh_flood_2d.preprocess.gate import GateLayout
from test_gate_solver import event_steps, run, setup

pytestmark = pytest.mark.usefixtures('taichi_cpu')


def gate_case(path, cells, types, bed, depth, *, gate_points, opening_seconds=.5,
              trigger_level=1., end=1., step=None, yield_step=1):
    """Build a seeded production-solver case with explicit physical cells."""
    gates = [dict(name=name, inside_point=inside, control_point=control)
             for name, inside, control in gate_points]
    domain, force, topo, initial, initial_bed = setup(
        path, cells=cells, types=types, bed=bed, depth=depth,
        gate=dict(gates=gates, opening_seconds=opening_seconds,
                  trigger_level=trigger_level, open_elevation=0.,
                  pond_land_type=10), yield_step=yield_step)
    if step is not None:
        event_steps(path, force, step, end)
    return domain, force, topo, initial, initial_bed


def assert_control_volume(budget, *, scale=1.):
    terms = ('initial_storage_m3', 'storage_m3', 'rain_m3',
             'infiltration_m3', 'positivity_m3', 'pipe_m3',
             'boundary_m3', 'rounding_m3', 'residual_m3')
    assert all(np.isfinite(budget[key]) for key in terms)
    assert abs(budget['residual_m3']) <= max(2e-4, 2e-6 * scale)


def large_components_case(path, *, component_count=7, gate_cells=500):
    """Seven separate 25 by 20 gate channels on square physical cells."""
    assert gate_cells == 500
    cells, types, bed, depth, points = [], [], [], [], []
    for group in range(component_count):
        y = group * 100
        for row in range(22):
            for col in range(25):
                x, bottom = 4 * col, y + 4 * row
                cells.append((x, x + 4, bottom, bottom + 4))
                types.append(1 if row == 0 else 10 if row == 21 else 11)
                bed.append(0. if row in (0, 21) else 1.)
                depth.append(2. if row == 0 else .3 if row == 21 else .2)
        points.append((f'G{group}', (2, y + 6), (2, y + 2)))
    return gate_case(path, cells, types, bed, depth, gate_points=points,
                     opening_seconds=.25, end=.5)


def paired_section_case(path, *, hanging, regime):
    """Represent the same 8 m river-to-pond section with whole or split faces."""
    cells = [(0, 8, 0, 8), (8, 16, 0, 8)]
    types = [1, 11]
    if hanging:
        cells.extend([(x, x+4, y, y+4) for x in (16, 20) for y in (0, 4)])
        types.extend([10] * 4)
    else:
        cells.append((16, 24, 0, 8))
        types.append(10)
    if regime == 'near_balance':
        beds = [0., .1] + [0.] * (len(cells) - 2)
        depths = [1.0001, .9] + [1.] * (len(cells) - 2)
        trigger = .5
    else:
        beds = [0., .1] + [0.] * (len(cells) - 2)
        depths = [2., .05] + [0.] * (len(cells) - 2)
        trigger = .5
    return gate_case(path, cells, types, beds, depths,
                     gate_points=[('A', (12, 2), (4, 2))],
                     opening_seconds=.25, trigger_level=trigger, end=2.)


def dry_channel_case(path, step):
    cells = [(4*i, 4*(i+1), 0, 4) for i in range(6)]
    return gate_case(path, cells, [1, 11, 11, 11, 11, 10],
                     [0., 2., 2., 2., 2., 0.], [4., 0., 0., 0., 0., 0.],
                     gate_points=[('A', (6, 2), (2, 2))],
                     opening_seconds=.5, trigger_level=1., end=16.,
                     step=step, yield_step=1)


def balance_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_seven_large_components_open_and_close_each_control_volume(tmp_path):
    domain, force, topo, initial, _ = large_components_case(tmp_path)
    state, manifest = run(domain, force, .5)
    report = manifest['gate_control']
    gates = report['gates']
    assert len(gates) == 7
    assert sum(len(g['cell_ids']) for g in gates) == 3500
    assert len(topo) // 256 >= 7
    assert [g['command_seconds'] for g in gates] == [0.] * 7
    assert [g['fully_open_seconds'] for g in gates] == [.25] * 7
    assert all(g['pond_in_m3'] > 0 and g['river_in_m3'] > 0 for g in gates)
    assert np.all(state.depth >= 0)
    assert all(state.bed[np.asarray(g['cell_ids'], dtype=int)].max() == 0 for g in gates)
    area = np.r_[0., np.full(7 * 550, 16.)]
    initial_storage = float(np.dot(initial.astype(float), area))
    final_storage = float(np.dot(state.depth.astype(float), area))
    global_budget = manifest['mass_balance_final']
    assert final_storage - initial_storage == pytest.approx(
        global_budget['storage_m3'] - global_budget['initial_storage_m3'], abs=.02)
    assert abs(global_budget['residual_m3']) < .05
    assert_control_volume(report['pond_budget'], scale=initial_storage)
    for gate in gates:
        channel = gate['channel_budget']
        assert_control_volume(channel, scale=channel['initial_storage_m3'])
        expected = (channel['river_net_m3'] - channel['pond_net_m3']
                    + channel['rain_m3'] - channel['infiltration_m3']
                    + channel['positivity_m3'] + channel['pipe_m3']
                    + channel['boundary_m3'] + channel['rounding_m3'])
        assert channel['storage_m3'] - channel['initial_storage_m3'] == pytest.approx(
            expected + channel['residual_m3'], abs=.003)
        assert channel['pond_net_m3'] == pytest.approx(gate['pond_net_m3'], abs=1e-5)


@pytest.mark.parametrize('regime', ['near_balance', 'dry_wet'])
def test_uniform_and_hanging_section_agree_on_width_and_cumulative_flow(tmp_path, regime):
    observations = []
    for hanging in (False, True):
        case = paired_section_case(tmp_path / ('hanging' if hanging else 'uniform'),
                                   hanging=hanging, regime=regime)
        domain, force, topo, initial, _ = case
        state, manifest = run(domain, force, .125)
        gate = manifest['gate_control']['gates'][0]
        pond = manifest['gate_control']['pond_budget']
        assert_control_volume(pond, scale=100.)
        assert_control_volume(gate['channel_budget'], scale=100.)
        assert np.all(state.depth >= 0)
        interface = (topo[:, 1] == 2) & (topo[:, 2] > 2)
        assert np.count_nonzero(interface) == (2 if hanging else 1)
        lengths = np.full(np.count_nonzero(interface), 4. if hanging else 8.)
        assert sum(lengths) == 8.
        width_flow = float(np.dot(state.q[interface], lengths))
        observations.append(dict(width_flow=width_flow, pond_net=gate['pond_net_m3'],
            river_net=gate['channel_budget']['river_net_m3'],
            pond_storage=pond['storage_m3'],
            channel_storage=gate['channel_budget']['storage_m3'],
            center_distance=6. if hanging else 8.))
    uniform, hanging = observations
    if regime == 'dry_wet':
        # In this first step, the positive gate water surface is 0.15 m and
        # the pond is dry. The face depth is 0.05 m. Splitting the pond into
        # 4 m squares moves its center from x=20 to x=18, so the same head
        # jump produces a 4/3 larger gradient across the hanging interface.
        # Both subedges share one physical 8 m face; no flux is duplicated.
        for observed in observations:
            expected = 9.81 * .05 * .15 / observed['center_distance'] * .125 * 8.
            assert observed['width_flow'] == pytest.approx(expected, rel=2e-5)
            assert observed['pond_net'] == pytest.approx(expected * .125, rel=2e-5)
            assert observed['pond_storage'] == pytest.approx(observed['pond_net'], abs=2e-5)
        assert hanging['width_flow'] / uniform['width_flow'] == pytest.approx(4/3, rel=2e-5)
        assert hanging['pond_net'] / uniform['pond_net'] == pytest.approx(4/3, rel=2e-5)
        assert hanging['river_net'] == pytest.approx(uniform['river_net'], rel=2e-5)
    else:
        np.testing.assert_allclose(
            [uniform[key] for key in ('width_flow', 'pond_net', 'river_net',
                                       'pond_storage', 'channel_storage')],
            [hanging[key] for key in ('width_flow', 'pond_net', 'river_net',
                                       'pond_storage', 'channel_storage')],
            rtol=2e-3, atol=2e-4)


def test_dry_multicell_transient_refines_first_water_volume_and_oscillation(tmp_path):
    measurements = []
    for step in (.125, .0625, .03125):
        path = tmp_path / str(step)
        domain, force, *_ = dry_channel_case(path, step)
        state, manifest = run(domain, force, 16.)
        gate = manifest['gate_control']['gates'][0]
        pond = manifest['gate_control']['pond_budget']
        assert_control_volume(pond, scale=100.)
        assert_control_volume(gate['channel_budget'], scale=100.)
        assert np.all(state.depth >= 0)
        assert gate['first_effective_flow_seconds'] is not None
        rows = [row for row in balance_rows(path/'out/gate_balance.csv')
                if row['region'] == 'A']
        cumulative = np.array([float(row['pond_net_m3']) for row in rows])
        increments = np.diff(cumulative)
        oscillation = float(np.sum(np.abs(np.diff(increments))))
        measurements.append(dict(step_s=step,
            first_water_s=gate['first_effective_flow_seconds'],
            pond_in_m3=gate['pond_in_m3'], pond_out_m3=gate['pond_out_m3'],
            oscillation_m3=oscillation))
    (tmp_path/'refinement_measurements.json').write_text(json.dumps(measurements, indent=2))
    coarse, middle, fine = measurements
    assert fine['first_water_s'] <= coarse['first_water_s'] + .5
    for key in ('pond_in_m3', 'pond_out_m3', 'oscillation_m3'):
        assert np.isfinite([row[key] for row in measurements]).all()
        assert abs(fine[key] - middle[key]) <= abs(middle[key] - coarse[key]) + .02
    assert fine['pond_in_m3'] > 0


@pytest.mark.parametrize('opening_seconds', [1800., 3600.])
def test_formal_30000s_command_and_f32_opening_age(tmp_path, opening_seconds):
    """Jump exact device clock ticks; avoid 30,000 seconds of no-op fluid steps."""
    bed = np.array([0., 8., 0., 0.], dtype='float32')
    depth_np = np.zeros(4, dtype='float32')
    area_np = np.array([0., 16., 16., 16.], dtype='float32')
    clock = StepClock(29999.)
    working_bed = ti.field(ti.f32, shape=4)
    working_bed.from_numpy(bed)
    depth = ti.field(ti.f32, shape=4)
    depth.from_numpy(depth_np)
    area = ti.field(ti.f32, shape=4)
    area.from_numpy(area_np)
    config = GateConfig(land_type=11, pond_land_type=10, enabled=True,
                        trigger_level=1., opening_seconds=opening_seconds,
                        open_elevation=0., gates=[dict(name='A',
                        inside_point=(6., 2.), control_point=(2., 2.))])
    layout = GateLayout(np.array([-1, 0, -1, -1], dtype=np.int32),
                        np.array([2], dtype=np.int32),
                        [np.array([1], dtype=np.int32)])
    controller = GateController(config, layout, bed, np.array([0, 11, 1, 10]),
                                clock, 0., 1e-5, 4, 4,
                                working_bed, depth, area)
    controller.initialize_inventory(depth_np, area_np)
    @ti.kernel
    def sample():
        controller.check()
    fractional_age = np.float32(1.234567)
    fractional_bed = np.float32(8. - 8. * np.float32(fractional_age / np.float32(opening_seconds)))
    for seconds, expected in ((29999., None), (30000., 8.),
                              (30001.234567, fractional_bed),
                              (30000. + opening_seconds / 4, 6.),
                              (30000. + opening_seconds / 2, 4.),
                              (30000. + opening_seconds, 0.)):
        if seconds == 30000.:
            depth[2] = 2.
        clock.now.from_numpy(_words(_seconds_to_ticks(seconds)))
        clock.active[None] = 1
        sample()
        assert controller.commanded[0] == int(expected is not None)
        if expected is not None:
            assert working_bed[1] == pytest.approx(expected, abs=2e-6)
    report = controller.report(depth_np, area_np, 30000. + opening_seconds)['gates'][0]
    assert report['command_seconds'] == 30000.
    assert report['scheduled_fully_open_seconds'] == 30000. + opening_seconds
    assert report['fully_open_seconds'] == 30000. + opening_seconds
    assert report['first_fully_open_sample_seconds'] == 30000. + opening_seconds
    assert report['command_clock_ticks'] == _words(_seconds_to_ticks(30000.)).tolist()
