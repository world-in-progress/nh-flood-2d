"""Exterior face carry and full-solver boundary behaviour."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
import taichi as ti
from rasterio.transform import from_origin

from src.nh_flood_2d.core.run_record import sha256
from src.nh_flood_2d.core.output import (list_depth_frames, read_bed_at,
                                         read_depth_frame, write_checkpoint)
from src.nh_flood_2d.core.solver_compact import run_simulation
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess.hydraulics import open_boundary_face_fractions
from src.nh_flood_2d.core.solver_compact import boundary_neighbor_sum



pytestmark = pytest.mark.usefixtures('taichi_cpu')

START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)
PERIOD = 86_400


@pytest.mark.parametrize("selected_id", [-1, 0, 1])
def test_open_face_fraction_uses_exterior_length_only_for_selected_cells(selected_id):
    # The same exterior geometry represents a closed cell, discharge boundary
    # (first boundary ID), or level boundary (second boundary ID).
    boundary_ids = np.array([-1, selected_id, -1], dtype=np.int32)
    side_lengths = np.array([0, 2, 3, 1, 3, 2, 2, 4], dtype=np.float32)
    side_topology = np.array([
        [0, 0, 0], [2, 0, 1], [2, 1, 2], [1, 0, 1],
        [1, 1, 2], [1, 1, 0], [1, 1, 2], [2, 2, 0],
    ], dtype=np.int32)
    faces = [
        [[], [], [], []],
        [[1], [2], [3, 4], [5, 6]],
        [[2], [7], [4], [7]],
    ]
    pointers = np.zeros((3, 4), dtype=np.int32)
    counts = np.zeros((3, 4), dtype=np.int32)
    flat = []
    for cell, cell_faces in enumerate(faces):
        for direction, ids in enumerate(cell_faces):
            pointers[cell, direction] = len(flat)
            counts[cell, direction] = len(ids)
            flat.extend(ids)

    fractions = open_boundary_face_fractions(
        np.array([False, selected_id >= 0, False, selected_id >= 0, False, selected_id >= 0, False, False]), side_lengths, side_topology, pointers, counts,
        np.array(flat, dtype=np.int32))
    assert fractions.shape == (3, 4)
    assert fractions.dtype == np.float32
    expected = [1., 0., .25, .5] if selected_id >= 0 else [0.] * 4
    np.testing.assert_allclose(fractions[1], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(fractions[[0, 2]], np.zeros((2, 4), dtype=np.float32))


def test_boundary_stencil_continues_uniform_and_linear_discharge():
    cases=np.array([[1.3,0.,.8,1.,0.],[-2.5,-1.,0.,0.,1.],
                    [1.3,.75*1.3,.5*1.3,.25,.5],[.7,0.,0.,1.,1.]],dtype='float32')
    data=ti.field(ti.f32,cases.shape);result=ti.field(ti.f32,len(cases));data.from_numpy(cases)
    @ti.kernel
    def evaluate():
        for i in result:
            result[i]=boundary_neighbor_sum(data[i,0],data[i,1],data[i,2],data[i,3],data[i,4])
    evaluate()
    np.testing.assert_allclose(result.to_numpy(),2*cases[:,0],rtol=2e-7,atol=2e-7)


def test_closed_stencil_and_theta_one_update_keep_original_bits():
    rng=np.random.default_rng(9041)
    values=rng.normal(size=(64,3)).astype('float32')
    data=ti.field(ti.f32,values.shape);out=ti.field(ti.f32,(len(values),4));data.from_numpy(values)
    theta=1.
    @ti.kernel
    def evaluate():
        for i in range(64):
            q,left,right=data[i,0],data[i,1],data[i,2]
            out[i,0]=left+right
            out[i,1]=boundary_neighbor_sum(q,left,right,0.,0.)
            out[i,2]=(theta*q+(1-theta)/2*(left+right)+.011)/1.003
            neighbours=left+right
            if ti.static(theta!=1.):
                neighbours=boundary_neighbor_sum(q,left,right,1.,0.)
            out[i,3]=(theta*q+(1-theta)/2*neighbours+.011)/1.003
    evaluate();bits=out.to_numpy().view('uint32')
    np.testing.assert_array_equal(bits[:,0],bits[:,1])
    np.testing.assert_array_equal(bits[:,2],bits[:,3])


def _write_grid(path: Path, nx: int, ny: int, dx: int) -> None:
    """Write shared sides for a flat, rectangular mesh of square cells."""
    sides = {}
    cells = []

    def side(orientation, x, y, lower, higher):
        key = orientation, x, y
        if key not in sides:
            sides[key] = [len(sides) + 1, orientation, 0, 0, 0, 0,
                          dx, x, y, 0, 0]
        row = sides[key]
        if orientation == 2:
            if lower:
                row[2] = lower
            if higher:
                row[3] = higher
        else:
            if lower:
                row[4] = lower
            if higher:
                row[5] = higher
        return row[0]

    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i + 1
            x, y = (i + .5) * dx, (j + .5) * dx
            ids = [side(2, i * dx, y, 0, cell),
                   side(2, (i + 1) * dx, y, cell, 0),
                   side(1, x, j * dx, 0, cell),
                   side(1, x, (j + 1) * dx, cell, 0)]
            cells.append([cell, 1, 1, 1, 1, *ids, x, y, 0, 10])
    (path / 'ne.txt').write_text(''.join(','.join(map(str, row)) + '\n' for row in cells))
    (path / 'ns.txt').write_text(''.join(','.join(map(str, row)) + '\n'
                                       for row in sides.values()))


def _run_case(path: Path, kind: str, theta: float, afa: float):
    path.mkdir()
    nx, ny = 40, 2
    dx = 10 if kind == 'channel' else 100
    duration = 6_000 if kind == 'channel' else 2 * PERIOD
    depth = 1. if kind == 'channel' else 3.
    _write_grid(path, nx, ny, dx)

    with rasterio.open(path / 'rain.tif', 'w', driver='GTiff', width=1,
                       height=1, count=1, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, ny * dx, nx * dx, ny * dx)) as raster:
        raster.write(np.zeros((1, 1), dtype=np.float32), 1)
    (path / 'intervals.csv').write_text(
        'interval_start,interval_end\n'
        f'{START.isoformat()},{(START + timedelta(seconds=duration)).isoformat()}\n')

    definitions = ([('upstream', 'discharge', 0, 10.),
                    ('downstream', 'level', nx * dx, 1.)]
                   if kind == 'channel' else [('sea', 'level', 0, None)])
    boundaries = []
    for name, boundary_kind, x, constant in definitions:
        line = path / f'{name}.geojson'
        series = path / f'{name}.csv'
        line.write_text(json.dumps({
            'type': 'LineString', 'coordinates': [[x, .1], [x, ny * dx - .1]],
            'crs': {'type': 'name', 'properties': {'name': 'EPSG:2326'}}}))
        times = [0, duration] if constant is not None else range(0, duration + 300, 300)
        with series.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time', 'value'])
            for seconds in times:
                value = constant if constant is not None else 3 + np.sin(2 * np.pi * seconds / PERIOD)
                writer.writerow([(START + timedelta(seconds=seconds)).isoformat(), value])
        boundaries.append(dict(name=name, kind=boundary_kind,
                               line=str(line), series=str(series)))

    initial_depth = np.r_[0., np.full(nx * ny, depth)].astype(np.float32)
    zeros = np.zeros_like(initial_depth)
    write_checkpoint(path / 'initial.fdb', depth=initial_depth, bed=zeros,
        q=np.zeros(len((path/'ns.txt').read_text().splitlines())+1,dtype='float32'),
        rain_time=zeros, infiltration=zeros, max_depth=initial_depth,
        first_time_hi=np.zeros(len(initial_depth),dtype='uint32'),
        first_time_lo=np.zeros(len(initial_depth),dtype='uint32'), max_speed=zeros,
        clock_hi=0, clock_lo=0, gate_open=False,
        metadata_json=json.dumps({'run_state': False,
            'ne_sha256': sha256(path / 'ne.txt'), 'ns_sha256': sha256(path / 'ns.txt'),
            'gate_config': 'null'}))
    domain = DomainConfig(
        ne=str(path / 'ne.txt'), ns=str(path / 'ns.txt'),
        epsg_code=2326, domain_dir=str(path / 'out'), start=START,
        duration=duration, warmup_duration=0,
        land_parameters={10: dict(manning_n=.03, infiltration='none')},
        sita=theta, afa=afa, yield_step=500 if kind == 'channel' else 600,
        restart_state=str(path / 'initial.fdb'))
    force = ForceConfig(boundaries=boundaries,
                        rainfall={'raster': str(path / 'rain.tif'),
                                  'intervals': str(path / 'intervals.csv')})
    budget = run_simulation(domain, force)
    frames = list_depth_frames(domain.output_dir)
    assert frames
    return budget, frames, nx, ny, dx


def test_open_channel_steady_profile_is_stable_across_theta_and_time_step(tmp_path):
    profiles = []
    for theta in (1., .7):
        for afa in (.4, .1):
            path = tmp_path / f'channel_theta{theta}_afa{afa}'
            budget, frames, nx, ny, dx = _run_case(path, 'channel', theta, afa)
            tail = [(read_bed_at(path/'out/output', seconds) + read_depth_frame(frame))[1:]
                    for seconds, frame in frames[-3:]]
            eta = [surface.reshape(ny, nx) for surface in tail]
            profile = eta[-1].mean(axis=0)
            profiles.append(profile)
            assert abs(profile[0] - profile[1]) < .005
            assert abs(profile[-2] - profile[-1]) < .005
            assert np.max(np.abs(eta[-1] - eta[-2])) < 2e-4
            with (path / 'out' / 'mass_balance.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            outflow = [-(float(b['level:downstream_m3']) - float(a['level:downstream_m3'])) /
                       (float(b['elapsed_seconds']) - float(a['elapsed_seconds']))
                       for a, b in zip(rows[-3:-1], rows[-2:])]
            assert all(abs(q - 10.) < .01 for q in outflow), (theta, afa, outflow)
            assert abs(budget['residual_m3']) / (
                budget['initial_storage_m3'] + budget['discharge:upstream_m3']) < 1e-4
    assert np.ptp(np.stack(profiles), axis=0).max() < 2e-4


def test_tidal_basin_keeps_amplitude_and_phase_across_time_steps(tmp_path):
    metrics = []
    omega = 2 * np.pi / PERIOD
    for afa in (.04, .015):
        _, frames, nx, _, _ = _run_case(tmp_path / f'tide_afa{afa}', 'tide', .7, afa)
        times, levels = [], []
        for seconds, frame in frames:
            times.append(seconds)
            levels.append(float(read_depth_frame(frame)[nx]))
        t = np.asarray(times)
        y = np.asarray(levels)
        last_cycle = t >= PERIOD
        t, y = t[last_cycle], y[last_cycle]
        basis = np.column_stack([np.sin(omega * t), np.cos(omega * t),
                                 np.ones(len(t)), (t - t.mean()) / PERIOD])
        sine, cosine, _, _ = np.linalg.lstsq(basis, y, rcond=None)[0]
        amplitude = float(np.hypot(sine, cosine))
        phase_minutes = float(np.arctan2(-cosine, sine) / omega / 60)
        metrics.append((amplitude, phase_minutes))
        assert abs(amplitude - 1.) < .01
        assert abs(phase_minutes) < 2.
    assert abs(metrics[0][0] - metrics[1][0]) < .002
    assert abs(metrics[0][1] - metrics[1][1]) < .1
