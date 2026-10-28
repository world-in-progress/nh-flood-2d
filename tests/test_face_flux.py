"""Face discharge on nonuniform rectangular meshes, including physical flow."""

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
from src.nh_flood_2d.core.output import (list_depth_frames, read_depth_frame,
                                         write_checkpoint)
from src.nh_flood_2d.core.solver_compact import face_discharge, solver
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.util.ti import init_taichi



pytestmark = pytest.mark.usefixtures('taichi_cpu')

START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


@pytest.mark.parametrize("length", [0.1, 0.7, 1.3, 2.2, 3.14159])
def test_single_side_face_and_theta_update_match_legacy_bits(length):
    """A one-side face takes the side value directly, even for rounded f32 data."""
    init_taichi(use_gpu=False)
    rng = np.random.default_rng(8726)
    q = np.r_[np.float32(0), rng.normal(size=64).astype(np.float32)]
    side_ids = np.arange(1, len(q), dtype=np.int32)
    n_sides = len(side_ids)
    qn = ti.field(ti.f32, shape=len(q))
    side_lengths = ti.field(ti.f32, shape=len(q))
    ids = ti.field(ti.i32, shape=len(side_ids))
    got = ti.field(ti.f32, shape=(len(side_ids), 2))
    qn.from_numpy(q)
    side_lengths.from_numpy(np.full(len(q), length, dtype=np.float32))
    ids.from_numpy(side_ids)

    @ti.kernel
    def evaluate():
        for i in range(n_sides):
            left = face_discharge(qn, side_lengths, ids, i, 1)
            right = face_discharge(qn, side_lengths, ids,
                                   (i + 1) % n_sides, 1)
            got[i, 0] = left
            got[i, 1] = (ti.cast(0.7, ti.f32) * qn[ids[i]]
                         + ti.cast(0.3, ti.f32) / 2 * (left + right)
                         + ti.cast(0.011, ti.f32)) / ti.cast(1.003, ti.f32)

    old = ti.field(ti.f32, shape=(len(side_ids), 2))

    @ti.kernel
    def evaluate_old():
        for i in range(n_sides):
            left = qn[ids[i]]
            right = qn[ids[(i + 1) % n_sides]]
            old[i, 0] = left
            old[i, 1] = (ti.cast(0.7, ti.f32) * qn[ids[i]]
                         + ti.cast(0.3, ti.f32) / 2 * (left + right)
                         + ti.cast(0.011, ti.f32)) / ti.cast(1.003, ti.f32)

    evaluate()
    evaluate_old()
    np.testing.assert_array_equal(got.to_numpy().view(np.uint32),
                                  old.to_numpy().view(np.uint32))


def test_multiside_faces_weight_by_each_side_length_in_all_directions():
    init_taichi(use_gpu=False)
    q = np.array([0, 1.25, -0.5, 2.75, -3, 0.125, 5, -1, 0.75, -2], dtype=np.float32)
    lengths = np.array([1, 0.3, 1.7, 0.2, 0.8, 1.5, 0.4, 2.5, 0.6, 1.3], dtype=np.float32)
    # left, right, bottom, top, and the sentinel face of element zero.
    groups = [[1, 2], [3, 4, 5], [6, 7], [8, 9], []]
    n_faces = len(groups)
    flat = np.array([side for group in groups for side in group], dtype=np.int32)
    starts = np.array([0, 2, 5, 7, 9], dtype=np.int32)
    counts = np.array([len(group) for group in groups], dtype=np.int32)
    qn = ti.field(ti.f32, shape=len(q))
    side_lengths = ti.field(ti.f32, shape=len(q))
    ids = ti.field(ti.i32, shape=len(flat))
    face_start = ti.field(ti.i32, shape=len(groups))
    face_count = ti.field(ti.i32, shape=len(groups))
    out = ti.field(ti.f32, shape=len(groups))
    qn.from_numpy(q)
    side_lengths.from_numpy(lengths)
    ids.from_numpy(flat)
    face_start.from_numpy(starts)
    face_count.from_numpy(counts)

    @ti.kernel
    def evaluate():
        for face in range(n_faces):
            out[face] = face_discharge(qn, side_lengths, ids,
                                       face_start[face], face_count[face])

    evaluate()
    result = out.to_numpy()
    for face, group in enumerate(groups[:4]):
        expected = np.average(q[group].astype(np.float64),
                              weights=lengths[group].astype(np.float64))
        assert result[face] == pytest.approx(expected, abs=2e-6)
    assert result[4] == 0
    for face, group in enumerate(groups[:4]):
        assert result[face] != q[group[0]]


def channel_mesh(path: Path, slope=0.002, length=1024):
    """A long 2 m wide channel: 2 m cells, then 1 m cells across the width."""
    cells = []
    half = length // 2
    for x in range(0, half, 2):
        cells.append((x, x + 2, 0, 2))
    for x in range(half, length):
        for y in (0, 1):
            cells.append((x, x + 1, y, y + 1))

    # Collect face endpoints at each coordinate. A coarse face is split only
    # where its neighboring fine cells require it.
    vertical = {}
    horizontal = {}
    for x0, x1, y0, y1 in cells:
        for x in (x0, x1):
            vertical.setdefault(x, set()).update((y0, y1))
        for y in (y0, y1):
            horizontal.setdefault(y, set()).update((x0, x1))
    side_owner = {}
    cell_faces = []
    for ei, (x0, x1, y0, y1) in enumerate(cells, start=1):
        faces = [[], [], [], []]
        for direction, coordinate, lower, upper, breaks, orient, owner in (
            (0, x0, y0, y1, vertical[x0], 2, 2),
            (1, x1, y0, y1, vertical[x1], 2, 1),
            (2, y0, x0, x1, horizontal[y0], 1, 2),
            (3, y1, x0, x1, horizontal[y1], 1, 1),
        ):
            points = sorted(p for p in breaks if lower <= p <= upper)
            for a, b in zip(points, points[1:]):
                key = (orient, coordinate, a, b)
                side_owner.setdefault(key, {})[owner] = ei
                faces[direction].append(key)
        cell_faces.append(faces)

    side_ids = {key: si for si, key in enumerate(sorted(side_owner), start=1)}
    ne_rows = []
    bed = np.zeros(len(cells) + 1, dtype=np.float32)
    for ei, ((x0, x1, y0, y1), faces) in enumerate(zip(cells, cell_faces), start=1):
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        bed[ei] = 10 - slope * x
        counts = [len(face) for face in faces]
        ids = [side_ids[key] for face in faces for key in face]
        ne_rows.append(','.join(map(str, [ei, *counts, *ids, x, y, bed[ei], 10])))
    ns_rows = []
    for key, si in side_ids.items():
        orient, coordinate, a, b = key
        owners = side_owner[key]
        x, y = (coordinate, (a + b) / 2) if orient == 2 else ((a + b) / 2, coordinate)
        if orient == 2:
            left, right = owners.get(1, 0), owners.get(2, 0)
            bottom = top = 0
        else:
            bottom, top = owners.get(1, 0), owners.get(2, 0)
            left = right = 0
        ns_rows.append(','.join(map(str, [si, orient, left, right, bottom, top,
                                          b - a, x, y, 0, 0])))
    (path / 'ne.txt').write_text('\n'.join(ne_rows) + '\n')
    (path / 'ns.txt').write_text('\n'.join(ns_rows) + '\n')
    return cells, bed


def run_channel(path: Path, theta: float):
    path.mkdir()
    cells, bed = channel_mesh(path)
    duration = 4000
    expected_q = np.sqrt(0.002) / 0.03
    with rasterio.open(path / 'rain.tif', 'w', driver='GTiff', width=1, height=1,
                       count=1, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 2, 1024, 2)) as dst:
        dst.write(np.zeros((1, 1), dtype=np.float32), 1)
    (path / 'times.csv').write_text(
        'interval_start,interval_end\n'
        f'{START.isoformat()},{(START + timedelta(seconds=duration)).isoformat()}\n')
    boundaries = []
    for name, x, value, kind in [
        ('upstream', 0, 2 * expected_q, 'discharge'),
        ('downstream', 1024, float(bed[-1] + 1), 'level'),
    ]:
        (path / f'{name}.geojson').write_text(json.dumps({
            'type': 'LineString', 'coordinates': [[x, 0.1], [x, 1.9]],
            'crs': {'type': 'name', 'properties': {'name': 'EPSG:2326'}}}))
        (path / f'{name}.csv').write_text(
            'time,value\n'
            f'{START.isoformat()},{value}\n'
            f'{(START + timedelta(seconds=duration)).isoformat()},{value}\n')
        boundaries.append({'name': name, 'kind': kind,
                           'line': str(path / f'{name}.geojson'),
                           'series': str(path / f'{name}.csv')})
    depth = np.ones(len(cells) + 1, dtype=np.float32)
    depth[0] = 0
    zeros = np.zeros_like(depth)
    write_checkpoint(path / 'initial.fdb', depth=depth, bed=bed,
        q=np.zeros(len((path/'ns.txt').read_text().splitlines())+1,dtype='float32'),
        rain_time=zeros, infiltration=zeros, max_depth=depth,
        first_time_hi=np.zeros(len(depth),dtype='uint32'),
        first_time_lo=np.zeros(len(depth),dtype='uint32'), max_speed=zeros,
        clock_hi=0, clock_lo=0, gate_open=False,
        metadata_json=json.dumps({'run_state': False,
            'ne_sha256': sha256(path / 'ne.txt'), 'ns_sha256': sha256(path / 'ns.txt'),
            'gate_config': 'null'}))
    sampled = [(i, bounds) for i, bounds in enumerate(cells, start=1)
               if bounds == (510, 512, 0, 2) or bounds[0:2] == (512, 513)]
    stations = {str(i): ((x0+x1)/2, (y0+y1)/2)
                for i, (x0, x1, y0, y1) in sampled}
    dc = DomainConfig(ne=str(path / 'ne.txt'), ns=str(path / 'ns.txt'),
                      epsg_code=2326, domain_dir=str(path / 'out'),
                      start=START, duration=duration, warmup_duration=0,
                      land_parameters={10:dict(manning_n=.03,infiltration='none')},
                      sita=theta, yield_step=300, stations=stations,
                      restart_state=str(path / 'initial.fdb'))
    fc = ForceConfig(boundaries=boundaries,
                     rainfall={'raster': str(path / 'rain.tif'),
                               'intervals': str(path / 'times.csv')})
    preprocess(dc, fc)
    budget = solver(dc, fc)
    frames = list_depth_frames(dc.output_dir)[-4:-1]
    with (Path(dc.output_dir)/'stations.csv').open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    from types import SimpleNamespace
    states = []
    for seconds, frame in frames:
        water = read_depth_frame(frame)
        velocity = np.zeros_like(water)
        for sample in samples:
            if float(sample['elapsed_seconds']) == seconds:
                velocity[int(sample['cell_id'])] = float(sample['u'])
        states.append(SimpleNamespace(depth=water, u=velocity))
    return cells, budget, states


@pytest.mark.parametrize('theta', [0.7, 1.0])
def test_coarse_fine_channel_reaches_continuous_steady_flow(tmp_path, theta):
    cells, budget, frames = run_channel(tmp_path / f'theta_{theta}', theta)
    coarse = next(i for i, (x0, x1, y0, y1) in enumerate(cells, start=1)
                  if (x0, x1, y0, y1) == (510, 512, 0, 2))
    fine = [i for i, (x0, x1, y0, y1) in enumerate(cells, start=1)
            if x0 == 512 and x1 == 513]
    predicted_q = np.sqrt(0.002) / 0.03  # H^(5/3) sqrt(S) / n, H = 1 m
    history = []
    for state in frames:
        q_coarse = float(state.depth[coarse] * state.u[coarse])
        q_fine = float(sum(state.depth[i] * state.u[i] for i in fine) / 2)
        history.append((q_coarse, q_fine))
    q_coarse, q_fine = history[-1]
    interface_depth = float((frames[-1].depth[coarse]
                             + sum(frames[-1].depth[i] for i in fine) / 2) / 2)
    print(f'theta={theta} q_expected={predicted_q:.6g} q_coarse={q_coarse:.6g} '
          f'q_fine={q_fine:.6g} H_interface={interface_depth:.6g} '
          f'tail={history} residual={budget["residual_m3"]:.6g}')
    assert q_coarse == pytest.approx(predicted_q, rel=2e-3)
    assert q_fine == pytest.approx(predicted_q, rel=2e-3)
    assert q_coarse == pytest.approx(q_fine, rel=1e-4)
    assert max(np.ptp([q[j] for q in history]) for j in (0, 1)) < 1e-4
    assert interface_depth == pytest.approx(1, abs=.02)
    # Scale mass error to the case's inventory/input, as in the basin check.
    water_scale = budget['initial_storage_m3'] + budget['discharge:upstream_m3']
    assert abs(budget['residual_m3']) / water_scale < 1e-4
