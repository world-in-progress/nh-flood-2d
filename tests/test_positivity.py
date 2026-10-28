"""Water shared by several outgoing faces and an extraction boundary."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint
from src.nh_flood_2d.core.run_record import sha256
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess


pytestmark = pytest.mark.usefixtures('taichi_cpu')
START = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


def _cell(row, col):
    return row * 3 + col + 1


def _vertical(row, col):
    return row * 4 + col + 1


def _horizontal(row, col):
    return 13 + row * 3 + col


def _grid(path):
    ne, ns = [], []
    for row in range(3):
        for col in range(3):
            index = _cell(row, col)
            ne.append(f'{index},1,1,1,1,{_vertical(row, col)},'
                      f'{_vertical(row, col + 1)},{_horizontal(row, col)},'
                      f'{_horizontal(row + 1, col)},{5 + 10 * col},'
                      f'{5 + 10 * row},0,10\n')
    for row in range(3):
        for col in range(4):
            left = _cell(row, col - 1) if col else 0
            right = _cell(row, col) if col < 3 else 0
            ns.append(f'{_vertical(row, col)},2,{left},{right},0,0,10,'
                      f'{10 * col},{5 + 10 * row},0,0\n')
    for row in range(4):
        for col in range(3):
            bottom = _cell(row - 1, col) if row else 0
            top = _cell(row, col) if row < 3 else 0
            ns.append(f'{_horizontal(row, col)},1,0,0,{bottom},{top},10,'
                      f'{5 + 10 * col},{10 * row},0,0\n')
    (path / 'ne.txt').write_text(''.join(ne))
    (path / 'ns.txt').write_text(''.join(ns))


def _run(path: Path, *, wet_cell: int, face_flows: dict[int, float], extraction=False):
    path.mkdir()
    _grid(path)
    with rasterio.open(path / 'rain.tif', 'w', driver='GTiff', width=1, height=1,
                       count=1, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 30, 30, 30)) as dst:
        dst.write(np.zeros((1, 1), dtype='float32'), 1)
    (path / 'intervals.csv').write_text(
        'interval_start,interval_end\n'
        f'{START.isoformat()},{(START + timedelta(seconds=1)).isoformat()}\n')

    depth = np.zeros(10, dtype=np.float32)
    depth[wet_cell] = np.float32(.1)  # 100 m2 * 0.1 m = 10 m3
    side_count = max(int(row.split(',')[0]) for row in (path / 'ns.txt').read_text().splitlines()) + 1
    q = np.zeros(side_count, dtype=np.float32)
    for side, value in face_flows.items():
        q[side] = value
    zeros = np.zeros_like(depth)
    time_words = np.zeros(len(depth), dtype=np.uint32)
    write_checkpoint(path / 'seed.fdb', depth=depth, bed=zeros, q=q,
                     rain_time=zeros, infiltration=zeros, max_depth=depth.copy(),
                     first_time_hi=time_words, first_time_lo=time_words,
                     max_speed=zeros, clock_hi=0, clock_lo=0,
                     gate_open=False, metadata_json=json.dumps({
                         'run_state': False,
                         'ne_sha256': sha256(path / 'ne.txt'),
                         'ns_sha256': sha256(path / 'ns.txt'),
                         'gate_config': 'null',
                     }))
    boundaries = []
    if extraction:
        (path / 'river.geojson').write_text(json.dumps({
            'type': 'LineString', 'coordinates': [[0, 0], [0, 10]],
            'crs': {'type': 'name', 'properties': {'name': 'EPSG:2326'}},
        }))
        (path / 'river.csv').write_text(
            'time,value\n'
            f'{START.isoformat()},-100\n'
            f'{(START + timedelta(seconds=1)).isoformat()},-100\n')
        boundaries.append(dict(name='river', kind='discharge',
                               line=str(path / 'river.geojson'),
                               series=str(path / 'river.csv')))
    domain = DomainConfig(ne=str(path / 'ne.txt'), ns=str(path / 'ns.txt'),
                          epsg_code=2326, domain_dir=str(path / 'out'),
                          start=START, duration=1, warmup_duration=0,
                          land_parameters={10: dict(manning_n=1e-6, infiltration='none')},
                          sita=1, yield_step=1, restart_state=str(path / 'seed.fdb'))
    forcing = ForceConfig(boundaries=boundaries, rainfall=dict(
        raster=str(path / 'rain.tif'), intervals=str(path / 'intervals.csv')))
    preprocess(domain, forcing)
    balance = solver(domain, forcing)
    state = read_checkpoint(Path(domain.output_dir) / 'state_final.fdb')
    manifest = json.loads((path / 'out' / 'manifest.json').read_text())
    return balance, state, manifest


def test_four_outgoing_faces_share_one_cell_inventory(tmp_path):
    # CFL for q=1 m2/s and h=0.1 m is about 0.455 s. Four unscaled
    # 10 m faces would carry more than the center's 10 m3 in that one step.
    first_cfl = .5 * 10 / (np.sqrt(9.81 * .1) + 1 / .1)
    assert 4 * 10 * first_cfl > 10
    balance, state, manifest = _run(
        tmp_path / 'center', wet_cell=_cell(1, 1), face_flows={
            _vertical(1, 1): -1, _vertical(1, 2): 1,
            _horizontal(1, 1): -1, _horizontal(2, 1): 1,
        })
    neighbours = state.depth[[ _cell(1, 0), _cell(1, 2),
                               _cell(0, 1), _cell(2, 1) ]]
    assert np.all(state.depth >= 0)
    assert np.all(neighbours > 0)
    np.testing.assert_allclose(neighbours, neighbours[0], rtol=0, atol=2e-7)
    assert balance['initial_storage_m3'] == pytest.approx(10)
    assert balance['storage_m3'] == pytest.approx(10, abs=1e-5)
    assert balance['rainfall_m3'] == balance['infiltration_m3'] == 0
    assert balance['bed_clamp_nonlevel_m3'] <= 1e-5
    assert abs(balance['residual_m3']) <= 1e-5
    assert manifest['steps'] >= 2


def test_negative_discharge_and_faces_share_corner_inventory(tmp_path):
    balance, state, _ = _run(
        tmp_path / 'corner', wet_cell=_cell(0, 0), face_flows={
            _vertical(0, 1): 1, _horizontal(1, 0): 1,
        }, extraction=True)
    assert np.all(state.depth >= 0)
    assert state.depth[_cell(0, 1)] > 0
    assert state.depth[_cell(1, 0)] > 0
    assert balance['requested_discharge:river_m3'] == pytest.approx(-100)
    assert -10 < balance['discharge:river_m3'] < 0
    assert balance['storage_m3'] > 0
    assert balance['bed_clamp_nonlevel_m3'] <= 1e-5
    assert abs(balance['residual_m3']) <= 1e-5
