"""Raster readers consume depth frames, bed events, and per-step extrema."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from contextlib import closing

import fastdb4py as fdb
import numpy as np
import pytest
import rasterio

from src.nh_flood_2d.core.output import write_bed, write_depth_frame, write_extrema, write_manifest
from src.nh_flood_2d.output import flood_map
from src.nh_flood_2d.schema.feature import Ne
from src.nh_flood_2d.util.ti import init_taichi

START = datetime(2023, 9, 7, 17, 30, tzinfo=timezone.utc)


def _ne(path):
    with closing(fdb.RecordEngine.truncate([fdb.Layout(Ne, 3)])) as db:
        column = db.table(Ne).column
        column.x[:] = [0, 5, 15]
        column.y[:] = [0, 5, 5]
        db.save(str(path))


def _output(root, depth, bed, maximum=None, events=None):
    output = root / 'output'
    write_bed(output / 'bed.fdb', np.asarray(bed, dtype=np.float32))
    write_depth_frame(output / 'frames/depth_0.fdb', np.asarray(depth, dtype=np.float32))
    max_depth = depth if maximum is None else maximum
    write_extrema(output / 'extrema.fdb', np.asarray(max_depth, dtype=np.float32),
                  np.zeros(3, dtype=np.uint32), np.zeros(3, dtype=np.uint32),
                  np.zeros(3, dtype=np.float32))
    write_manifest(root / 'manifest.json', {'bed_events': events or []})


def _geometry(monkeypatch):
    monkeypatch.setattr(flood_map, 'TILE_SIZE', 16)
    monkeypatch.setattr(flood_map, 'init_taichi', lambda: init_taichi(use_gpu=False))
    monkeypatch.setattr(flood_map, 'get_area_meta',
        lambda *_: (np.array([0, 0, 20, 10]), (10., 10.),
                    np.array([5., 5.], dtype=np.float32), np.array([5., 5.], dtype=np.float32)))


def test_flood_exports_use_frame_depth_and_per_step_maximum(tmp_path, monkeypatch):
    ne, ns = tmp_path / 'ne.fdb', tmp_path / 'ns.fdb'
    _ne(ne)
    ns.touch()
    _output(tmp_path, [0, .25, .0001], [0, 0, 10000], [0, .5, .0001])
    cfg = SimpleNamespace(ne_fdb=str(ne), ns_fdb=str(ns), domain_dir=str(tmp_path),
                          start=START, epsg_code=2326, min_h=.00001,
                          flood_map_dir=str(tmp_path / 'flood'),
                          max_inundation_dir=str(tmp_path / 'maximum'))
    Path(cfg.max_inundation_dir).mkdir()
    _geometry(monkeypatch)

    flood_map.generate_flood_map(cfg)
    with rasterio.open(next(Path(cfg.flood_map_dir).glob('*.tif'))) as src:
        np.testing.assert_allclose(src.read(1)[0], [.25, .0001])
    flood_map.generate_max_inundation_extent_map(cfg, min_depth=.00001)
    with rasterio.open(Path(cfg.max_inundation_dir) / 'max_inundation.tif') as src:
        np.testing.assert_allclose(src.read(1)[0], [.5, .0001])
    flood_map.generate_max_inundation_extent_map(cfg, min_depth=.00001,
                                                 invalid_data=-128, is_absolute=True)
    with rasterio.open(Path(cfg.max_inundation_dir) / 'max_inundation.tif') as src:
        np.testing.assert_array_equal(src.read(1)[0], [1, 1])


def test_wse_mae_replays_gate_event_on_each_mesh(tmp_path, monkeypatch):
    configs = []
    for name, first_depth in [('reference', .25), ('comparison', .5)]:
        root = tmp_path / name
        root.mkdir()
        ne, ns = root / 'ne.fdb', root / 'ns.fdb'
        _ne(ne)
        ns.touch()
        _output(root, [0, first_depth, 0], [0, 10, 0],
                events=[{'elapsed_seconds': 0, 'cell_ids': [1], 'new_bed': [0]}])
        configs.append(SimpleNamespace(ne_fdb=str(ne), ns_fdb=str(ns),
                                       domain_dir=str(root), start=START))
    _geometry(monkeypatch)
    mae = flood_map.plot_spatial_mae_curve(*configs, min_depth=.05, show=False,
                                           output_path=str(tmp_path / 'mae.png'))
    assert mae == pytest.approx([.25])


@pytest.mark.parametrize('depths', [([0, 0, 0], [0, 0, 0]),
                                   ([0, .25, 0], [0, 0, .5])])
def test_wse_mae_is_undefined_without_common_wet_cells(tmp_path, monkeypatch, depths):
    configs = []
    for name, depth in zip(['reference', 'comparison'], depths):
        root = tmp_path / name
        root.mkdir()
        ne, ns = root / 'ne.fdb', root / 'ns.fdb'
        _ne(ne)
        ns.touch()
        _output(root, depth, [0, 0, 0])
        configs.append(SimpleNamespace(ne_fdb=str(ne), ns_fdb=str(ns),
                                       domain_dir=str(root), start=START))
    _geometry(monkeypatch)
    values = flood_map.plot_spatial_mae_curve(*configs, min_depth=.05, show=False,
                                               output_path=str(tmp_path / 'mae.png'))
    assert len(values) == 1 and np.isnan(values[0])
