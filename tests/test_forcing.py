import csv
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.input.force import BoundaryConfig, RainfallConfig, absolute_time
from src.nh_flood_2d.preprocess.force import (ExteriorEdges, MeshGeometry, RainField,
                                             TimeSeries, load_boundaries)

ORIGIN = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)


def test_replaced_helpers_are_no_longer_exposed():
    from src.nh_flood_2d.preprocess.force import Forcing
    from src.nh_flood_2d.input import DomainConfig
    assert not hasattr(Forcing, 'next_event')
    assert not hasattr(DomainConfig, 'hydrograph_dir')


def test_rectangular_non_square_mesh_is_rejected(tmp_path):
    import fastdb4py as fdb
    from test_restart import _inputs
    from src.nh_flood_2d.input import DomainConfig
    from src.nh_flood_2d.preprocess import prepare_domain
    from src.nh_flood_2d.schema.feature import Ns
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    cfg = DomainConfig(ne=str(inputs / 'ne.txt'), ns=str(inputs / 'ns.txt'),
        epsg_code=2326, domain_dir=str(tmp_path / 'out'), start=ORIGIN,
        duration=1, warmup_duration=0,
        land_parameters={10: dict(manning_n=.03, infiltration='none')})
    prepare_domain(cfg)
    ns = fdb.ORM.load(cfg.ns_fdb, from_file=True)
    ne = fdb.ORM.load(cfg.ne_fdb, from_file=True)
    try:
        column = ns[Ns][Ns].column
        column.y[column.y == 10] = 20
        with pytest.raises(ValueError, match='square'):
            MeshGeometry.from_fdb(ne, ns)
    finally:
        ne.close()
        ns.close()


@pytest.fixture
def mesh():
    edges = ExteriorEdges(
        side_ids=np.arange(1, 9),
        cell_ids=np.array([1, 1, 2, 2, 3, 3, 4, 4]),
        face_ids=np.array([0, 2, 2, 1, 0, 3, 3, 1]),
        lengths=np.ones(8),
        segments=np.array([
            [[0, 0], [0, 1]], [[0, 0], [1, 0]],
            [[1, 0], [2, 0]], [[2, 0], [2, 1]],
            [[0, 1], [0, 2]], [[0, 2], [1, 2]],
            [[1, 2], [2, 2]], [[2, 1], [2, 2]],
        ], dtype=float),
    )
    return MeshGeometry(np.array([0., .5, 1.5, .5, 1.5]),
                        np.array([0., .5, .5, 1.5, 1.5]),
                        np.array([[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 2, 1],
                                  [0, 1, 1, 2], [1, 1, 2, 2]]),
                        np.array([0., 1., 1., 1., 1.]), edges)


def line(tmp_path, coords, epsg=2326):
    path = tmp_path / f'line{len(list(tmp_path.glob("*.geojson")))}.geojson'
    path.write_text(json.dumps({'type': 'LineString', 'coordinates': coords,
                               'crs': {'type': 'name', 'properties': {'name': f'EPSG:{epsg}'}}}))
    return str(path)


def series(tmp_path):
    path = tmp_path / 'q.csv'
    path.write_text('time,value\n2023-09-07T16:00:00+08:00,4\n2023-09-07T16:10:00+08:00,8\n')
    return str(path)


def boundary(tmp_path, coords, **kwargs):
    return BoundaryConfig(name=kwargs.get('name', 'river'), kind='discharge',
                          line=line(tmp_path, coords, kwargs.get('epsg', 2326)), series=series(tmp_path))


def test_exterior_line_and_discharge_sum(tmp_path, mesh):
    b = boundary(tmp_path, [[0, 0], [0, 2]])
    result = load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)[0]
    np.testing.assert_array_equal(result.cells, [1, 3])
    np.testing.assert_array_equal(result.side_ids, [1, 5])
    assert result.distribute(7.3).sum() == pytest.approx(7.3, abs=1e-14)
    assert result.series.at(300) == 6


def test_corner_touch_is_excluded(tmp_path, mesh):
    b = boundary(tmp_path, [[-1, -1], [0, 0]])
    with pytest.raises(ValueError, match='snap the line'):
        load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)


def test_crossing_exterior_at_points_is_excluded(tmp_path, mesh):
    b = boundary(tmp_path, [[-.1, .5], [2.1, .5]])
    with pytest.raises(ValueError, match='no exterior edges'):
        load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)


def test_partial_positive_overlap_selects_the_whole_side(tmp_path, mesh):
    b = boundary(tmp_path, [[0, .2], [0, .3]])
    result = load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)[0]
    np.testing.assert_array_equal(result.side_ids, [1])
    np.testing.assert_array_equal(result.cells, [1])
    np.testing.assert_array_equal(result.weights, [1.])


def test_weights_use_only_selected_edges(tmp_path, mesh):
    b = boundary(tmp_path, [[0, 1], [0, 0], [2, 0]])
    result = load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)[0]
    np.testing.assert_array_equal(result.side_ids, [1, 2, 3])
    np.testing.assert_array_equal(result.cells, [1, 2])
    np.testing.assert_allclose(result.weights, [2 / 3, 1 / 3])


@pytest.mark.parametrize('coords,epsg,error', [([[5, 5], [6, 6]], 2326, 'no exterior'),
                                            ([[0, 0], [1, 1]], 4326, 'CRS differs')])
def test_invalid_boundary(tmp_path, mesh, coords, epsg, error):
    b = boundary(tmp_path, coords, epsg=epsg)
    with pytest.raises(ValueError, match=error):
        load_boundaries([b], mesh, 2326, ORIGIN, 0, 600)


def test_overlap_rejected(tmp_path, mesh):
    a = boundary(tmp_path, [[0, 0], [0, 1]])
    b = boundary(tmp_path, [[0, 0], [0, 1]], name='sea')
    with pytest.raises(ValueError, match='overlap'):
        load_boundaries([a, b], mesh, 2326, ORIGIN, 0, 600)


@pytest.mark.parametrize('begin,end', [(-1, 600), (0, 601)])
def test_sequence_coverage_rejected(tmp_path, begin, end):
    with pytest.raises(ValueError, match='does not cover'):
        TimeSeries.load(series(tmp_path), ORIGIN, begin, end)


def test_timezone_required():
    with pytest.raises(ValueError, match='timezone'):
        absolute_time('2023-09-07T08:00:00')


def rain_input(tmp_path, data, transform=None, nodata=None):
    data = np.asarray(data, dtype='float32')
    path = tmp_path / 'rain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[2], height=data.shape[1],
                       count=len(data), dtype='float32', crs='EPSG:2326',
                       transform=transform or from_origin(0, 2, 1, 1), nodata=nodata) as dst:
        dst.write(data)
    intervals = tmp_path / 'rain.csv'
    intervals.write_text('interval_start,interval_end\n2023-09-07T08:00:00Z,2023-09-07T08:05:00Z\n'
                         '2023-09-07T08:05:00Z,2023-09-07T08:10:00Z\n')
    return RainfallConfig(raster=str(path), intervals=str(intervals))


def test_rain_mapping_piecewise_and_volume(tmp_path, mesh):
    r = RainField.load(rain_input(tmp_path, [[[10, 20], [0, 30]], [[0, 0], [0, 0]]]),
                       mesh, 2326, ORIGIN, 0, 600)
    rates = r.rates(r.index_at(299))[r.pixel_index][1:]
    np.testing.assert_allclose(rates * 300, [0, .03, .01, .02])
    assert np.sum(rates * mesh.area[1:] * 300) == pytest.approx(.06)
    assert r.index_at(300) == 1
    assert np.all(r.rates(1) == 0)


@pytest.mark.parametrize('outside', [False, True])
def test_rain_nodata_or_outside_rejected(tmp_path, mesh, outside):
    cfg = rain_input(tmp_path, [[[-9999, 0], [0, 0]], [[0, 0], [0, 0]]],
                     from_origin(10, 2, 1, 1) if outside else None, nodata=-9999)
    with pytest.raises(ValueError, match='outside|NoData'):
        RainField.load(cfg, mesh, 2326, ORIGIN, 0, 600)


@pytest.mark.parametrize('start', [1694054400, '2023-09-07T16:00:00'])
def test_start_json_requires_timezone_iso8601(tmp_path, start):
    from pydantic import ValidationError
    from src.nh_flood_2d.input import DomainConfig
    (tmp_path/'ne').write_text('mesh')
    (tmp_path/'ns').write_text('mesh')
    config=dict(ne=str(tmp_path/'ne'),ns=str(tmp_path/'ns'),epsg_code=2326,
                domain_dir=str(tmp_path/'out'),start=start,duration=1,
                warmup_duration=0,land_parameters={10:dict(manning_n=.03,infiltration='none')})
    with pytest.raises(ValidationError):
        DomainConfig.model_validate_json(json.dumps(config))
