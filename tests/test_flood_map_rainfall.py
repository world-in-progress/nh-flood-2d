from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.nh_flood_2d.input.force import ForceConfig, RainfallConfig
from src.nh_flood_2d.output import flood_map
from src.nh_flood_2d.preprocess.force import ExteriorEdges, MeshGeometry


def test_rainfall_plot_uses_cell_area_interval_duration_and_domain_timezone(tmp_path, monkeypatch):
    start = datetime(2023, 9, 7, 16, tzinfo=timezone(timedelta(hours=8)))
    domain = SimpleNamespace(ne_fdb=str(tmp_path / 'ne.fdb'), ns_fdb=str(tmp_path / 'ns.fdb'),
                             epsg_code=2326, start=start, duration=3600)
    mesh = MeshGeometry(
        x=np.array([0., .5, 1.5]), y=np.array([0., .5, .5]),
        bounds=np.array([[0., 0., 0., 0.], [0., 0., 1., 1.], [1., 0., 2., 1.]]),
        area=np.array([0., 1., 3.]),
        exterior_edges=ExteriorEdges(np.array([], dtype=int), np.array([], dtype=int),
                                    np.array([], dtype=int), np.array([], dtype=float),
                                    np.empty((0, 2, 2))),
    )
    monkeypatch.setattr(flood_map.fdb.RecordEngine, 'load', lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(flood_map.MeshGeometry, 'from_fdb', lambda *args: mesh)

    raster = tmp_path / 'rain.tif'
    with rasterio.open(raster, 'w', driver='GTiff', width=2, height=1, count=3,
                       dtype='float32', crs='EPSG:2326', transform=from_origin(0, 1, 1, 1)) as dst:
        dst.write(np.array([[[2., 6.]], [[4., 0.]], [[0., 8.]]], dtype='float32'))
    intervals = tmp_path / 'intervals.csv'
    intervals.write_text(
        'interval_start,interval_end\n'
        '2023-09-07T07:55:00Z,2023-09-07T08:10:00Z\n'
        '2023-09-07T08:10:00Z,2023-09-07T08:40:00Z\n'
        '2023-09-07T08:40:00Z,2023-09-07T09:10:00Z\n'
    )
    force = ForceConfig(boundaries=[], rainfall=RainfallConfig(raster=str(raster),
                                                                intervals=str(intervals)))

    dates, rates, widths = flood_map._load_rainfall_plot_series(domain, force)

    assert dates == [datetime(2023, 9, 7, 16), datetime(2023, 9, 7, 16, 10),
                     datetime(2023, 9, 7, 16, 40)]
    np.testing.assert_allclose(widths, [600., 1800., 1200.])
    np.testing.assert_allclose(rates, [20., 2., 12.])
    # Integration includes only the official simulation window, weighted by 1:3 m².
    assert np.dot(rates, widths / 3600.) == pytest.approx(25. / 3.)
