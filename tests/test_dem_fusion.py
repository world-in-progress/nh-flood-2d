"""Integration checks for optional external DEM and point-cloud inputs."""
from pathlib import Path

import numpy as np
import pytest
import rasterio


POINTS = Path('归档 2/gcd.txt')
DEM = Path('Digital Terrain Model.tif')


@pytest.mark.skipif(not POINTS.is_file(), reason='External point cloud 归档 2/gcd.txt is unavailable')
def test_point_reading():
    from src.nh_flood_2d.dem.dem_fusion import read_xyz_points
    x, y, z = read_xyz_points(str(POINTS), skip_rows=0)
    assert len(x) == len(y) == len(z) and len(x) > 0
    assert np.all(np.isfinite(np.column_stack([x, y, z])))


@pytest.mark.skipif(not DEM.is_file(), reason='External DEM Digital Terrain Model.tif is unavailable')
def test_dem_reading():
    with rasterio.open(DEM) as source:
        assert source.width > 0 and source.height > 0 and source.crs is not None
        data = source.read(1, window=rasterio.windows.Window(0, 0, 100, 100), masked=True)
        assert data.count() > 0
        assert np.all(np.isfinite(data.compressed()))
