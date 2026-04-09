#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DEM融合脚本（分块掩模替换法）
"""

import pytest
import numpy as np
import tempfile
import os
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入待测试模块
from src.nh_flood_2d.dem.dem_fusion_mask_replacement import resample_dem_to_4m, rasterize_mask, read_bay_points, read_shenzhenhe_points, merge_point_clouds, create_tin_from_points


def test_resample_dem_to_4m():
    """测试DEM重采样功能"""
    # 创建测试数据
    import rasterio
    from rasterio.transform import from_origin

    # 创建临时文件名
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tif')
    os.close(tmp_fd)  # 关闭文件描述符

    try:
        # 创建测试DEM（100x100，10米分辨率）
        transform = from_origin(0, 1000, 10, 10)
        data = np.random.rand(100, 100).astype(np.float32)

        with rasterio.open(tmp_path, 'w', driver='GTiff',
                          height=100, width=100,
                          count=1, dtype=data.dtype,
                          crs='EPSG:4326',
                          transform=transform,
                          nodata=-9999) as dst:
            dst.write(data, 1)

        # 测试重采样到4米
        dem_array, dem_meta = resample_dem_to_4m(tmp_path, 4.0)

        # 验证结果
        assert dem_array is not None
        assert 'width' in dem_meta
        assert 'height' in dem_meta
        assert 'resolution' in dem_meta
        assert dem_meta['resolution'] == 4.0

        # 计算预期尺寸：100*10/4 = 250
        expected_width = int(np.ceil(100 * 10 / 4))
        expected_height = int(np.ceil(100 * 10 / 4))

        assert dem_meta['width'] == expected_width
        assert dem_meta['height'] == expected_height
        assert dem_array.shape == (expected_height, expected_width)

    finally:
        # 清理
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_rasterize_mask():
    """测试掩模栅格化功能"""
    # 创建一个临时的shapefile
    import geopandas as gpd
    from shapely.geometry import Polygon

    # 创建测试多边形
    polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

    with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp:
        tmp_path = tmp.name
        # 需要创建多个shapefile文件（.shp, .shx, .dbf等）
        shapefile_name = tmp_path.replace('.shp', '')

    try:
        # 保存shapefile
        gdf.to_file(f"{shapefile_name}.shp")

        # 定义目标网格参数
        from rasterio.transform import from_origin
        target_transform = from_origin(0, 100, 10, 10)  # 10米分辨率
        target_width = 20
        target_height = 20
        target_crs = 'EPSG:4326'

        # 测试栅格化
        mask_raster = rasterize_mask(
            f"{shapefile_name}.shp",
            target_transform,
            target_width,
            target_height,
            target_crs
        )

        # 验证结果
        assert mask_raster is not None
        assert mask_raster.shape == (target_height, target_width)
        assert mask_raster.dtype == np.uint8

        # 检查是否有像元被栅格化
        assert np.sum(mask_raster) > 0

    finally:
        # 清理shapefile文件
        import glob
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            for file in glob.glob(f"{shapefile_name}*{ext}"):
                try:
                    os.unlink(file)
                except:
                    pass


def test_read_bay_points():
    """测试bay.txt点云读取"""
    # 创建测试数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write("1 100.0 200.0 10.5\n")
        tmp.write("2 101.0 201.0 11.2\n")
        tmp.write("3 102.0 202.0 12.8\n")
        tmp.write("\n")  # 空行
        tmp.write("4 103.0 203.0 13.1\n")
        tmp_path = tmp.name

    try:
        # 导入并测试函数
        from src.nh_flood_2d.dem.dem_fusion_mask_replacement import read_bay_points

        x, y, z = read_bay_points(tmp_path)

        # 验证结果
        assert len(x) == 4
        assert len(y) == 4
        assert len(z) == 4

        assert x[0] == 100.0
        assert y[1] == 201.0
        assert z[2] == 12.8
        assert x[3] == 103.0

    finally:
        os.unlink(tmp_path)


def test_read_shenzhenhe_points():
    """测试shenzhenhe.csv点云读取"""
    # 创建测试数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write("x,y,z\n")
        tmp.write("100.0,200.0,10.5\n")
        tmp.write("101.0,201.0,11.2\n")
        tmp.write("102.0,202.0,12.8\n")
        tmp.write("\n")  # 空行
        tmp.write("103.0,203.0,13.1\n")
        tmp_path = tmp.name

    try:
        # 导入并测试函数
        from src.nh_flood_2d.dem.dem_fusion_mask_replacement import read_shenzhenhe_points

        x, y, z = read_shenzhenhe_points(tmp_path)

        # 验证结果
        assert len(x) == 4
        assert len(y) == 4
        assert len(z) == 4

        assert x[0] == 100.0
        assert y[1] == 201.0
        assert z[2] == 12.8
        assert x[3] == 103.0

    finally:
        os.unlink(tmp_path)

def test_triangle_area():
    """测试三角形面积计算"""
    from src.nh_flood_2d.dem.tin_utils import triangle_area

    # 测试直角三角形 (3-4-5)

    p1 = np.array([0, 0])
    p2 = np.array([3, 0])
    p3 = np.array([0, 4])

    area = triangle_area(p1, p2, p3)

    expected_area = 0.5 * 3 * 4  # 6.0

    assert abs(area - expected_area) < 0.001

    # 测试一般三角形

    p4 = np.array([1, 2])
    p5 = np.array([4, 6])
    p6 = np.array([7, 3])

    area2 = triangle_area(p4, p5, p6)

    # 使用行列式公式验证

    x1, y1 = p4
    x2, y2 = p5
    x3, y3 = p6
    expected_area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2

    assert abs(area2 - expected_area2) < 0.001


def test_create_constrained_tin():
    """测试带面积约束的TIN创建"""
    from src.nh_flood_2d.dem.tin_utils import create_constrained_tin

    # 创建测试点云

    points = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [5, 5],
        [2, 2],
        [8, 2],
        [8, 8],
        [2, 8]
    ])

    # 创建TIN（限制最大面积5平方米）

    tin = create_constrained_tin(points, max_triangle_area=5.0)

    # 验证TIN创建成功

    assert tin is not None
    assert hasattr(tin, 'points')

    assert hasattr(tin, 'simplices')
    assert len(tin.simplices) > 0

    # 验证三角形面积不超过限制（允许10%误差）

    from src.nh_flood_2d.dem.tin_utils import triangle_area

    for simplex in tin.simplices:
        tri_points = tin.points[simplex]

        area = triangle_area(tri_points[0], tri_points[1], tri_points[2])

        assert area <= 5.0 * 1.1, f"三角形面积{area}超过限制5.0平方米"


def test_create_tin_from_points():
    """测试从点云创建TIN"""
    # 创建测试点云数据（分散的点，避免共线）
    points = np.array([
        [100.0, 200.0, 10.5],
        [110.0, 190.0, 11.2],
        [105.0, 205.0, 12.8],
        [95.0, 195.0, 13.1],
        [115.0, 210.0, 14.2],
        [90.0, 185.0, 15.5]
    ])

    # 测试TIN创建（使用较大的面积限制，避免触发分割逻辑）
    tin, xy_points = create_tin_from_points(points, max_triangle_area=100.0)

    # 验证结果
    assert tin is not None
    assert hasattr(tin, 'simplices')
    assert len(tin.simplices) > 0

    assert xy_points is not None
    assert xy_points.shape[0] == len(points)
    assert xy_points.shape[1] == 2  # 只有xy坐标

    # 验证TIN使用正确的点
    assert np.array_equal(xy_points, points[:, :2])
