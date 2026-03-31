#!/usr/bin/env python3
"""
DEM融合（TIN替换方法）单元测试

测试TIN工具模块和主脚本的功能。

测试内容：
1. TIN构建功能测试
2. TIN插值功能测试
3. 点采样功能测试
4. 主脚本集成测试

作者: Claude Code
日期: 2026-03-31
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import Polygon

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入待测试模块
try:
    from tin_utils import (
        build_delaunay_triangulation,
        build_tin,
        interpolate_tin,
        calculate_tin_quality
    )
except ImportError as e:
    print(f"导入TIN工具模块失败: {e}")
    sys.exit(1)


class TestTinUtils(unittest.TestCase):
    """TIN工具模块测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建测试数据：一个简单的正方形区域
        self.test_points = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],  # 边界点
            [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]  # 内部点
        ])

        self.test_segments = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0]  # 边界约束
        ])

        self.test_values = np.array([0.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.8])

    def test_build_delaunay_triangulation(self):
        """测试Delaunay三角剖分"""
        print("测试Delaunay三角剖分...")

        result = build_delaunay_triangulation(
            self.test_points,
            self.test_segments
        )

        # 验证结果
        self.assertIn('vertices', result)
        self.assertIn('triangles', result)

        vertices = result['vertices']
        triangles = result['triangles']

        self.assertEqual(len(vertices), len(self.test_points))
        self.assertGreater(len(triangles), 0)

        # 验证三角形索引有效
        for tri in triangles:
            self.assertEqual(len(tri), 3)
            for idx in tri:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(vertices))

        print(f"三角剖分成功: {len(triangles)} 个三角形")

    def test_build_tin(self):
        """测试TIN构建"""
        print("测试TIN构建...")

        vertices, triangles = build_tin(
            boundary_points=self.test_points[:4],
            interior_points=self.test_points[4:],
            boundary_segments=self.test_segments
        )

        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(triangles, np.ndarray)
        self.assertEqual(vertices.shape[1], 2)
        self.assertEqual(triangles.shape[1], 3)

        print(f"TIN构建成功: {len(vertices)} 顶点, {len(triangles)} 三角形")

    def test_interpolate_tin(self):
        """测试TIN插值"""
        print("测试TIN插值...")

        # 构建简单的TIN
        vertices = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])

        vertex_values = np.array([0.0, 0.0, 1.0, 1.0])

        # 查询点
        query_points = np.array([
            [0.5, 0.5],  # 在TIN内
            [0.2, 0.8],  # 在TIN内
            [1.5, 1.5]   # 在TIN外
        ])

        interpolated = interpolate_tin(
            vertices, triangles, query_points, vertex_values
        )

        self.assertEqual(len(interpolated), len(query_points))
        self.assertFalse(np.isnan(interpolated[0]))  # 第一个点应该有值
        self.assertFalse(np.isnan(interpolated[1]))  # 第二个点应该有值
        self.assertTrue(np.isnan(interpolated[2]))   # 第三个点应该在TIN外

        # 验证插值合理性
        self.assertAlmostEqual(interpolated[0], 0.5, delta=0.1)

        print(f"插值成功: {interpolated}")

    def test_calculate_tin_quality(self):
        """测试TIN质量指标计算"""
        print("测试TIN质量指标计算...")

        vertices = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])

        quality = calculate_tin_quality(vertices, triangles)

        self.assertIn('avg_aspect_ratio', quality)
        self.assertIn('min_angle', quality)
        self.assertIn('max_angle', quality)
        self.assertIn('avg_area', quality)
        self.assertIn('num_triangles', quality)

        self.assertEqual(quality['num_triangles'], 2)
        self.assertGreater(quality['avg_area'], 0)

        print(f"质量指标: {quality}")

    def test_empty_input(self):
        """测试空输入"""
        print("测试空输入处理...")

        with self.assertRaises(ValueError):
            build_delaunay_triangulation(np.array([]))

        # 测试空三角形质量计算
        quality = calculate_tin_quality(
            np.array([[0, 0]]),
            np.array([])
        )

        self.assertEqual(quality['num_triangles'], 0)
        self.assertEqual(quality['avg_area'], 0.0)

        print("空输入处理测试完成")


class TestDemFusionIntegration(unittest.TestCase):
    """DEM融合集成测试类"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        print(f"创建临时目录: {self.temp_dir}")

    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"清理临时目录: {self.temp_dir}")

    def test_create_test_mask(self):
        """测试创建掩膜文件"""
        print("测试创建掩膜文件...")

        # 创建一个简单的正方形掩膜
        polygon = Polygon([
            (0, 0), (10, 0), (10, 10), (0, 10)
        ])

        gdf = gpd.GeoDataFrame(
            {'id': [1], 'geometry': [polygon]},
            crs='EPSG:32650'
        )

        mask_path = os.path.join(self.temp_dir, 'test_mask.shp')
        gdf.to_file(mask_path)

        self.assertTrue(os.path.exists(mask_path))
        print(f"掩膜文件创建成功: {mask_path}")

    def test_create_test_dem(self):
        """测试创建DEM文件"""
        print("测试创建DEM文件...")

        # 创建一个简单的DEM
        dem_array = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ], dtype=np.float32)

        transform = Affine.translation(0, 3) * Affine.scale(1, -1)

        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': 3,
            'height': 3,
            'count': 1,
            'crs': 'EPSG:32650',
            'transform': transform,
            'compress': 'lzw'
        }

        dem_path = os.path.join(self.temp_dir, 'test_dem.tif')
        with rasterio.open(dem_path, 'w', **profile) as dst:
            dst.write(dem_array, 1)

        self.assertTrue(os.path.exists(dem_path))
        print(f"DEM文件创建成功: {dem_path}")

    def test_main_script_import(self):
        """测试主脚本导入"""
        print("测试主脚本导入...")

        try:
            # 尝试导入主脚本
            import dem_fusion_tin_replacement
            self.assertTrue(hasattr(dem_fusion_tin_replacement, 'main'))
            print("主脚本导入成功")
        except ImportError as e:
            self.fail(f"导入主脚本失败: {e}")


class TestPerformance(unittest.TestCase):
    """性能测试类"""

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        print("测试大数据集性能...")

        # 创建大量测试点
        n_points = 1000
        np.random.seed(42)
        points = np.random.rand(n_points, 2) * 100

        # 创建边界约束（简单的凸包）
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        boundary_points = points[hull.vertices]
        n_boundary = len(boundary_points)

        # 创建边界线段
        segments = np.array([[i, (i + 1) % n_boundary] for i in range(n_boundary)])

        print(f"测试大数据集: {n_points} 个点, {len(boundary_points)} 个边界点")

        # 测试三角剖分性能
        import time
        start_time = time.time()

        result = build_delaunay_triangulation(points, segments)

        elapsed = time.time() - start_time
        print(f"三角剖分耗时: {elapsed:.3f} 秒")

        vertices = result['vertices']
        triangles = result['triangles']

        self.assertEqual(len(vertices), n_points)
        self.assertGreater(len(triangles), 0)

        # 测试质量指标计算性能
        start_time = time.time()
        quality = calculate_tin_quality(vertices, triangles)
        elapsed = time.time() - start_time
        print(f"质量指标计算耗时: {elapsed:.3f} 秒")

        self.assertIn('num_triangles', quality)
        print(f"生成三角形数量: {quality['num_triangles']}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行DEM融合（TIN替换方法）单元测试")
    print("=" * 60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTinUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestDemFusionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print("测试完成")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)