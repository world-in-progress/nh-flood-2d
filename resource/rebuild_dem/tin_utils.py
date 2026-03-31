"""
TIN（三角不规则网络）工具模块

该模块提供TIN构建、采样和插值相关的工具函数，
用于DEM融合的TIN替换方法。

主要功能：
1. 构建Delaunay三角网
2. 在TIN表面采样点
3. 在TIN表面进行插值
4. 与rasterio和numpy集成

依赖：
- triangle: Python的Delaunay三角剖分库
- numpy: 数值计算
- scipy: 空间计算

作者: Claude Code
日期: 2026-03-31
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

try:
    import triangle
    logger.info("成功导入triangle库")
except ImportError as e:
    logger.error(f"导入triangle库失败: {e}")
    logger.info("请安装triangle库: pip install triangle")
    raise

try:
    from scipy.spatial import Delaunay
    logger.info("成功导入scipy.spatial.Delaunay")
except ImportError as e:
    logger.error(f"导入scipy.spatial.Delaunay失败: {e}")
    raise


def build_delaunay_triangulation(points: np.ndarray,
                                 segments: Optional[np.ndarray] = None,
                                 holes: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    构建Delaunay三角剖分

    参数:
    ----------
    points : np.ndarray
        点坐标数组，形状为 (n, 2)
    segments : np.ndarray, optional
        约束线段，形状为 (m, 2)，每行包含两个点索引
    holes : np.ndarray, optional
        孔洞点坐标，形状为 (k, 2)

    返回:
    ----------
    dict
        包含三角剖分结果的字典，键包括：
        - 'vertices': 顶点坐标
        - 'triangles': 三角形索引
        - 'segments': 约束线段
        - 'holes': 孔洞点
    """
    logger.info(f"构建Delaunay三角剖分，输入点数量: {len(points)}")

    # 准备triangle库的输入数据
    tri_data = {
        'vertices': points
    }

    if segments is not None and len(segments) > 0:
        tri_data['segments'] = segments
        logger.info(f"添加约束线段: {len(segments)} 条")

    if holes is not None and len(holes) > 0:
        tri_data['holes'] = holes
        logger.info(f"添加孔洞: {len(holes)} 个")

    try:
        # 使用triangle库进行约束Delaunay三角剖分
        result = triangle.triangulate(tri_data, 'p')

        vertices = result['vertices']
        triangles = result['triangles']

        logger.info(f"三角剖分完成，顶点数量: {len(vertices)}, 三角形数量: {len(triangles)}")

        return result

    except Exception as e:
        logger.error(f"三角剖分失败: {e}")
        raise


def sample_uniform_points_in_polygon(polygon_vertices: np.ndarray,
                                     spacing: float = 5.0) -> np.ndarray:
    """
    在多边形内均匀采样点

    参数:
    ----------
    polygon_vertices : np.ndarray
        多边形顶点坐标，形状为 (n, 2)
    spacing : float
        采样间距（米）

    返回:
    ----------
    np.ndarray
        采样点坐标，形状为 (m, 2)
    """
    logger.info(f"在多边形内均匀采样点，间距: {spacing}m")

    # TODO: 实现均匀点采样逻辑
    # 1. 计算多边形边界框
    # 2. 生成规则网格点
    # 3. 筛选在多边形内的点

    # 临时返回空数组
    return np.array([])


def build_tin(boundary_points: np.ndarray,
              interior_points: Optional[np.ndarray] = None,
              boundary_segments: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建TIN（三角不规则网络）

    参数:
    ----------
    boundary_points : np.ndarray
        边界点坐标，形状为 (n, 2)
    interior_points : np.ndarray, optional
        内部点坐标，形状为 (m, 2)
    boundary_segments : np.ndarray, optional
        边界线段约束，形状为 (k, 2)

    返回:
    ----------
    Tuple[np.ndarray, np.ndarray]
        (顶点坐标, 三角形索引)
    """
    logger.info("构建TIN")

    # 合并边界点和内部点
    if interior_points is not None and len(interior_points) > 0:
        all_points = np.vstack([boundary_points, interior_points])
        logger.info(f"合并点: {len(boundary_points)} 边界点 + {len(interior_points)} 内部点 = {len(all_points)} 总点")
    else:
        all_points = boundary_points
        logger.info(f"仅使用边界点: {len(all_points)} 个点")

    # 如果没有提供边界线段，则自动生成
    if boundary_segments is None:
        n_boundary = len(boundary_points)
        boundary_segments = np.array([[i, (i + 1) % n_boundary] for i in range(n_boundary)])
        logger.info(f"自动生成边界线段: {len(boundary_segments)} 条")

    # 构建三角剖分
    tri_result = build_delaunay_triangulation(
        points=all_points,
        segments=boundary_segments
    )

    vertices = tri_result['vertices']
    triangles = tri_result['triangles']

    logger.info(f"TIN构建完成: {len(vertices)} 个顶点, {len(triangles)} 个三角形")

    return vertices, triangles


def interpolate_tin(vertices: np.ndarray,
                    triangles: np.ndarray,
                    query_points: np.ndarray,
                    vertex_values: np.ndarray) -> np.ndarray:
    """
    在TIN表面进行插值

    参数:
    ----------
    vertices : np.ndarray
        顶点坐标，形状为 (n, 2)
    triangles : np.ndarray
        三角形索引，形状为 (m, 3)
    query_points : np.ndarray
        查询点坐标，形状为 (k, 2)
    vertex_values : np.ndarray
        顶点值（高程），形状为 (n,)

    返回:
    ----------
    np.ndarray
        插值结果，形状为 (k,)
    """
    logger.info(f"TIN插值: {len(query_points)} 个查询点")

    if len(vertices) != len(vertex_values):
        raise ValueError(f"顶点数量 ({len(vertices)}) 与顶点值数量 ({len(vertex_values)}) 不匹配")

    # 使用scipy的Delaunay进行点定位
    try:
        tri = Delaunay(vertices)

        # 查找每个查询点所在的三角形
        simplex_indices = tri.find_simplex(query_points)

        # 初始化插值结果
        interpolated_values = np.full(len(query_points), np.nan)

        # 对每个查询点进行插值
        for i, (point, simplex_idx) in enumerate(zip(query_points, simplex_indices)):
            if simplex_idx == -1:
                # 点不在任何三角形内
                continue

            # 获取三角形顶点
            triangle = triangles[simplex_idx]
            v0, v1, v2 = vertices[triangle]

            # 计算重心坐标
            v0_value = vertex_values[triangle[0]]
            v1_value = vertex_values[triangle[1]]
            v2_value = vertex_values[triangle[2]]

            # 使用重心坐标进行插值
            denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) +
                    (v2[0] - v1[0]) * (v0[1] - v2[1]))

            if abs(denom) < 1e-12:
                continue

            w0 = ((v1[1] - v2[1]) * (point[0] - v2[0]) +
                 (v2[0] - v1[0]) * (point[1] - v2[1])) / denom
            w1 = ((v2[1] - v0[1]) * (point[0] - v2[0]) +
                 (v0[0] - v2[0]) * (point[1] - v2[1])) / denom
            w2 = 1 - w0 - w1

            # 插值结果
            interpolated_values[i] = (w0 * v0_value + w1 * v1_value + w2 * v2_value)

        # 统计有效插值点
        valid_count = np.sum(~np.isnan(interpolated_values))
        logger.info(f"TIN插值完成: {valid_count}/{len(query_points)} 个点有效")

        return interpolated_values

    except Exception as e:
        logger.error(f"TIN插值失败: {e}")
        raise


def sample_tin_points(vertices: np.ndarray,
                      triangles: np.ndarray,
                      resolution: float = 1.0) -> np.ndarray:
    """
    在TIN表面采样规则网格点

    参数:
    ----------
    vertices : np.ndarray
        顶点坐标，形状为 (n, 2)
    triangles : np.ndarray
        三角形索引，形状为 (m, 3)
    resolution : float
        采样分辨率（米）

    返回:
    ----------
    np.ndarray
        采样点坐标，形状为 (k, 2)
    """
    logger.info(f"在TIN表面采样规则网格点，分辨率: {resolution}m")

    # TODO: 实现TIN表面规则网格采样
    # 1. 计算TIN的边界框
    # 2. 生成规则网格
    # 3. 筛选在TIN内的点

    # 临时返回空数组
    return np.array([])


def calculate_tin_quality(vertices: np.ndarray, triangles: np.ndarray) -> Dict[str, float]:
    """
    计算TIN质量指标

    参数:
    ----------
    vertices : np.ndarray
        顶点坐标
    triangles : np.ndarray
        三角形索引

    返回:
    ----------
    Dict[str, float]
        质量指标字典，包含：
        - 'avg_aspect_ratio': 平均长宽比
        - 'min_angle': 最小内角（度）
        - 'max_angle': 最大内角（度）
        - 'avg_area': 平均面积
    """
    logger.info("计算TIN质量指标")

    if len(triangles) == 0:
        return {
            'avg_aspect_ratio': 0.0,
            'min_angle': 0.0,
            'max_angle': 0.0,
            'avg_area': 0.0
        }

    aspect_ratios = []
    min_angles = []
    max_angles = []
    areas = []

    for tri in triangles:
        # 获取三角形顶点
        a, b, c = vertices[tri]

        # 计算边长
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)

        # 计算面积（海伦公式）
        s = (ab + bc + ca) / 2
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        areas.append(area)

        # 计算内角
        angle_a = np.arccos(np.clip((ab**2 + ca**2 - bc**2) / (2 * ab * ca), -1, 1))
        angle_b = np.arccos(np.clip((ab**2 + bc**2 - ca**2) / (2 * ab * bc), -1, 1))
        angle_c = np.pi - angle_a - angle_b

        angles = np.degrees([angle_a, angle_b, angle_c])
        min_angles.append(np.min(angles))
        max_angles.append(np.max(angles))

        # 计算长宽比（等边三角形为1，越细长值越大）
        if area > 0:
            aspect_ratio = (ab**2 + bc**2 + ca**2) / (4 * np.sqrt(3) * area)
            aspect_ratios.append(aspect_ratio)

    result = {
        'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0.0,
        'min_angle': np.min(min_angles) if min_angles else 0.0,
        'max_angle': np.max(max_angles) if max_angles else 0.0,
        'avg_area': np.mean(areas) if areas else 0.0,
        'num_triangles': len(triangles)
    }

    logger.info(f"TIN质量指标: {result}")
    return result


if __name__ == "__main__":
    # 模块测试代码
    import logging
    logging.basicConfig(level=logging.INFO)

    logger.info("测试TIN工具模块")

    # 创建一个简单的测试用例
    test_points = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]
    ])

    test_segments = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0]
    ])

    logger.info("测试三角剖分...")
    result = build_delaunay_triangulation(test_points, test_segments)

    logger.info("测试质量指标计算...")
    quality = calculate_tin_quality(result['vertices'], result['triangles'])

    logger.info("测试完成")