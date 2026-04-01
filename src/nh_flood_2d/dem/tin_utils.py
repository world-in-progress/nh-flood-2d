#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIN工具模块 - 提供带面积约束的三角网生成功能
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三角形面积

    Args:
        p1, p2, p3: 三角形三个顶点的坐标 (x, y)

    Returns:
        float: 三角形面积
    """
    # 使用向量叉积计算面积
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def split_triangle(tri_points: np.ndarray, max_area: float) -> List[np.ndarray]:
    """
    分割面积过大的三角形

    Args:
        tri_points: 三角形顶点数组 (3x2)
        max_area: 最大允许面积

    Returns:
        List[np.ndarray]: 分割后的三角形顶点列表
    """
    p1, p2, p3 = tri_points

    # 计算各边中点
    mid12 = (p1 + p2) / 2
    mid23 = (p2 + p3) / 2
    mid31 = (p3 + p1) / 2

    # 从最长边的中点分割
    sides = [
        (np.linalg.norm(p2 - p1), 1, 2),
        (np.linalg.norm(p3 - p2), 2, 3),
        (np.linalg.norm(p1 - p3), 3, 1)
    ]
    sides.sort(reverse=True)  # 从最长边开始

    # 使用最长边的中点进行分割
    if sides[0][1] == 1 and sides[0][2] == 2:
        # 分割p1-p2边
        return [
            np.array([p1, mid12, p3]),
            np.array([p2, mid12, p3])
        ]
    elif sides[0][1] == 2 and sides[0][2] == 3:
        # 分割p2-p3边
        return [
            np.array([p1, p2, mid23]),
            np.array([p1, mid23, p3])
        ]
    else:
        # 分割p3-p1边
        return [
            np.array([p1, p2, mid31]),
            np.array([p2, p3, mid31])
        ]


def create_constrained_tin(points: np.ndarray, max_triangle_area: float = 8.0) -> Delaunay:
    """
    创建带面积约束的Delaunay三角网

    Args:
        points: Nx2点云数组 (x, y)
        max_triangle_area: 最大三角形面积（平方米）

    Returns:
        Delaunay: 约束后的三角网
    """
    logger.info(f"创建带面积约束的TIN: {len(points)}点，最大三角形面积={max_triangle_area}m²")

    # 初始Delaunay三角剖分
    tri = Delaunay(points)

    # 检查三角形面积并递归分割
    modified = True
    iteration = 0

    while modified and iteration < 10:  # 防止无限循环
        modified = False
        iteration += 1

        # 收集需要分割的三角形
        triangles_to_split = []
        for i, simplex in enumerate(tri.simplices):
            # 获取三角形顶点（使用tri.points确保索引正确）
            tri_points = tri.points[simplex]

            # 计算面积

            area = triangle_area(tri_points[0], tri_points[1], tri_points[2])

            if area > max_triangle_area:
                triangles_to_split.append((i, tri_points, area))

        if not triangles_to_split:
            break

        logger.info(f"迭代{iteration}: 发现{len(triangles_to_split)}个三角形需要分割")

        # 分割三角形并添加新点

        new_points = []
        for idx, tri_points, area in triangles_to_split:
            # 分割三角形

            new_tris = split_triangle(tri_points, max_triangle_area)

            # 添加新点（中点）

            for new_tri in new_tris:
                # 添加新三角形的顶点（过滤重复点）
                for point in new_tri:
                    if not any(np.allclose(point, existing) for existing in points):
                        if not any(np.allclose(point, new) for new in new_points):
                            new_points.append(point)

        if new_points:
            # 添加新点并重新三角化
            new_points_array = np.array(new_points)
            points = np.vstack([points, new_points_array])
            tri = Delaunay(points)
            modified = True

            logger.info(f"添加{len(new_points)}个新点，现在总点数: {len(points)}")

    # 最终检查

    large_triangles = []
    for simplex in tri.simplices:
        tri_points = tri.points[simplex]

        area = triangle_area(tri_points[0], tri_points[1], tri_points[2])

        if area > max_triangle_area * 1.1:  # 允许10%误差
            large_triangles.append(area)

    if large_triangles:
        logger.warning(f"仍有{len(large_triangles)}个三角形超过面积限制: "

                      f"最大{max(large_triangles):.2f}m² (限制{max_triangle_area}m²)")

    else:
        logger.info(f"TIN创建完成: {len(tri.simplices)}个三角形，最大面积≤{max_triangle_area}m²")

    return tri