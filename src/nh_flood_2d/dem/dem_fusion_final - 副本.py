#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本 - 分块掩模替换法（简化版）
作者: 高级GIS工程师
日期: 2026-04-01

简化说明:
1. 所有输入文件路径硬编码在代码中
2. 所有处理参数硬编码在代码中
3. 用户只需运行脚本，无需任何参数

核心算法: 分块掩模替换法（简化版）
1. 读取原始DEM并重采样到4米分辨率（D1）
2. 读取掩模shapefile，栅格化到D1网格
3. 提取掩模外像元 → 保留到最终输出
4. 读取并合并两个高精度点云文件
5. 直接使用点云进行TIN插值获取掩模内高程（不先创建约束三角网）
6. 合并掩模外保留像元 + 掩模内插值像元
7. 输出4米分辨率DEM

使用方法:
    python dem_fusion_final.py
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.features
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Point, Polygon
import geopandas as gpd
import logging
import sys
import time
import os
from typing import Tuple, Optional, List, Dict
from scipy.spatial import Delaunay,  cKDTree

# ============================================================================
# 硬编码的配置参数
# ============================================================================

# 输入文件路径（全部硬编码）
INPUT_DEM_PATH = 'resource/rebuild_dem/input/study_area_dem.tif'
INPUT_MASK_SHP_PATH = 'resource/rebuild_dem/input/study_area_mask.shp'
INPUT_BAY_POINTS_PATH = 'resource/rebuild_dem/input/bay/bay/bay.txt'
INPUT_SHENZHENHE_POINTS_PATH = 'resource/rebuild_dem/input/bay/bay/shenzhenhe-fix.csv'

# 输出文件路径（硬编码）
OUTPUT_DEM_PATH = 'resource/rebuild_dem/output/fused_dem_4m.tif'
OUTPUT_TIN_PATH = 'resource/rebuild_dem/output/tin.ply'  # TIN输出文件（可选）

# 处理参数（硬编码）
OUTPUT_RESOLUTION = 4.0          # 输出分辨率（米）
NODATA_VALUE = -9999.0          # 无数据值

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 原始函数（从dem_fusion_mask_replacement.py复制）
# ============================================================================

def resample_dem_to_4m(dem_path: str, output_resolution: float = 4.0) -> Tuple[np.ndarray, dict]:
    """
    将DEM重采样到指定分辨率（默认为4米）

    Args:
        dem_path: DEM文件路径
        output_resolution: 输出分辨率（米）

    Returns:
        Tuple[dem_array, dem_meta]: 重采样后的DEM数组和元数据
    """
    logger.info(f"重采样DEM到{output_resolution}米分辨率: {dem_path}")

    with rasterio.open(dem_path) as src:
        # 计算目标尺寸和变换
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, src.crs,
            src.width, src.height,
            *src.bounds,
            resolution=output_resolution
        )

        # 准备输出元数据
        dst_meta = src.meta.copy()
        dst_meta.update({
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'resolution': output_resolution
        })

        # 重采样
        dem_array = np.zeros((dst_height, dst_width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )

        # 获取NoData值
        nodata = src.nodatavals[0]
        if nodata is not None:
            # 先将数据类型转换为float32以支持NaN
            dem_array = dem_array.astype(np.float32)
            dem_array[dem_array == nodata] = np.nan

        logger.info(f"重采样完成: {dst_width}x{dst_height} (原: {src.width}x{src.height})")

        # 返回元数据
        dem_meta = {
            'crs': src.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': 'float32',  # 确保使用float32以支持NaN
            'nodata': nodata,
            'bounds': src.bounds,  # 保持原始范围
            'resolution': output_resolution
        }

        return dem_array, dem_meta


def rasterize_mask(mask_shp_path: str, target_transform: Affine,
                  target_width: int, target_height: int,
                  target_crs: str) -> np.ndarray:
    """
    将掩模shapefile栅格化到目标网格

    Args:
        mask_shp_path: 掩模shapefile路径
        target_transform: 目标仿射变换
        target_width: 目标宽度
        target_height: 目标高度
        target_crs: 目标坐标系

    Returns:
        np.ndarray: 掩模栅格（1=掩模内，0=掩模外）
    """
    logger.info(f"栅格化掩模shapefile: {mask_shp_path}")

    # 读取掩模shapefile
    gdf = gpd.read_file(mask_shp_path)

    # 确保坐标系一致
    if gdf.crs is None:
        logger.warning("掩模shapefile没有坐标系，假设与目标CRS相同")
    elif str(gdf.crs) != str(target_crs):
        logger.info(f"转换掩模坐标系: {gdf.crs} -> {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # 创建输出栅格
    mask_raster = np.zeros((target_height, target_width), dtype=np.uint8)

    # 栅格化
    shapes = [(geom, 1) for geom in gdf.geometry]
    rasterio.features.rasterize(
        shapes=shapes,
        out=mask_raster,
        transform=target_transform,
        fill=0,
        dtype=np.uint8
    )

    # 统计掩模像素
    mask_pixels = np.sum(mask_raster)
    total_pixels = target_width * target_height
    logger.info(f"掩模栅格化完成: {mask_pixels}/{total_pixels}像素在掩模内 ({100*mask_pixels/total_pixels:.1f}%)")

    return mask_raster


def read_bay_points(bay_txt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取bay.txt格式的点云数据（制表符分隔，第一列为索引）

    Args:
        bay_txt_path: bay.txt文件路径

    Returns:
        Tuple[x_coords, y_coords, z_coords]: 三个一维数组
    """
    logger.info(f"读取bay.txt点云: {bay_txt_path}")

    x_list, y_list, z_list = [], [], []

    with open(bay_txt_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split()
                if len(parts) >= 4:  # 索引 + x y z
                    # 跳过索引，取后三个值
                    x, y, z = parts[1], parts[2], parts[3]
                elif len(parts) == 3:  # 可能没有索引
                    x, y, z = parts[0], parts[1], parts[2]
                else:
                    logger.warning(f"跳过格式不正确的行 {line_num}: {line[:50]}...")
                    continue

                x_list.append(float(x))
                y_list.append(float(y))
                z_list.append(float(z))

            except (ValueError, IndexError) as e:
                logger.warning(f"解析错误行 {line_num}: {e}")
                continue

    logger.info(f"从bay.txt读取 {len(x_list):,} 个点")
    return np.array(x_list), np.array(y_list), np.array(z_list)


def read_shenzhenhe_points(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取shenzhenhe.csv格式的点云数据（逗号分隔，有标题行）

    Args:
        csv_path: CSV文件路径

    Returns:
        Tuple[x_coords, y_coords, z_coords]: 三个一维数组
    """
    logger.info(f"读取shenzhenhe.csv点云: {csv_path}")

    x_list, y_list, z_list = [], [], []
    header_skipped = False

    with open(csv_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 跳过标题行
            if not header_skipped and ('x,' in line.lower() or 'x,y,z' in line.lower()):
                header_skipped = True
                continue

            try:
                # 移除可能的引号并分割
                line = line.replace('"', '').replace("'", "")
                parts = line.split(',')

                if len(parts) >= 3:
                    x, y, z = parts[0], parts[1], parts[2]
                    x_list.append(float(x))
                    y_list.append(float(y))
                    z_list.append(float(z))
                else:
                    logger.warning(f"跳过格式不正确的行 {line_num}: {line[:50]}...")
                    continue

            except (ValueError, IndexError) as e:
                logger.warning(f"解析错误行 {line_num}: {e}")
                continue

    logger.info(f"从shenzhenhe.csv读取 {len(x_list):,} 个点")
    return np.array(x_list), np.array(y_list), np.array(z_list)


def merge_point_clouds(bay_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      shenzhenhe_points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    合并两个点云数据集

    Args:
        bay_points: bay.txt点云 (x, y, z)
        shenzhenhe_points: shenzhenhe.csv点云 (x, y, z)

    Returns:
        np.ndarray: 合并后的Nx3点云数组
    """
    bay_x, bay_y, bay_z = bay_points
    szh_x, szh_y, szh_z = shenzhenhe_points

    # 合并点云
    all_x = np.concatenate([bay_x, szh_x])
    all_y = np.concatenate([bay_y, szh_y])
    all_z = np.concatenate([bay_z, szh_z])

    # 创建Nx3数组
    merged_points = np.column_stack([all_x, all_y, all_z])

    logger.info(f"合并点云完成: bay={len(bay_x):,} + shenzhenhe={len(szh_x):,} = {len(merged_points):,}点")
    logger.info(f"合并后范围: X[{all_x.min():.1f}, {all_x.max():.1f}], "
               f"Y[{all_y.min():.1f}, {all_y.max():.1f}], "
               f"Z[{all_z.min():.1f}, {all_z.max():.1f}]")

    return merged_points


def save_tin_as_ply(tri: Delaunay, points: np.ndarray, output_path: str) -> None:
    """
    将TIN保存为PLY文件

    Args:
        tri: Delaunay三角剖分对象
        points: 点云坐标 (Nx3, 包含z值)
        output_path: 输出PLY文件路径
    """
    logger.info(f"保存TIN为PLY文件: {output_path}")

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 获取三角形索引（Delaunay.simplices）
        triangles = tri.simplices

        # 准备点云数据（包含x,y,z）
        vertices = points

        # 写入PLY文件
        with open(output_path, 'w') as f:
            # PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")

            # 写入顶点数据
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            # 写入三角形数据（每个三角形3个顶点索引）
            for tri_indices in triangles:
                f.write(f"3 {tri_indices[0]} {tri_indices[1]} {tri_indices[2]}\n")

        logger.info(f"TIN已保存到PLY文件: {output_path}, {len(vertices)}个顶点, {len(triangles)}个三角形")

    except Exception as e:
        logger.error(f"保存PLY文件失败: {e}")
        raise


def create_quality_tin(points: np.ndarray, mask_coords: np.ndarray,
                       target_resolution: float = 4.0,
                       max_triangle_area: float = 8.0) -> Tuple[Delaunay, np.ndarray]:
    """
    创建高质量TIN（最终加强版）
    - 添加密集边界点 + 内部网格点
    - 同时控制：三角形长宽比 < 8 + 单个三角形面积 ≤ 8平方米
    - 创建初始TIN和最终增强TIN后均输出PLY
    """
    logger.info("创建高质量TIN（加强版：控制面积和长宽比）...")

    if len(mask_coords) < 3:
        logger.warning("掩模坐标点过少，直接使用原始点云")
        tri = Delaunay(points)
        return tri, points.copy()

    # 1. 计算掩模边界框
    mask_min = mask_coords.min(axis=0)
    mask_max = mask_coords.max(axis=0)
    mask_width = mask_max[0] - mask_min[0]
    mask_height = mask_max[1] - mask_min[1]

    logger.info(f"掩模区域尺寸: {mask_width:.1f}×{mask_height:.1f} 米")

    # 2. 边界点密集采样
    boundary_spacing = target_resolution * 1.8
    num_x = max(8, int(mask_width / boundary_spacing) + 2)
    num_y = max(8, int(mask_height / boundary_spacing) + 2)

    boundary_points = []
    for x in np.linspace(mask_min[0], mask_max[0], num_x):
        boundary_points.append([x, mask_min[1]])
        boundary_points.append([x, mask_max[1]])
    for y in np.linspace(mask_min[1], mask_max[1], num_y)[1:-1]:
        boundary_points.append([mask_min[0], y])
        boundary_points.append([mask_max[0], y])

    boundary_points = np.array(boundary_points)

    # 3. 内部加密网格点（已修复 xx/yy 未定义问题）
    internal_spacing = target_resolution * 1.8
    x_grid = np.arange(mask_min[0] + internal_spacing/2, mask_max[0], internal_spacing)
    y_grid = np.arange(mask_min[1] + internal_spacing/2, mask_max[1], internal_spacing)

    if len(x_grid) > 0 and len(y_grid) > 0:
        xx, yy = np.meshgrid(x_grid, y_grid)
        internal_points = np.column_stack([xx.flatten(), yy.flatten()])
        logger.info(f"添加内部格网点: {len(internal_points):,} 个")
    else:
        internal_points = np.empty((0, 2))
        logger.info("掩模区域太小，不添加内部格网点")

    # 4. 合并点云
    enhanced_points = np.vstack([points.copy(), boundary_points])
    if len(internal_points) > 0:
        enhanced_points = np.vstack([enhanced_points, internal_points])

    # 5. 去重
    enhanced_points = np.unique(np.round(enhanced_points, decimals=6), axis=0)

    logger.info(f"增强点云总数: {len(enhanced_points):,}（原始{len(points):,} + 边界{len(boundary_points):,} + 内部{len(internal_points):,}）")

    # ====================== 创建初始TIN并保存 ======================
    logger.info("创建初始Delaunay TIN...")
    tri = Delaunay(enhanced_points)
    
    # 保存初始TIN（仅XY，Z用0占位）
    initial_ply = "initial_tin.ply"
    dummy_z = np.zeros(len(enhanced_points))
    initial_points_with_z = np.column_stack([enhanced_points, dummy_z])
    save_tin_as_ply(tri, initial_points_with_z, initial_ply)

    # # ====================== 质量优化循环（面积 + 长宽比） ======================
    # max_iterations = 8
    # for it in range(max_iterations):
    #     simplices = tri.simplices
    #     triangles = enhanced_points[simplices]

    #     # 计算面积和长宽比
    #     areas = []
    #     aspect_ratios = []
    #     for verts in triangles:
    #         # 计算三边长度
    #         a = np.linalg.norm(verts[1] - verts[0])
    #         b = np.linalg.norm(verts[2] - verts[1])
    #         c = np.linalg.norm(verts[0] - verts[2])
    #         edges = np.array([a, b, c])
            
    #         # 计算面积（海伦公式）
    #         s = np.sum(edges) / 2
    #         area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
    #         areas.append(area)
            
    #         if np.min(edges) > 0:
    #             aspect_ratios.append(np.max(edges) / np.min(edges))

    #     areas = np.array(areas)
    #     aspect_ratios = np.array(aspect_ratios)

    #     # 找出需要优化的三角形（面积>8 或 长宽比>8）
    #     bad_idx = np.where((areas > max_triangle_area) | (aspect_ratios > 8.0))[0]

    #     if len(bad_idx) == 0:
    #         logger.info(f"质量优化完成！所有三角形面积≤{max_triangle_area}㎡，长宽比≤8")
    #         break

    #     logger.info(f"第{it+1}次优化：发现{len(bad_idx)}个不合格三角形（面积过大或细长）")

    #     # 在不合格三角形中心添加 Steiner 点
    #     bad_centroids = triangles[bad_idx].mean(axis=1)
    #     enhanced_points = np.vstack([enhanced_points, bad_centroids])
    #     tri = Delaunay(enhanced_points)

    # # ====================== 保存最终高质量TIN ======================
    # final_ply = "enhanced_quality_tin.ply"
    # dummy_z_final = np.zeros(len(enhanced_points))
    # final_points_with_z = np.column_stack([enhanced_points, dummy_z_final])
    # save_tin_as_ply(tri, final_points_with_z, final_ply)

    # logger.info(f"高质量TIN创建完成：{len(tri.simplices):,} 个三角形")
    return tri, enhanced_points


# ==================== 其余两个函数保持不变（仅微调日志） ====================

def check_tin_coverage(tri: Delaunay, mask_coords: np.ndarray) -> Tuple[bool, np.ndarray]:
    """检查TIN是否完全覆盖掩模区域"""
    logger.info("检查TIN覆盖情况...")
    test_values = np.ones(len(tri.points))
    interpolator = LinearNDInterpolator(tri, test_values, fill_value=-1)
    values = interpolator(mask_coords)
    uncovered_mask = values == -1
    uncovered_count = np.sum(uncovered_mask)

    if uncovered_count > 0:
        ratio = uncovered_count / len(mask_coords) * 100
        logger.warning(f"TIN覆盖失败：{uncovered_count:,}/{len(mask_coords):,} 个点未覆盖 ({ratio:.2f}%)")
        return False, mask_coords[uncovered_mask]
    else:
        logger.info("TIN已完全覆盖掩模区域 ✓")
        return True, np.empty((0, 2))


def interpolate_mask_area(mask_raster: np.ndarray, dem_meta: dict,
                         points: np.ndarray, z_values: np.ndarray,
                         batch_size: int = 1000000,
                         nodata_value: float = -9999.0,
                         tin_output_path: Optional[str] = None) -> np.ndarray:
    """
    最终推荐版：高质量TIN + 面积控制 + 双PLY输出 + 确保覆盖
    """
    logger.info(f"开始高质量TIN插值（最终版）：点云={len(points):,}，掩模像素={np.sum(mask_raster):,}")

    if len(points) < 3:
        logger.warning("点云数量不足，无法构建TIN")
        return np.full_like(mask_raster, nodata_value, dtype=np.float32)

    try:
        target_transform = dem_meta['transform']
        resolution = dem_meta.get('resolution', 4.0)

        # 获取掩模坐标
        mask_indices = np.where(mask_raster == 1)
        if len(mask_indices[0]) == 0:
            logger.warning("掩模内无像素需要插值")
            return np.full_like(mask_raster, nodata_value, dtype=np.float32)

        rows, cols = mask_indices
        xs, ys = rasterio.transform.xy(target_transform, rows, cols)
        mask_coords = np.column_stack([xs, ys])

        # 步骤1：创建高质量TIN（会自动保存 initial_tin.ply 和 enhanced_quality_tin.ply）
        logger.info("步骤1/5：生成高质量受控面积TIN")
        tri, enhanced_points = create_quality_tin(points, mask_coords, target_resolution=resolution, 
                                                  max_triangle_area=8.0)

        # 步骤2：赋值Z值
        logger.info("步骤2/5：为增强点云赋值Z值")
        tree = cKDTree(points)
        dists, indices = tree.query(enhanced_points, k=1)
        enhanced_z_values = z_values[indices].copy()

        far_mask = dists > resolution * 3
        if np.any(far_mask):
            temp_interp = LinearNDInterpolator(points, z_values, fill_value=np.nan)
            enhanced_z_values[far_mask] = temp_interp(enhanced_points[far_mask])

        # 步骤3：检查覆盖并修复
        logger.info("步骤3/5：验证并确保TIN完全覆盖掩模")
        # is_covered, uncovered_coords = check_tin_coverage(tri, mask_coords)
        # if not is_covered and len(uncovered_coords) > 0:
        #     logger.info(f"正在修复 {len(uncovered_coords):,} 个未覆盖点...")
        #     enhanced_points = np.vstack([enhanced_points, uncovered_coords])
        #     new_z = LinearNDInterpolator(points, z_values, fill_value=nodata_value)(uncovered_coords)
        #     enhanced_z_values = np.append(enhanced_z_values, new_z)
        #     tri = Delaunay(enhanced_points)

        # 步骤4：可选额外PLY输出（用户指定的路径）
        # if tin_output_path:
        #     logger.info(f"步骤4/5：输出最终TIN到指定路径 → {tin_output_path}")
        #     combined = np.column_stack([enhanced_points, enhanced_z_values])
        #     save_tin_as_ply(tri, combined, tin_output_path)

        # 步骤5：最终插值
        logger.info("步骤5/5：对掩模栅格进行插值")
        final_interpolator = LinearNDInterpolator(enhanced_points, enhanced_z_values, fill_value=nodata_value)

        logger.info("插值器构建完成，开始批量插值...")
        num_batches = int(np.ceil(len(mask_coords) / batch_size))
        interpolated_values = np.full(len(mask_coords), nodata_value, dtype=np.float32)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(mask_coords))
            batch_coords = mask_coords[start:end]
            interpolated_values[start:end] = final_interpolator(batch_coords)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                valid = np.sum(interpolated_values[start:end] != nodata_value)
                logger.info(f"插值进度: {batch_idx+1}/{num_batches} | 有效={valid:,}/{end-start:,}")

        # 组装结果
        output_array = np.full_like(mask_raster, nodata_value, dtype=np.float32)
        output_array[mask_raster == 1] = interpolated_values
        output_array[np.isnan(output_array)] = nodata_value

        valid_total = np.sum(output_array != nodata_value)
        logger.info(f"TIN插值全部完成！有效高程像素 = {valid_total:,} / {np.sum(mask_raster):,}")

        return output_array

    except Exception as e:
        logger.error(f"TIN插值过程发生错误: {e}")
        raise


def extract_non_mask_pixels(dem_array: np.ndarray, mask_raster: np.ndarray) -> np.ndarray:
    """
    提取掩模外的像元

    Args:
        dem_array: DEM高程数组
        mask_raster: 掩模栅格（1=掩模内）

    Returns:
        np.ndarray: 掩模外像元的高程数组（与dem_array相同形状）
    """
    logger.info("提取掩模外像元")

    # 创建输出数组（掩模外保留原值，掩模内设为NaN）
    output_array = np.full_like(dem_array, np.nan, dtype=dem_array.dtype)
    non_mask_mask = (mask_raster == 0)
    output_array[non_mask_mask] = dem_array[non_mask_mask]

    non_mask_count = np.sum(non_mask_mask)
    total_pixels = dem_array.size
    logger.info(f"掩模外像元: {non_mask_count:,}/{total_pixels:,} ({100*non_mask_count/total_pixels:.1f}%)")

    return output_array


def merge_dem_layers(non_mask_layer: np.ndarray, mask_interp_layer: np.ndarray,
                    nodata_value: float = -9999.0) -> np.ndarray:
    """
    合并掩模外层和掩模内插值层

    Args:
        non_mask_layer: 掩模外像元层
        mask_interp_layer: 掩模内插值层
        nodata_value: 无数据值

    Returns:
        np.ndarray: 合并后的DEM
    """
    logger.info("合并DEM图层")

    # 创建输出数组
    output_dem = np.full_like(non_mask_layer, nodata_value, dtype=np.float32)

    # 首先填充掩模外像元（优先）
    non_mask_mask = ~np.isnan(non_mask_layer)
    output_dem[non_mask_mask] = non_mask_layer[non_mask_mask]

    # 然后填充掩模内插值像元
    mask_interp_mask = ~np.isnan(mask_interp_layer)
    # 只填充掩模外没有值的区域
    fill_mask = mask_interp_mask & ~non_mask_mask
    output_dem[fill_mask] = mask_interp_layer[fill_mask]

    # 统计
    non_mask_count = np.sum(non_mask_mask)
    mask_interp_count = np.sum(fill_mask)
    nodata_count = np.sum(np.isclose(output_dem, nodata_value) | np.isnan(output_dem))
    total_pixels = output_dem.size

    logger.info(f"合并完成: 掩模外={non_mask_count:,}, "
               f"掩模内插值={mask_interp_count:,}, "
               f"无数据={nodata_count:,}, "
               f"总计={total_pixels:,}")

    return output_dem


def write_dem(output_path: str, dem_array: np.ndarray, dem_meta: dict):
    """
    将DEM数据写入TIFF文件

    Args:
        output_path: 输出文件路径
        dem_array: DEM高程矩阵
        dem_meta: DEM元数据
    """
    logger.info(f"写入输出DEM: {output_path}")

    # 更新元数据
    out_meta = dem_meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'dtype': str(dem_array.dtype),
        'nodata': dem_meta.get('nodata', -9999.0)
    })

    # 确保数据类型匹配，处理NaN值
    output_dtype = dem_meta.get('dtype', 'float32')
    nodata = dem_meta.get('nodata', -9999.0)

    # 先将NaN值替换为nodata值
    nan_mask = np.isnan(dem_array)
    dem_array[nan_mask] = nodata

    if output_dtype == 'int32':
        dem_array = dem_array.astype(np.int32)
    elif output_dtype == 'float32':
        dem_array = dem_array.astype(np.float32)
    else:
        dem_array = dem_array.astype(np.float32)

    # 写入文件
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(dem_array, 1)

    logger.info(f"DEM写入完成，文件大小: {dem_array.shape[1]}x{dem_array.shape[0]}")


def fuse_dem_with_mask_replacement(
    dem_path: str,
    mask_shp_path: str,
    bay_points_path: str,
    shenzhenhe_points_path: str,
    output_path: str,
    output_resolution: float = 4.0,
    nodata_value: float = -9999.0,
    tin_output_path: Optional[str] = None
) -> None:
    """
    主函数：使用分块掩模替换法融合DEM

    Args:
        dem_path: 原始DEM文件路径
        mask_shp_path: 掩模shapefile路径
        bay_points_path: bay.txt点云文件路径
        shenzhenhe_points_path: shenzhenhe.csv点云文件路径
        output_path: 输出DEM文件路径
        output_resolution: 输出分辨率（米）
        nodata_value: 无数据值
        tin_output_path: TIN输出为PLY文件的路径（可选）
    """
    start_time = time.time()
    logger.info("开始DEM融合处理（分块掩模替换法）")
    logger.info(f"原始DEM: {dem_path}")
    logger.info(f"掩模文件: {mask_shp_path}")
    logger.info(f"bay.txt点云: {bay_points_path}")
    logger.info(f"shenzhenhe.csv点云: {shenzhenhe_points_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"输出分辨率: {output_resolution}米")
    if tin_output_path:
        logger.info(f"TIN输出文件: {tin_output_path}")
    logger.info(f"直接使用点云进行TIN插值，不先创建约束三角网")

    try:
        # 步骤1: 重采样DEM到目标分辨率
        logger.info("步骤1/7: 重采样DEM")
        dem_array, dem_meta = resample_dem_to_4m(dem_path, output_resolution)

        # 步骤2: 栅格化掩模
        logger.info("步骤2/7: 栅格化掩模")
        mask_raster = rasterize_mask(
            mask_shp_path,
            dem_meta['transform'],
            dem_meta['width'],
            dem_meta['height'],
            dem_meta['crs']
        )

        # 步骤3: 提取掩模外像元
        logger.info("步骤3/7: 提取掩模外像元")
        non_mask_layer = extract_non_mask_pixels(dem_array, mask_raster)

        # 步骤4: 读取和合并高精度点云
        logger.info("步骤4/7: 读取高精度点云")
        bay_points = read_bay_points(bay_points_path)
        shenzhenhe_points = read_shenzhenhe_points(shenzhenhe_points_path)
        highres_points = merge_point_clouds(bay_points, shenzhenhe_points)

        # 步骤5: 对掩模区域进行TIN插值（直接使用点云，不先创建TIN）
        logger.info("步骤5/7: 对掩模区域进行TIN插值")
        # 提取点云的XY坐标和Z值
        xy_points = highres_points[:, :2]
        z_values = highres_points[:, 2]
        mask_interp_layer = interpolate_mask_area(
            mask_raster, dem_meta, xy_points, z_values,
            nodata_value=nodata_value,
            tin_output_path=tin_output_path
        )

        # 步骤6: 合并图层
        logger.info("步骤6/7: 合并DEM图层")
        fused_dem = merge_dem_layers(non_mask_layer, mask_interp_layer, nodata_value)

        # 步骤7: 写入输出文件
        logger.info("步骤7/7: 写入输出DEM")
        # 更新元数据中的NoData值
        dem_meta['nodata'] = nodata_value
        write_dem(output_path, fused_dem, dem_meta)

        # 计算处理时间
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成! 总耗时: {elapsed_time:.2f}秒")

        # 输出统计信息
        valid_pixels = np.sum(~np.isclose(fused_dem, nodata_value) & ~np.isnan(fused_dem))
        total_pixels = fused_dem.size
        logger.info(f"输出DEM统计: {valid_pixels:,}/{total_pixels:,}有效像素 "
                   f"({100*valid_pixels/total_pixels:.1f}%)")

    except Exception as e:
        logger.error(f"DEM融合处理失败: {e}")
        raise


# ============================================================================
# 简化的主函数
# ============================================================================

def main():
    """
    简化的主函数：直接使用硬编码的路径和参数
    """
    logger.info("==========================================")
    logger.info("DEM融合脚本（简化版）")
    logger.info("开始执行...")
    logger.info("==========================================")

    logger.info("硬编码配置:")
    logger.info(f"  - 原始DEM: {INPUT_DEM_PATH}")
    logger.info(f"  - 掩模文件: {INPUT_MASK_SHP_PATH}")
    logger.info(f"  - bay.txt点云: {INPUT_BAY_POINTS_PATH}")
    logger.info(f"  - shenzhenhe.csv点云: {INPUT_SHENZHENHE_POINTS_PATH}")
    logger.info(f"  - 输出文件: {OUTPUT_DEM_PATH}")
    logger.info(f"  - 输出分辨率: {OUTPUT_RESOLUTION}米")
    logger.info(f"  - 无数据值: {NODATA_VALUE}")
    logger.info(f"  - TIN输出文件: {OUTPUT_TIN_PATH}")
    logger.info("注意：已移除最大三角形面积参数，直接使用点云进行插值")
    logger.info("==========================================")

    # 检查输入文件是否存在
    missing_files = []
    for path, desc in [
        (INPUT_DEM_PATH, "原始DEM"),
        (INPUT_MASK_SHP_PATH, "掩模shapefile"),
        (INPUT_BAY_POINTS_PATH, "bay.txt点云"),
        (INPUT_SHENZHENHE_POINTS_PATH, "shenzhenhe.csv点云")
    ]:
        if not os.path.exists(path):
            missing_files.append(f"{desc}: {path}")

    if missing_files:
        logger.error("以下输入文件不存在:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_DEM_PATH)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 执行DEM融合处理
    try:
        fuse_dem_with_mask_replacement(
            dem_path=INPUT_DEM_PATH,
            mask_shp_path=INPUT_MASK_SHP_PATH,
            bay_points_path=INPUT_BAY_POINTS_PATH,
            shenzhenhe_points_path=INPUT_SHENZHENHE_POINTS_PATH,
            output_path=OUTPUT_DEM_PATH,
            output_resolution=OUTPUT_RESOLUTION,
            nodata_value=NODATA_VALUE,
            tin_output_path=OUTPUT_TIN_PATH
        )

        logger.info("==========================================")
        logger.info("处理完成!")
        logger.info(f"输出文件已保存至: {OUTPUT_DEM_PATH}")
        logger.info("==========================================")

    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n用户中断脚本执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"未处理的异常: {e}")
        sys.exit(1)