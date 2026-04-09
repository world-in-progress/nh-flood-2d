#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本 - 分块掩模替换法
作者: 高级GIS工程师
日期: 2026-04-01

核心算法: 分块掩模替换法（带三角形面积约束）
1. 读取原始DEM并重采样到4米分辨率（D1）
2. 读取掩模shapefile，栅格化到D1网格
3. 提取掩模外像元 → 保留到最终输出
4. 读取并合并两个高精度点云文件
5. 构建带面积约束的Delaunay三角网（最大三角形面积8m²）
6. 对掩模内每个像元：TIN插值获取高程
7. 合并掩模外保留像元 + 掩模内插值像元
8. 输出4米分辨率DEM
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.features
import pandas as pd
import pygmt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
try:
    from .tin_utils import create_constrained_tin, triangle_area
except ImportError:
    # 尝试绝对导入
    from tin_utils import create_constrained_tin, triangle_area
import geopandas as gpd
import logging
import sys
import time
import os
from typing import Tuple, Optional, List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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




def interpolate_mask_area(
    mask_raster,      # 原掩膜栅格 (xarray.DataArray 或 numpy.ndarray)
    dem_meta,         # 原DEM的元数据（可保留备用）
    xy_points,        # 采样点坐标 (DataFrame 或 array，列为 x, y)
    z_values,         # 采样点的高程值 (array 或 Series)
    nodata_value,     # NoData 值
    sea_output_path   # 输出路径（这里改为 surface 输出路径）
):
    """
    使用 pygmt.surface 对指定掩膜区域内的采样点进行高精度插值，
    并生成与原 mask_raster 像素完全对齐的水下 DEM。

    参数说明：
        mask_raster: xarray.DataArray 或 numpy.ndarray，掩膜栅格（值为1的区域为需要插值的区域）
        xy_points:   采样点的 (x, y) 坐标
        z_values:    对应的高程值
        nodata_value: NoData 值（通常 -9999 或 np.nan）
        sea_output_path: 输出 TIFF 文件路径
    """
    logger.info("使用 pygmt.surface 进行插值...")

    # 1. 合并采样点为 pygmt.surface 需要的格式 (x, y, z)
    if isinstance(xy_points, pd.DataFrame):
        points_df = pd.DataFrame({
            "x": xy_points.iloc[:, 0] if len(xy_points.columns) >= 2 else xy_points["x"],
            "y": xy_points.iloc[:, 1] if len(xy_points.columns) >= 2 else xy_points["y"],
            "z": z_values
        })
    else:
        points_df = pd.DataFrame({"x": xy_points[:, 0], "y": xy_points[:, 1], "z": z_values})

    # 2. 提取原栅格的参数（确保完美对齐）
    # 首先检查 mask_raster 的类型
    if hasattr(mask_raster, 'x') and hasattr(mask_raster, 'y'):
        # 如果 mask_raster 是 xarray.DataArray
        region = [
            mask_raster.x.min().item(),
            mask_raster.x.max().item(),
            mask_raster.y.min().item(),
            mask_raster.y.max().item()
        ]

        # spacing: 与原栅格完全一致的分辨率
        dx = abs(mask_raster.x.diff("x").values[0])
        dy = abs(mask_raster.y.diff("y").values[0])
        spacing = f"{dx}/{dy}"

        # 获取坐标参考系和变换
        if hasattr(mask_raster, 'rio') and hasattr(mask_raster.rio, 'crs'):
            crs = mask_raster.rio.crs
            transform = mask_raster.rio.transform()
        else:
            crs = None
            transform = None

        # 获取 mask 数组
        mask = mask_raster.values.astype(bool)

    else:
        # 如果 mask_raster 是 numpy.ndarray，使用 dem_meta 获取参数
        logger.info("mask_raster 是 numpy.ndarray，使用 dem_meta 提取参数")

        # 从 dem_meta 获取变换参数
        target_transform = dem_meta['transform']
        target_width = dem_meta['width']
        target_height = dem_meta['height']

        # 计算边界框
        left, bottom = target_transform * (0, target_height)
        right, top = target_transform * (target_width, 0)
        region = [left, right, bottom, top]

        # 获取分辨率
        dx = abs(target_transform.a)
        dy = abs(target_transform.e)
        spacing = f"{dx}/{dy}"

        # 从 dem_meta 获取 CRS
        crs = dem_meta.get('crs')
        transform = target_transform

        # mask 数组
        mask = mask_raster.astype(bool)

    logger.info(f"插值区域: {region}")
    logger.info(f"分辨率: {spacing}")

    # 3. 核心插值：使用 pygmt.surface 生成水下网格
    logger.info("正在进行 pygmt.surface 插值...")

    try:
        underwater_grid = pygmt.surface(
            data=points_df,           # 输入点数据 (DataFrame 或文件路径)
            region=region,
            spacing=spacing,
            tension=0.25,             # 张力参数，推荐 0.25~0.35（地形/水深常用）
            # 可选参数：
            # max_radius="0c",        # 如果采样点稀疏，可限制外推距离（单位：网格单元）
            # verbose=True
        )
    except Exception as e:
        logger.error(f"pygmt.surface 插值失败: {e}")
        raise

    # 4. 应用掩膜（只保留 mask_raster 中有效区域，其余设为 NoData）
    # 创建最终输出数组
    final_grid = underwater_grid.copy(deep=True)
    final_grid.values[~mask] = nodata_value   # 或 np.nan，如果你希望用 NaN

    # 5. 如果指定了输出路径，保存为 GeoTIFF（保留原坐标信息）
    if sea_output_path:
        logger.info(f"保存海底DEM到: {sea_output_path}")

        # 确保输出目录存在
        output_dir = os.path.dirname(sea_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            # 写入坐标参考系和变换
            if crs:
                final_grid.rio.write_crs(crs, inplace=True)
            if transform:
                final_grid.rio.write_transform(transform, inplace=True)

            final_grid.rio.to_raster(sea_output_path, nodata=nodata_value)
            logger.info(f"海底DEM已保存至: {sea_output_path}")

        except Exception as e:
            logger.error(f"保存海底DEM失败: {e}")
            # 继续处理，不中断流程

    # 6. 返回插值结果（作为 numpy 数组）
    # 注意：为了与 merge_dem_layers 兼容，我们返回 numpy 数组
    result_array = final_grid.values

    # 确保结果形状与 mask_raster 一致
    if result_array.shape != mask.shape:
        logger.warning(f"插值结果形状不匹配: {result_array.shape} != {mask.shape}")
        # 如果形状不匹配，调整结果形状
        if result_array.shape[0] == mask.shape[0] and result_array.shape[1] == mask.shape[1]:
            # 形状相同，直接返回
            pass
        else:
            # 形状不同，尝试调整
            logger.warning("无法调整形状，返回全nodata数组")
            result_array = np.full_like(mask, nodata_value, dtype=np.float32)

    logger.info(f"插值完成，返回数组形状: {result_array.shape}")

    return result_array
    """
    对掩模区域进行TIN插值（改进版：生成高质量三角形并确保完全覆盖）

    步骤：
    1. 生成高质量TIN（添加边界点和内部点，避免细长三角形）
    2. 检查TIN是否完全覆盖掩模区域，验证覆盖
    3. 使用TIN插值所有掩模栅格

    Args:
        mask_raster: 掩模栅格（1=掩模内）
        dem_meta: DEM元数据
        points: 点云XY坐标
        z_values: 点云Z值
        batch_size: 批处理大小
        nodata_value: 无数据值
        tin_output_path: TIN输出为PLY文件的路径（可选）

    Returns:
        np.ndarray: 插值后的高程数组（与mask_raster相同形状）
    """
    logger.info(f"开始TIN插值（改进版）：点云数量={len(points):,}，掩模像素={np.sum(mask_raster):,}")

    # 检查是否有足够的点进行插值
    if len(points) < 3:
        logger.warning("点云数量不足，无法进行三角剖分插值")
        return np.full_like(mask_raster, nodata_value, dtype=np.float32)

    try:
        # 获取原始DEM的变换和尺寸
        target_transform = dem_meta['transform']
        target_width = dem_meta['width']
        target_height = dem_meta['height']
        resolution = dem_meta.get('resolution', 4.0)

        logger.info(f"目标网格尺寸: {target_width}x{target_height}，分辨率: {resolution}米")

        # 获取掩模内像素的坐标
        mask_indices = np.where(mask_raster == 1)
        if len(mask_indices[0]) == 0:
            logger.warning("掩模内没有像素需要插值")
            return np.full_like(mask_raster, nodata_value, dtype=np.float32)

        rows, cols = mask_indices
        xs, ys = rasterio.transform.xy(target_transform, rows, cols)
        mask_coords = np.column_stack([xs, ys])

        logger.info(f"需要插值的掩模像素坐标数量: {len(mask_coords):,}")

        # 1. 生成高质量TIN（添加边界点和内部点）
        logger.info("步骤1/4: 生成高质量TIN（避免细长三角形）")
        tri, enhanced_points = create_quality_tin(points, mask_coords, resolution)

        # 为增强点云创建对应的Z值（原始点用真实Z值，新增点用插值估计）
        # 首先创建一个临时的插值器用于估算新增点的Z值
        temp_interpolator = LinearNDInterpolator(points, z_values, fill_value=np.nan)

        # 为增强点云创建Z值数组
        enhanced_z_values = np.empty(len(enhanced_points))

        # 为原始点分配真实的Z值
        # 创建一个映射：查找原始点在增强点云中的位置
        enhanced_z_values[:] = np.nan

        # 对于每个原始点，找到在增强点云中的对应位置
        for i in range(len(points)):
            # 查找距离最近的点（应该就是同一个点）
            dists = np.linalg.norm(enhanced_points - points[i], axis=1)
            closest_idx = np.argmin(dists)
            enhanced_z_values[closest_idx] = z_values[i]

        # 为新增点插值Z值
        nan_mask = np.isnan(enhanced_z_values)
        if np.any(nan_mask):
            estimated_z = temp_interpolator(enhanced_points[nan_mask])
            enhanced_z_values[nan_mask] = estimated_z

        # 2. 检查TIN是否完全覆盖掩模区域
        logger.info("步骤2/4: 验证TIN覆盖情况")
        is_covered, uncovered_coords = check_tin_coverage(tri, mask_coords)

        if not is_covered:
            logger.warning(f"TIN未完全覆盖掩模区域，有{len(uncovered_coords):,}个点未覆盖")
            logger.info("尝试添加额外点来改善覆盖...")

            # 为未覆盖的点找到最近的点
            for i, coord in enumerate(uncovered_coords[:10]):  # 只添加前10个（避免太多点）
                # 计算坐标到所有增强点的距离
                dists = np.linalg.norm(enhanced_points - coord, axis=1)

                # 找到最近的点
                min_dist_idx = np.argmin(dists)
                nearest_point = enhanced_points[min_dist_idx]

                # 在最近点和未覆盖点之间添加一个新点
                new_point = (coord + nearest_point) / 2

                # 添加到增强点云
                enhanced_points = np.vstack([enhanced_points, new_point])

                # 为这个新点插值Z值
                new_z = temp_interpolator([new_point])[0]
                enhanced_z_values = np.append(enhanced_z_values, new_z)

            # 重新生成TIN
            logger.info("重新生成TIN...")
            tri = Delaunay(enhanced_points)

            # 再次检查覆盖
            is_covered, uncovered_coords = check_tin_coverage(tri, mask_coords)

            if not is_covered:
                logger.error(f"即使添加额外点，仍有{len(uncovered_coords):,}个点未覆盖")
                logger.warning("将继续处理，但部分区域可能无法正确插值")

        # 3. 输出TIN为PLY文件（如果指定了路径）
        if tin_output_path:
            logger.info(f"步骤3/4: 将TIN输出为PLY文件: {tin_output_path}")
            combined_points = np.column_stack([enhanced_points, enhanced_z_values])
            save_tin_as_ply(tri, combined_points, tin_output_path)

        # 4. 使用TIN插值所有掩模栅格
        logger.info("步骤4/4: 使用高质量TIN插值所有掩模栅格")

        # 使用增强点云创建最终的插值器
        final_interpolator = LinearNDInterpolator(enhanced_points, enhanced_z_values, fill_value=nodata_value)

        # 分批处理以防止内存溢出
        num_batches = int(np.ceil(len(mask_coords) / batch_size))
        interpolated_values = np.full(len(mask_coords), nodata_value)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(mask_coords))

            batch_coords = mask_coords[start_idx:end_idx]
            interpolated_values[start_idx:end_idx] = final_interpolator(batch_coords)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                # 统计有效插值
                valid_count = np.sum(interpolated_values[start_idx:end_idx] != nodata_value)
                logger.info(f"插值进度: {batch_idx + 1}/{num_batches}批次 "
                          f"({end_idx:,}/{len(mask_coords):,}点), 有效插值={valid_count:,}")

        # 检查插值结果
        failed_count = np.sum(interpolated_values == nodata_value)
        if failed_count > 0:
            logger.warning(f"有{failed_count:,}个像素无法插值，将保持为nodata值")
        else:
            logger.info("所有掩模像素都成功插值")

        # 创建输出数组
        output_array = np.full_like(mask_raster, nodata_value, dtype=np.float32)
        output_array[mask_raster == 1] = interpolated_values

        # 处理NaN值
        nan_mask = np.isnan(output_array)
        output_array[nan_mask] = nodata_value

        # 统计有效值
        valid_total = np.sum(output_array != nodata_value)
        logger.info(f"插值完成: 有效值={valid_total:,}/{np.sum(mask_raster):,}个掩模像素")

        logger.info(f"最终TIN统计: {len(tri.simplices):,}个三角形, {len(enhanced_points):,}个顶点")

        # 报告三角形质量
        if len(tri.simplices) > 0:
            # 简单计算三角形边长统计
            triangles = enhanced_points[tri.simplices]
            edge_lengths = []
            for tri_vertices in triangles:
                edges = [
                    np.linalg.norm(tri_vertices[1] - tri_vertices[0]),
                    np.linalg.norm(tri_vertices[2] - tri_vertices[1]),
                    np.linalg.norm(tri_vertices[0] - tri_vertices[2])
                ]
                edge_lengths.extend(edges)

            if edge_lengths:
                edge_lengths = np.array(edge_lengths)
                logger.info(f"三角形边长统计: 平均={np.mean(edge_lengths):.2f}米, 最小={np.min(edge_lengths):.2f}米, 最大={np.max(edge_lengths):.2f}米")

        return output_array

    except Exception as e:
        logger.error(f"插值过程中发生错误: {e}")
        raise
    """
    对掩模区域进行TIN插值（按照用户要求的新方法）

    步骤：
    1. 依据点云生成Delaunay三角网（TIN）
    2. 确保生成的TIN覆盖所有掩模栅格（1=掩模内）
    3. 将TIN输出成PLY文件（可选）
    4. 使用TIN插值所有的掩模栅格

    Args:
        mask_raster: 掩模栅格（1=掩模内）
        dem_meta: DEM元数据
        points: 点云XY坐标
        z_values: 点云Z值
        batch_size: 批处理大小
        nodata_value: 无数据值
        tin_output_path: TIN输出为PLY文件的路径（可选）

    Returns:
        np.ndarray: 插值后的高程数组（与mask_raster相同形状）
    """
    logger.info(f"开始TIN插值：点云数量={len(points):,}，掩模像素={np.sum(mask_raster):,}")

    # 检查是否有足够的点进行插值
    if len(points) < 3:
        logger.warning("点云数量不足，无法进行三角剖分插值")
        return np.full_like(mask_raster, nodata_value, dtype=np.float32)

    try:
        # 获取原始DEM的变换和尺寸
        target_transform = dem_meta['transform']
        target_width = dem_meta['width']
        target_height = dem_meta['height']
        resolution = dem_meta.get('resolution', 4.0)

        logger.info(f"目标网格尺寸: {target_width}x{target_height}，分辨率: {resolution}米")

        # 1. 生成Delaunay三角网
        logger.info("步骤1/4: 生成Delaunay三角网（TIN）")
        combined_points = np.column_stack([points, z_values])  # 组合成完整的点云数据

        # 创建Delaunay三角剖分
        logger.info("创建Delaunay三角剖分...")
        tri = Delaunay(points)

        logger.info(f"TIN生成完成: {len(tri.simplices):,}个三角形，{len(combined_points):,}个顶点")

        # 2. 检查TIN是否覆盖所有掩模栅格
        logger.info("步骤2/4: 检查TIN是否覆盖所有掩模栅格")

        # 获取掩模内像素的坐标
        mask_indices = np.where(mask_raster == 1)
        if len(mask_indices[0]) == 0:
            logger.warning("掩模内没有像素需要插值")
            return np.full_like(mask_raster, nodata_value, dtype=np.float32)

        rows, cols = mask_indices
        xs, ys = rasterio.transform.xy(target_transform, rows, cols)
        mask_coords = np.column_stack([xs, ys])

        logger.info(f"需要插值的掩模像素坐标数量: {len(mask_coords):,}")

        # 检查点云是否覆盖掩模区域
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        mask_coords_min = mask_coords.min(axis=0)
        mask_coords_max = mask_coords.max(axis=0)

        logger.info(f"点云范围: X[{points_min[0]:.1f}, {points_max[0]:.1f}], Y[{points_min[1]:.1f}, {points_max[1]:.1f}]")
        logger.info(f"掩模范围: X[{mask_coords_min[0]:.1f}, {mask_coords_max[0]:.1f}], Y[{mask_coords_min[1]:.1f}, {mask_coords_max[1]:.1f}]")

        # 检查点云范围是否包含掩模范围
        coverage_x = (mask_coords_min[0] >= points_min[0] and mask_coords_max[0] <= points_max[0])
        coverage_y = (mask_coords_min[1] >= points_min[1] and mask_coords_max[1] <= points_max[1])

        if coverage_x and coverage_y:
            logger.info("点云范围完全覆盖掩模区域")
        else:
            logger.warning("点云范围不完全覆盖掩模区域，可能需要额外处理")
            logger.info(f"X方向覆盖: {coverage_x}, Y方向覆盖: {coverage_y}")

        # 3. 输出TIN为PLY文件（如果指定了路径）
        if tin_output_path:
            logger.info(f"步骤3/4: 将TIN输出为PLY文件: {tin_output_path}")
            save_tin_as_ply(tri, combined_points, tin_output_path)

        # 4. 使用TIN插值所有掩模栅格
        logger.info("步骤4/4: 使用TIN插值所有掩模栅格")

        # 使用LinearNDInterpolator进行基于TIN的插值
        logger.info("使用Delaunay三角网进行线性插值...")
        interpolator = LinearNDInterpolator(points, z_values, fill_value=nodata_value)

        # 分批处理以防止内存溢出
        num_batches = int(np.ceil(len(mask_coords) / batch_size))
        interpolated_values = np.full(len(mask_coords), nodata_value)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(mask_coords))

            batch_coords = mask_coords[start_idx:end_idx]
            interpolated_values[start_idx:end_idx] = interpolator(batch_coords)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                # 统计有效插值
                valid_count = np.sum(interpolated_values[start_idx:end_idx] != nodata_value)
                logger.info(f"插值进度: {batch_idx + 1}/{num_batches}批次 "
                          f"({end_idx:,}/{len(mask_coords):,}点), 有效插值={valid_count:,}")

        # 检查插值结果
        failed_count = np.sum(interpolated_values == nodata_value)
        if failed_count > 0:
            logger.warning(f"有{failed_count:,}个像素无法插值（可能位于TIN凸包外），将保持为nodata值")
        else:
            logger.info("所有掩模像素都成功插值")

        # 创建输出数组
        output_array = np.full_like(mask_raster, nodata_value, dtype=np.float32)
        output_array[mask_raster == 1] = interpolated_values

        # 处理NaN值
        nan_mask = np.isnan(output_array)
        output_array[nan_mask] = nodata_value

        # 统计有效值
        valid_total = np.sum(output_array != nodata_value)
        logger.info(f"插值完成: 有效值={valid_total:,}/{np.sum(mask_raster):,}个掩模像素")

        return output_array

    except Exception as e:
        logger.error(f"插值过程中发生错误: {e}")
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
    tin_output_path: Optional[str] = None,
    sea_output_path: Optional[str] = None
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
        sea_output_path: 海底DEM输出路径（可选）
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
    # 注意：已移除最大三角形面积参数，直接使用点云进行插值

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
            sea_output_path=sea_output_path
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


def main():
    """主函数：命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='DEM融合脚本（分块掩模替换法）')
    parser.add_argument('--dem', required=True, help='原始DEM文件路径')
    parser.add_argument('--mask', required=True, help='掩模shapefile路径')
    parser.add_argument('--bay-points', required=True, help='bay.txt点云文件路径')
    parser.add_argument('--shenzhenhe-points', required=True, help='shenzhenhe.csv点云文件路径')
    parser.add_argument('--output', required=True, help='输出DEM文件路径')
    parser.add_argument('--resolution', type=float, default=4.0, help='输出分辨率（米）')
    parser.add_argument('--nodata', type=float, default=-9999.0, help='无数据值')
    parser.add_argument('--tin-output', help='TIN输出为PLY文件的路径（可选）')
    parser.add_argument('--sea-output', help='海底DEM输出路径（可选）')

    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        fuse_dem_with_mask_replacement(
            dem_path=args.dem,
            mask_shp_path=args.mask,
            bay_points_path=args.bay_points,
            shenzhenhe_points_path=args.shenzhenhe_points,
            output_path=args.output,
            output_resolution=args.resolution,
            nodata_value=args.nodata,
            tin_output_path=args.tin_output,
            sea_output_path=args.sea_output
        )

    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()