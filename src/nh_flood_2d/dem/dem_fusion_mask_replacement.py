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
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from .tin_utils import create_constrained_tin, triangle_area
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
            dem_array[dem_array == nodata] = np.nan

        logger.info(f"重采样完成: {dst_width}x{dst_height} (原: {src.width}x{src.height})")

        # 返回元数据
        dem_meta = {
            'crs': src.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'dtype': str(dem_array.dtype),
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


def create_tin_from_points(points: np.ndarray, max_triangle_area: float = 8.0) -> Tuple[Delaunay, np.ndarray]:
    """
    从点云创建带面积约束的TIN

    Args:
        points: Nx3点云数组 (x, y, z)
        max_triangle_area: 最大三角形面积（平方米）

    Returns:
        Tuple[Delaunay, np.ndarray]: 三角网对象和点云数组
    """
    logger.info(f"从{len(points):,}个点创建TIN，最大三角形面积={max_triangle_area}m²")

    # 提取XY坐标

    xy_points = points[:, :2]

    # 创建带约束的三角网

    tin = create_constrained_tin(xy_points, max_triangle_area)

    # 验证TIN覆盖范围

    if len(tin.simplices) == 0:
        raise ValueError("无法创建有效的三角网")

    # 计算三角形统计信息

    areas = []
    for simplex in tin.simplices:
        tri_points = xy_points[simplex]

        area = triangle_area(tri_points[0], tri_points[1], tri_points[2])

        areas.append(area)

    logger.info(f"TIN统计: {len(tin.simplices):,}个三角形，"

               f"平均面积={np.mean(areas):.2f}m²，"

               f"最大面积={np.max(areas):.2f}m²，"

               f"最小面积={np.min(areas):.2f}m²")

    return tin, xy_points


def interpolate_mask_area(mask_raster: np.ndarray, dem_meta: dict,
                         tin: Delaunay, points: np.ndarray,
                         z_values: np.ndarray,
                         batch_size: int = 100000) -> np.ndarray:
    """
    对掩模区域进行TIN插值

    Args:
        mask_raster: 掩模栅格（1=掩模内）
        dem_meta: DEM元数据
        tin: 三角网对象
        points: 点云XY坐标
        z_values: 点云Z值
        batch_size: 批处理大小

    Returns:
        np.ndarray: 插值后的高程数组（与mask_raster相同形状）
    """
    logger.info(f"对掩模区域进行TIN插值，掩模像素={np.sum(mask_raster):,}")

    # 创建插值器
    from scipy.interpolate import LinearNDInterpolator
    interpolator = LinearNDInterpolator(points, z_values, fill_value=np.nan)

    # 获取掩模内像素的坐标
    mask_indices = np.where(mask_raster == 1)
    if len(mask_indices[0]) == 0:
        logger.warning("掩模内没有像素需要插值")
        return np.full_like(mask_raster, np.nan, dtype=np.float32)

    # 将像素索引转换为地理坐标
    rows, cols = mask_indices
    xs, ys = rasterio.transform.xy(dem_meta['transform'], rows, cols)
    mask_coords = np.column_stack([xs, ys])

    # 分批插值以避免内存问题
    num_points = len(mask_coords)
    num_batches = (num_points + batch_size - 1) // batch_size

    interpolated_values = np.full(num_points, np.nan, dtype=np.float32)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)

        batch_coords = mask_coords[start_idx:end_idx]
        batch_interp = interpolator(batch_coords)

        interpolated_values[start_idx:end_idx] = batch_interp

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            valid_count = np.sum(~np.isnan(batch_interp))
            logger.info(f"  批处理 {batch_idx+1}/{num_batches}: {start_idx:,}-{end_idx:,}, "
                       f"有效插值={valid_count:,}")

    # 创建输出数组
    output_array = np.full_like(mask_raster, np.nan, dtype=np.float32)
    output_array[mask_raster == 1] = interpolated_values

    valid_total = np.sum(~np.isnan(output_array))
    logger.info(f"插值完成: 有效值={valid_total:,}/{np.sum(mask_raster):,}个掩模像素")

    return output_array


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

    # 确保数据类型匹配
    output_dtype = dem_meta.get('dtype', 'float32')
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
    max_triangle_area: float = 8.0,
    nodata_value: float = -9999.0
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
        max_triangle_area: 最大三角形面积（平方米）
        nodata_value: 无数据值
    """
    start_time = time.time()
    logger.info("开始DEM融合处理（分块掩模替换法）")
    logger.info(f"原始DEM: {dem_path}")
    logger.info(f"掩模文件: {mask_shp_path}")
    logger.info(f"bay.txt点云: {bay_points_path}")
    logger.info(f"shenzhenhe.csv点云: {shenzhenhe_points_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"输出分辨率: {output_resolution}米")
    logger.info(f"最大三角形面积: {max_triangle_area}平方米")

    try:
        # 步骤1: 重采样DEM到目标分辨率
        logger.info("步骤1/8: 重采样DEM")
        dem_array, dem_meta = resample_dem_to_4m(dem_path, output_resolution)

        # 步骤2: 栅格化掩模
        logger.info("步骤2/8: 栅格化掩模")
        mask_raster = rasterize_mask(
            mask_shp_path,
            dem_meta['transform'],
            dem_meta['width'],
            dem_meta['height'],
            dem_meta['crs']
        )

        # 步骤3: 提取掩模外像元
        logger.info("步骤3/8: 提取掩模外像元")
        non_mask_layer = extract_non_mask_pixels(dem_array, mask_raster)

        # 步骤4: 读取和合并高精度点云
        logger.info("步骤4/8: 读取高精度点云")
        bay_points = read_bay_points(bay_points_path)
        shenzhenhe_points = read_shenzhenhe_points(shenzhenhe_points_path)
        highres_points = merge_point_clouds(bay_points, shenzhenhe_points)

        # 步骤5: 创建带面积约束的TIN
        logger.info("步骤5/8: 创建带面积约束的TIN")
        tin, xy_points = create_tin_from_points(highres_points, max_triangle_area)

        # 步骤6: 对掩模区域进行TIN插值
        logger.info("步骤6/8: 对掩模区域进行TIN插值")
        mask_interp_layer = interpolate_mask_area(
            mask_raster, dem_meta, tin, xy_points, highres_points[:, 2]
        )

        # 步骤7: 合并图层
        logger.info("步骤7/8: 合并DEM图层")
        fused_dem = merge_dem_layers(non_mask_layer, mask_interp_layer, nodata_value)

        # 步骤8: 写入输出文件
        logger.info("步骤8/8: 写入输出DEM")
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
    parser.add_argument('--max-triangle-area', type=float, default=8.0, help='最大三角形面积（平方米）')
    parser.add_argument('--nodata', type=float, default=-9999.0, help='无数据值')

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
            max_triangle_area=args.max_triangle_area,
            nodata_value=args.nodata
        )

    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()