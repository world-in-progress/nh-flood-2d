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