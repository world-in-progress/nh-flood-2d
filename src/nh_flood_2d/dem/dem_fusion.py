#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本 - 将低精度区域DEM与高精度局部高程点云融合
作者: 高级GIS工程师
日期: 2026-03-31 (修改版)

核心算法: 点云替换与统一插值法
1. 读取区域DEM并展平为点云
2. 读取掩模shapefile作为替换区域边界
3. 从DEM点云中剔除掩模内点（挖洞）
4. 读取高精度点云，筛选落在掩模内的点
5. 合并剩余DEM点云与掩模内高精度点云
6. 构建TIN并重采样到固定4m分辨率网格
7. 输出融合后的DEM
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Point, MultiPoint, Polygon
import geopandas as gpd
import logging
import sys
import time
from typing import Tuple, Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def read_xyz_points(file_path: str, skip_rows: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    支持格式：
    1. 纯"x;y;z"格式（分号分隔）- gcd.txt格式
    2. 纯"x,y,z"格式（逗号分隔）- shenzhenhe.txt格式
    3. "序号 x;y;z" 或 "序号 x,y,z"

    Args:
        file_path: 点云文件路径
        skip_rows: 跳过的行数（用于跳过表头）

    Returns:
        Tuple[x_coords, y_coords, z_coords]: 三个一维数组
    """
    logger.info(f"读取点云文件: {file_path}")

    # 首先探测文件格式
    with open(file_path, 'r') as f:
        sample_lines = []
        for i, line in enumerate(f):
            if i >= 20 + skip_rows:  # 读取20行进行格式探测
                break
            if i >= skip_rows:
                sample_lines.append(line.strip())

    # 分析格式
    line_formats = []
    for line in sample_lines:
        if ';' in line and '\t' not in line and line.count(';') == 2:
            # 格式: x;y;z (gcd.txt格式)
            line_formats.append('pure_semicolon')
        elif ';' in line and '\t' in line and line.count(';') == 2:
            # 格式: 序号\tx;y;z
            line_formats.append('index_semicolon')
        elif ',' in line and '\t' not in line and line.count(',') == 2:
            # 格式: x,y,z
            line_formats.append('pure_comma')
        elif ',' in line and '\t' in line and line.count(',') == 2:
            # 格式: 序号\tx,y,z
            line_formats.append('index_comma')
        else:
            line_formats.append('unknown')

    # 确定主要格式
    format_counts = {}
    for fmt in line_formats:
        if fmt != 'unknown':
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

    if format_counts:
        detected_format = max(format_counts.items(), key=lambda x: x[1])[0]
    else:
        detected_format = 'unknown'

    logger.info(f"检测到文件格式: {detected_format}")
    logger.info(f"样本行格式分布: {format_counts}")

    # 读取数据
    x_list, y_list, z_list = [], [], []
    point_count = 0
    error_count = 0

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < skip_rows:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                if detected_format == 'pure_semicolon':
                    # 格式: x;y;z (gcd.txt)
                    coords = line.split(';')
                    if len(coords) == 3:
                        x, y, z = coords
                    else:
                        continue

                elif detected_format == 'index_semicolon':
                    # 格式: 序号\tx;y;z
                    parts = line.split('\t')
                    if len(parts) == 2:
                        _, coord_str = parts
                        coords = coord_str.split(';')
                        if len(coords) == 3:
                            x, y, z = coords
                        else:
                            continue
                    else:
                        continue

                elif detected_format == 'pure_comma':
                    # 格式: x,y,z
                    coords = line.split(',')
                    if len(coords) == 3:
                        x, y, z = coords
                    else:
                        continue

                elif detected_format == 'index_comma':
                    # 格式: 序号\tx,y,z
                    parts = line.split('\t')
                    if len(parts) == 2:
                        _, coord_str = parts
                        coords = coord_str.split(',')
                        if len(coords) == 3:
                            x, y, z = coords
                        else:
                            continue
                    else:
                        continue

                else:
                    # 尝试通用解析
                    # 移除可能的序号
                    parts = line.split()
                    if len(parts) >= 3:
                        # 尝试最后三个值
                        for j in range(len(parts) - 2):
                            try:
                                x = parts[j]
                                y = parts[j+1]
                                z = parts[j+2]
                                # 检查是否为数字
                                float(x); float(y); float(z)
                                break
                            except ValueError:
                                continue
                        else:
                            continue
                    else:
                        continue

                # 转换为浮点数
                x_val = float(x)
                y_val = float(y)
                z_val = float(z)

                x_list.append(x_val)
                y_list.append(y_val)
                z_list.append(z_val)
                point_count += 1

                # 进度显示
                if point_count % 1000000 == 0:
                    logger.info(f"已读取 {point_count:,} 个点...")

            except (ValueError, IndexError) as e:
                error_count += 1
                if error_count <= 5:  # 只记录前几个错误
                    logger.debug(f"解析错误第 {error_count} 行: {line[:50]}... - {e}")
                continue

    logger.info(f"成功读取 {point_count:,} 个点，跳过 {error_count} 个错误行")

    if point_count == 0:
        raise ValueError("未成功读取任何点云数据")

    return np.array(x_list), np.array(y_list), np.array(z_list)


def extract_dem_points(dem_path: str, sample_fraction: float = 1.0) -> Tuple[np.ndarray, dict]:
    """
    从DEM文件中提取点云数据

    Args:
        dem_path: DEM文件路径
        sample_fraction: 采样比例 (0-1]，用于减少点云数量

    Returns:
        Tuple[points_xyz, dem_meta]: 点云Nx3数组和DEM元数据
    """
    logger.info(f"读取DEM文件: {dem_path}")

    with rasterio.open(dem_path) as src:
        # 获取DEM元数据
        dem_meta = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'dtype': src.dtypes[0],
            'nodata': src.nodatavals[0],
            'bounds': src.bounds
        }

        # 读取DEM数据
        dem_array = src.read(1)
        nodata = src.nodatavals[0]

        # 创建坐标网格
        rows, cols = np.indices(dem_array.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows.flatten(), cols.flatten())
        zs = dem_array.flatten()

        # 处理NoData值
        if nodata is not None:
            valid_mask = zs != nodata
        else:
            valid_mask = np.isfinite(zs)

        # 采样（如果需要）
        if sample_fraction < 1.0:
            valid_indices = np.where(valid_mask)[0]
            sample_size = int(len(valid_indices) * sample_fraction)
            if sample_size < len(valid_indices):
                sampled_indices = np.random.choice(valid_indices, sample_size, replace=False)
                valid_mask[:] = False
                valid_mask[sampled_indices] = True

        # 提取有效点
        valid_xs = np.array(xs)[valid_mask]
        valid_ys = np.array(ys)[valid_mask]
        valid_zs = np.array(zs)[valid_mask]

        # 合并为Nx3数组
        points_xyz = np.column_stack([valid_xs, valid_ys, valid_zs])

        logger.info(f"从DEM提取 {len(points_xyz):,} 个有效点")
        logger.info(f"DEM范围: X[{valid_xs.min():.1f}, {valid_xs.max():.1f}], "
                   f"Y[{valid_ys.min():.1f}, {valid_ys.max():.1f}], "
                   f"Z[{valid_zs.min():.1f}, {valid_zs.max():.1f}]")

    return points_xyz, dem_meta


def load_mask_polygon(mask_shp_path: str) -> Polygon:
    """
    从shapefile加载掩模多边形

    Args:
        mask_shp_path: shapefile文件路径

    Returns:
        Polygon: 掩模多边形
    """
    logger.info(f"加载掩模shapefile: {mask_shp_path}")

    try:
        # 使用geopandas读取shapefile
        gdf = gpd.read_file(mask_shp_path)

        if len(gdf) == 0:
            raise ValueError("shapefile中没有要素")

        # 获取所有几何图形
        geometries = gdf.geometry.tolist()

        # 合并所有多边形为一个
        mask_polygon = geometries[0]
        for geom in geometries[1:]:
            mask_polygon = mask_polygon.union(geom)

        logger.info(f"掩模面积: {mask_polygon.area:.2f} 平方米")
        logger.info(f"掩模边界: {mask_polygon.bounds}")

        return mask_polygon

    except Exception as e:
        logger.error(f"加载shapefile失败: {e}")
        raise


def filter_points_inside_polygon(points_xyz: np.ndarray, polygon: Polygon) -> np.ndarray:
    """
    筛选落在多边形内的点云

    Args:
        points_xyz: Nx3点云数组
        polygon: Shapely多边形

    Returns:
        np.ndarray: 多边形内的点云
    """
    logger.info("筛选多边形内点云")

    # 创建边界框用于快速预筛选
    minx, miny, maxx, maxy = polygon.bounds

    # 快速边界框筛选
    x_coords = points_xyz[:, 0]
    y_coords = points_xyz[:, 1]

    bbox_mask = (x_coords >= minx) & (x_coords <= maxx) & (y_coords >= miny) & (y_coords <= maxy)

    # 对于边界框内的点进行精确多边形判断
    inside_indices = np.where(bbox_mask)[0]
    if len(inside_indices) == 0:
        logger.info("边界框内没有点")
        return np.array([])

    inside_points = points_xyz[inside_indices]

    # 向量化多边形判断（使用列表推导式，可以考虑优化）
    points_in_poly = np.array([
        polygon.contains(Point(x, y))
        for x, y in zip(inside_points[:, 0], inside_points[:, 1])
    ])

    # 提取多边形内的点
    final_indices = inside_indices[points_in_poly]
    filtered_points = points_xyz[final_indices]

    logger.info(f"筛选前: {len(points_xyz):,} 个点")
    logger.info(f"筛选后: {len(filtered_points):,} 个点（多边形内）")

    return filtered_points


def filter_points_by_polygon(points_xyz: np.ndarray, polygon: Polygon) -> np.ndarray:
    """
    使用多边形过滤点云（移除多边形内的点，保留多边形外的点）

    Args:
        points_xyz: Nx3点云数组
        polygon: Shapely多边形

    Returns:
        np.ndarray: 过滤后的点云（多边形外）
    """
    logger.info("过滤点云（挖洞操作）")

    # 创建边界框用于快速预筛选
    minx, miny, maxx, maxy = polygon.bounds

    # 快速边界框筛选
    x_coords = points_xyz[:, 0]
    y_coords = points_xyz[:, 1]

    bbox_mask = (x_coords < minx) | (x_coords > maxx) | (y_coords < miny) | (y_coords > maxy)

    # 对于边界框内的点进行精确多边形判断
    inside_mask = ~bbox_mask
    if np.any(inside_mask):
        inside_indices = np.where(inside_mask)[0]
        inside_points = points_xyz[inside_mask]

        # 向量化多边形判断
        points_in_poly = np.array([
            polygon.contains(Point(x, y))
            for x, y in zip(inside_points[:, 0], inside_points[:, 1])
        ])

        # 更新mask：保留不在多边形内的点
        inside_mask[inside_mask] = ~points_in_poly

    # 合并mask
    final_mask = bbox_mask | inside_mask

    filtered_points = points_xyz[final_mask]

    logger.info(f"过滤前: {len(points_xyz):,} 个点")
    logger.info(f"过滤后: {len(filtered_points):,} 个点")
    logger.info(f"移除: {len(points_xyz) - len(filtered_points):,} 个点（挖洞）")

    return filtered_points


def interpolate_to_grid(points_xyz: np.ndarray, dem_meta: dict,
                       nodata_value: float = -9999.0) -> np.ndarray:
    """
    使用TIN插值将点云重采样到固定4m分辨率的网格

    Args:
        points_xyz: Nx3点云数组
        dem_meta: DEM元数据（仅用于获取原始范围和CRS）
        nodata_value: 输出中的无数据值

    Returns:
        np.ndarray: 插值后的4m分辨率DEM矩阵
    """
    logger.info("构建TIN并重采样到固定4m分辨率网格")

    # 提取点云数据
    xy_points = points_xyz[:, :2]
    z_values = points_xyz[:, 2]

    # 检查是否有足够的点进行插值
    if len(xy_points) < 3:
        logger.warning("点云数量不足，无法进行三角剖分插值")
        # 返回空网格（这里需要先计算4m网格的尺寸，暂时返回空数组，后续可优化）
        return np.array([])  # 临时处理，后续可根据范围生成

    try:
        # 创建线性插值器（Delaunay三角剖分）
        interpolator = LinearNDInterpolator(xy_points, z_values, fill_value=nodata_value)

        # ==================== 修改部分开始 ====================
        # 强制使用 4m 分辨率
        target_resolution = 4.0  # 单位：米

        # 获取原始DEM的地理范围
        bounds = dem_meta['bounds']          # (left, bottom, right, top)
        left, bottom, right, top = bounds

        # 计算4m分辨率下的新宽度和高度
        new_width = int(np.ceil((right - left) / target_resolution))
        new_height = int(np.ceil((top - bottom) / target_resolution))

        # 创建新的仿射变换（左上角坐标 + 4m分辨率）
        new_transform = rasterio.transform.from_origin(
            west=left,
            north=top,
            xsize=target_resolution,
            ysize=target_resolution
        )

        # 生成4m分辨率的网格坐标
        cols, rows = np.meshgrid(np.arange(new_width), np.arange(new_height))
        xs, ys = rasterio.transform.xy(new_transform, rows.flatten(), cols.flatten())
        grid_coords = np.column_stack([xs, ys])
        # ==================== 修改部分结束 ====================

        # 分批处理以防止内存溢出
        batch_size = 1000000
        num_batches = int(np.ceil(len(grid_coords) / batch_size))

        interpolated_values = np.full(len(grid_coords), nodata_value)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(grid_coords))

            batch_coords = grid_coords[start_idx:end_idx]
            interpolated_values[start_idx:end_idx] = interpolator(batch_coords)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(f"插值进度: {batch_idx + 1}/{num_batches}批次 "
                          f"({end_idx:,}/{len(grid_coords):,}点)")

        # 重塑为新的4m分辨率形状
        dem_interpolated = interpolated_values.reshape((new_height, new_width))

        # 处理插值失败的区域（NaN值）
        nan_mask = np.isnan(dem_interpolated)
        dem_interpolated[nan_mask] = nodata_value

        logger.info(f"插值完成（4m分辨率），有效像素: {np.sum(dem_interpolated != nodata_value):,}/"
                   f"{dem_interpolated.size:,} | 尺寸: {new_width} x {new_height}")

        return dem_interpolated

    except Exception as e:
        logger.error(f"插值过程中发生错误: {e}")
        raise


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


def fuse_dem_and_points(dem_path: str, points_path: str, output_path: str,
                       mask_shp_path: str = None,
                       points_skip_rows: int = 0, dem_sample_fraction: float = 1.0,
                       chunk_size: int = 1000000):
    """
    主函数：融合DEM与高精度点云（使用shapefile掩模）

    Args:
        dem_path: 基础DEM文件路径
        points_path: 高精度点云文件路径
        output_path: 输出DEM文件路径
        mask_shp_path: 掩模shapefile文件路径（默认为bay/bay/baby_mask.shp）
        points_skip_rows: 点云文件跳过的行数
        dem_sample_fraction: DEM采样比例（0-1]
        chunk_size: 处理大点云时的分块大小
    """
    start_time = time.time()
    logger.info("开始DEM融合处理（使用shapefile掩模）")
    logger.info(f"基础DEM: {dem_path}")
    logger.info(f"高精度点云: {points_path}")
    logger.info(f"掩模文件: {mask_shp_path if mask_shp_path else 'bay/bay/baby_mask.shp'}")
    logger.info(f"输出文件: {output_path}")

    try:
        # 步骤1: 加载掩模多边形
        logger.info("步骤1/7: 加载掩模多边形")
        if mask_shp_path is None:
            mask_shp_path = "bay/bay/baby_mask.shp"
        mask_polygon = load_mask_polygon(mask_shp_path)

        # 步骤2: 读取高精度点云
        logger.info("步骤2/7: 读取高精度点云")
        highres_x, highres_y, highres_z = read_xyz_points(points_path, points_skip_rows)
        highres_points = np.column_stack([highres_x, highres_y, highres_z])

        logger.info(f"高精度点云统计: {len(highres_points):,}个点")
        logger.info(f"X范围: [{highres_x.min():.3f}, {highres_x.max():.3f}]")
        logger.info(f"Y范围: [{highres_y.min():.3f}, {highres_y.max():.3f}]")
        logger.info(f"Z范围: [{highres_z.min():.3f}, {highres_z.max():.3f}]")

        # 步骤3: 筛选高精度点云中落在掩模内的点
        logger.info("步骤3/7: 筛选掩模内的高精度点")
        highres_points_in_mask = filter_points_inside_polygon(highres_points, mask_polygon)

        if len(highres_points_in_mask) == 0:
            logger.warning("没有高精度点落在掩模区域内，将使用原始DEM数据")
            # 如果没有高精度点，直接使用原始DEM
            with rasterio.open(dem_path) as src:
                dem_array = src.read(1)
                dem_meta = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodatavals[0],
                    'bounds': src.bounds
                }
                write_dem(output_path, dem_array, dem_meta)
                logger.info("处理完成：直接复制原始DEM（没有高精度点落在掩模内）")
                return

        logger.info(f"掩模内高精度点: {len(highres_points_in_mask):,}个点")

        # 步骤4: 读取基础DEM点云
        logger.info("步骤4/7: 读取基础DEM点云")
        dem_points, dem_meta = extract_dem_points(dem_path, dem_sample_fraction)

        # 步骤5: 从DEM点云中剔除掩模内点（挖洞）
        logger.info("步骤5/7: 从DEM点云中挖洞（移除掩模内点）")
        filtered_dem_points = filter_points_by_polygon(dem_points, mask_polygon)

        # 步骤6: 合并点云
        logger.info("步骤6/7: 合并点云")
        if len(filtered_dem_points) > 0:
            merged_points = np.vstack([filtered_dem_points, highres_points_in_mask])
        else:
            merged_points = highres_points_in_mask

        logger.info(f"合并后点云总数: {len(merged_points):,}个点")
        logger.info(f"  - DEM点云（掩模外）: {len(filtered_dem_points):,}个点")
        logger.info(f"  - 高精度点（掩模内）: {len(highres_points_in_mask):,}个点")

        # 步骤7: 构建TIN并重采样
        logger.info("步骤7/7: 构建TIN并重采样到网格")
        interpolated_dem = interpolate_to_grid(merged_points, dem_meta)

        # 写入输出文件
        write_dem(output_path, interpolated_dem, dem_meta)

        # 计算处理时间
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成! 总耗时: {elapsed_time:.2f}秒")
        logger.info(f"输出文件: {output_path}")

        # 输出统计信息
        valid_pixels = np.sum(interpolated_dem != dem_meta.get('nodata', -9999.0))
        total_pixels = interpolated_dem.size
        logger.info(f"输出DEM统计: {valid_pixels:,}/{total_pixels:,}有效像素 "
                   f"({100*valid_pixels/total_pixels:.1f}%)")

    except Exception as e:
        logger.error(f"DEM融合处理失败: {e}")
        raise


def main():
    """主函数：示例调用"""
    # 设置文件路径
    dem_path = "study_area_dem.tif"
    points_path = "bay/bay/bay.txt"
    mask_shp_path = "bay/bay/baby_mask.shp"
    output_path = "fused_dem_result.tif"

    # 配置参数
    config = {
        'mask_shp_path': mask_shp_path,
        'points_skip_rows': 0,      # 没有表头行
        'dem_sample_fraction': 0.02, # 采样2%的DEM点（约240万点）
    }

    try:
        # 调用融合函数
        fuse_dem_and_points(
            dem_path=dem_path,
            points_path=points_path,
            output_path=output_path,
            **config
        )

    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()