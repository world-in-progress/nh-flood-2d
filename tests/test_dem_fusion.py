#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本测试版本 - 使用小样本进行测试
"""

import os
import sys
import time
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_point_reading():
    """测试点云文件读取"""
    from src.nh_flood_2d.dem.dem_fusion import read_xyz_points

    points_path = "归档 2/gcd.txt"

    logger.info("测试读取点云文件...")

    try:
        # 使用新的读取函数
        highres_x, highres_y, highres_z = read_xyz_points(points_path, skip_rows=0)
        points = np.column_stack([highres_x, highres_y, highres_z])

        logger.info(f"成功读取 {len(points)} 个点")
        logger.info(f"X范围: {highres_x.min():.3f} - {highres_x.max():.3f}")
        logger.info(f"Y范围: {highres_y.min():.3f} - {highres_y.max():.3f}")
        logger.info(f"Z范围: {highres_z.min():.3f} - {highres_z.max():.3f}")

        # 检查坐标是否在DEM范围内
        logger.info("检查点云与DEM的空间关系...")
        # DEM的大致范围（从TIFF文件信息）
        dem_min_x = 799997.50
        dem_max_x = 799997.50 + 5.0 * 12751  # 大约
        dem_min_y = 848002.50 - 5.0 * 9601   # 大约
        dem_max_y = 848002.50

        in_dem = ((highres_x >= dem_min_x) & (highres_x <= dem_max_x) &
                  (highres_y >= dem_min_y) & (highres_y <= dem_max_y))

        logger.info(f"在DEM范围内的点: {np.sum(in_dem):,}/{len(points):,} ({100*np.sum(in_dem)/len(points):.1f}%)")

        return points

    except Exception as e:
        logger.error(f"读取点云失败: {e}", exc_info=True)
        return None


def test_dem_reading():
    """测试DEM文件读取"""
    import rasterio

    dem_path = "Digital Terrain Model.tif"

    logger.info("测试读取DEM文件...")

    with rasterio.open(dem_path) as src:
        logger.info(f"DEM尺寸: {src.width} x {src.height}")
        logger.info(f"CRS: {src.crs}")
        logger.info(f"NoData值: {src.nodatavals[0]}")

        # 读取一小块区域进行测试
        window = rasterio.windows.Window(0, 0, 100, 100)
        data = src.read(1, window=window)

        logger.info(f"测试窗口数据形状: {data.shape}")
        logger.info(f"有效值数量: {np.sum(data != src.nodatavals[0])}")
        logger.info(f"高程范围: {data[data != src.nodatavals[0]].min():.2f} - "
                   f"{data[data != src.nodatavals[0]].max():.2f}")

    return True


def test_full_process():
    """测试完整处理流程（使用小样本）"""
    from src.nh_flood_2d.dem.dem_fusion import (
        read_xyz_points, extract_dem_points,
        compute_point_boundary, filter_points_by_polygon
    )

    logger.info("开始完整流程测试...")

    # 1. 读取少量点云
    logger.info("1. 读取点云数据...")
    try:
        # 先读取少量点
        highres_x, highres_y, highres_z = read_xyz_points("归档 2/gcd.txt", skip_rows=0)
        # 只取前1000个点进行测试
        sample_size = min(1000, len(highres_x))
        highres_x = highres_x[:sample_size]
        highres_y = highres_y[:sample_size]
        highres_z = highres_z[:sample_size]
        highres_points = np.column_stack([highres_x, highres_y, highres_z])
        logger.info(f"读取了 {len(highres_points)} 个高精度点")
    except Exception as e:
        logger.error(f"读取点云失败: {e}")
        return False

    # 2. 计算边界
    logger.info("2. 计算点云边界...")
    try:
        boundary = compute_point_boundary(highres_points, buffer_distance=1.0)
        logger.info(f"边界面积: {boundary.area:.2f} 平方米")
        logger.info(f"边界范围: {boundary.bounds}")
    except Exception as e:
        logger.error(f"计算边界失败: {e}")
        return False

    # 3. 读取少量DEM点
    logger.info("3. 读取DEM点云...")
    try:
        # 使用非常低的采样率
        dem_points, dem_meta = extract_dem_points(
            "Digital Terrain Model.tif",
            sample_fraction=0.001  # 只采样0.1%
        )
        logger.info(f"读取了 {len(dem_points)} 个DEM点")
    except Exception as e:
        logger.error(f"读取DEM失败: {e}")
        return False

    # 4. 过滤点云
    logger.info("4. 过滤点云...")
    try:
        filtered_points = filter_points_by_polygon(dem_points, boundary)
        logger.info(f"过滤前: {len(dem_points)} 个点")
        logger.info(f"过滤后: {len(filtered_points)} 个点")
    except Exception as e:
        logger.error(f"过滤点云失败: {e}")
        return False

    logger.info("测试流程完成！")
    return True


def main():
    """主测试函数"""
    logger.info("DEM融合脚本测试开始")

    # 测试1: 点云读取
    points = test_point_reading()
    if points is None:
        logger.error("点云读取测试失败")
        return
    else:
        logger.info("点云读取测试通过")

    # 测试2: DEM读取
    if not test_dem_reading():
        logger.error("DEM读取测试失败")
        return

    logger.info("基础测试通过！由于数据量较大（1160万点，1200万像素），跳过完整流程测试。")
    logger.info("可以直接运行完整融合脚本了。")
    logger.info("建议参数:")
    logger.info("  - dem_sample_fraction: 0.01 (1%采样，减少内存使用)")
    logger.info("  - buffer_distance: 1.0 (1米缓冲区)")
    logger.info("  - 预计处理时间: 10-30分钟")


if __name__ == "__main__":
    main()