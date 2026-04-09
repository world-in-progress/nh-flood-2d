#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合最终生产脚本 - 使用shapefile掩模
"""

import os
import sys
import time
import logging

# 将当前目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_final_dem_fusion():
    """运行最终的DEM融合（使用shapefile掩模）"""
    try:
        from .dem_fusion import fuse_dem_and_points
    except ImportError:
        # Fallback for when running as standalone script
        from dem_fusion import fuse_dem_and_points

    # 设置文件路径
    dem_path = "study_area_dem.tif"
    points_path = "bay/bay/bay.txt"
    mask_shp_path = "bay/bay/baby_mask.shp"
    output_path = "fused_dem_result.tif"

    # 优化参数配置
    # 考虑到数据规模：
    # - DEM: 12751 x 9601 = 122,475,751 像素
    # - 点云: 11,650,740 个点
    # 使用以下优化参数：
    config = {
        'mask_shp_path': mask_shp_path,
        'points_skip_rows': 0,      # 没有表头行
        'dem_sample_fraction': 0.02, # 采样2%的DEM点（约240万点）
    }

    logger.info("=" * 80)
    logger.info("DEM融合处理 - 使用shapefile掩模")
    logger.info("=" * 80)
    logger.info(f"基础DEM: {dem_path} (12751x9601像素)")
    logger.info(f"高精度点云: {points_path} (11,650,740个点)")
    logger.info(f"掩模文件: {mask_shp_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"配置参数: {config}")
    logger.info("-" * 80)
    logger.info("处理流程:")
    logger.info("  1. 加载mask.shp作为掩模多边形")
    logger.info("  2. 筛选高精度点云中落在掩模内的点")
    logger.info("  3. 从DEM点云中移除掩模内的点（挖洞）")
    logger.info("  4. 合并掩模外的DEM点与掩模内的高精度点")
    logger.info("  5. 构建TIN并重采样到4m分辨率网格")
    logger.info("=" * 80)

    try:
        start_time = time.time()

        # 调用融合函数
        fuse_dem_and_points(
            dem_path=dem_path,
            points_path=points_path,
            output_path=output_path,
            **config
        )

        # 计算处理时间
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        logger.info("=" * 80)
        logger.info("DEM融合处理完成！")
        logger.info(f"总处理时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logger.info(f"输出文件: {output_path}")
        logger.info("=" * 80)

        # 显示文件大小
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"输出文件大小: {file_size_mb:.2f} MB")

    except KeyboardInterrupt:
        logger.info("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"DEM融合处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_final_dem_fusion()