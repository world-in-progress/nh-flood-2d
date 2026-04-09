#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本运行器 - 适配当前数据格式
"""

import os
import sys
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


def run_dem_fusion():
    """运行DEM融合"""
    try:
        from .dem_fusion import fuse_dem_and_points
    except ImportError:
        # Fallback for when running as standalone script
        from dem_fusion import fuse_dem_and_points

    # 设置文件路径
    dem_path = "Digital Terrain Model.tif"
    points_path = "bay/bay/bay.txt"
    output_path = "fused_dem_result.tif"

    # 针对gcd.txt格式的配置
    # 格式: "序号 x;y;z" (分号分隔)
    config = {
        'points_skip_rows': 0,      # 没有表头行
        'dem_sample_fraction': 0.05, # 采样5%的DEM点以减少计算量（大DEM时建议）
        'buffer_distance': 1.0,     # 凸包缓冲区距离（米），稍微扩大确保完全覆盖
    }

    logger.info("开始DEM融合处理")
    logger.info(f"基础DEM: {dem_path}")
    logger.info(f"高精度点云: {points_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"配置参数: {config}")

    try:
        # 调用融合函数
        fuse_dem_and_points(
            dem_path=dem_path,
            points_path=points_path,
            output_path=output_path,
            **config
        )

        logger.info("DEM融合处理完成！")

    except Exception as e:
        logger.error(f"DEM融合处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_dem_fusion()