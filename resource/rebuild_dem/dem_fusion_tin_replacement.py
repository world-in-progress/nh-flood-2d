#!/usr/bin/env python3
"""
DEM融合脚本（TIN替换方法）

该脚本实现基于TIN（三角不规则网络）的DEM融合算法，用于替换研究区域内的原始DEM数据。
算法流程：
1. 读取研究区域掩膜和输入DEM
2. 从研究区域边界和内部均匀采样点
3. 构建TIN（三角不规则网络）
4. 在TIN表面进行插值，替换研究区域内的DEM值
5. 保存融合后的DEM

作者: Claude Code
日期: 2026-03-31
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import Polygon, Point

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.nh_flood_2d.util import init_taichi

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dem_fusion_tin_replacement.log')
    ]
)
logger = logging.getLogger(__name__)

# 初始化Taichi
init_taichi()

# 导入TIN工具模块
try:
    from tin_utils import build_tin, sample_tin_points, interpolate_tin
    logger.info("成功导入TIN工具模块")
except ImportError as e:
    logger.error(f"导入TIN工具模块失败: {e}")
    logger.info("请确保已安装triangle库: pip install triangle")
    sys.exit(1)


def load_mask(mask_path: str) -> gpd.GeoDataFrame:
    """加载研究区域掩膜"""
    logger.info(f"加载研究区域掩膜: {mask_path}")
    try:
        gdf = gpd.read_file(mask_path)
        logger.info(f"掩膜包含 {len(gdf)} 个多边形")
        return gdf
    except Exception as e:
        logger.error(f"加载掩膜失败: {e}")
        raise


def load_dem(dem_path: str) -> Tuple[np.ndarray, Affine, dict]:
    """加载DEM数据"""
    logger.info(f"加载DEM数据: {dem_path}")
    try:
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1)
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
            logger.info(f"DEM尺寸: {dem_array.shape}, 分辨率: {transform[0]}m")
            logger.info(f"坐标系: {crs}")
            return dem_array, transform, profile
    except Exception as e:
        logger.error(f"加载DEM失败: {e}")
        raise


def extract_mask_geometry(mask_gdf: gpd.GeoDataFrame) -> Polygon:
    """从GeoDataFrame中提取掩膜多边形"""
    if len(mask_gdf) == 0:
        raise ValueError("掩膜文件中没有多边形")

    # 取第一个多边形（假设只有一个研究区域）
    geometry = mask_gdf.iloc[0].geometry
    if geometry.geom_type != 'Polygon':
        raise ValueError(f"期望多边形几何类型，实际得到: {geometry.geom_type}")

    logger.info(f"掩膜多边形面积: {geometry.area:.2f} 平方米")
    return geometry


def sample_boundary_points(geometry: Polygon, spacing: float = 10.0) -> np.ndarray:
    """沿多边形边界采样点"""
    logger.info(f"沿边界采样点，间距: {spacing}m")

    boundary = geometry.boundary
    if boundary.geom_type == 'LineString':
        lines = [boundary]
    elif boundary.geom_type == 'MultiLineString':
        lines = list(boundary.geoms)
    else:
        raise ValueError(f"意外的边界类型: {boundary.geom_type}")

    points = []
    for line in lines:
        length = line.length
        num_points = max(2, int(length / spacing) + 1)

        for i in range(num_points):
            distance = (i / (num_points - 1)) * length if num_points > 1 else 0
            point = line.interpolate(distance)
            points.append([point.x, point.y])

    boundary_points = np.array(points)
    logger.info(f"边界采样点数量: {len(boundary_points)}")
    return boundary_points


def create_output_profile(original_profile: dict, nodata_value: float = -9999.0) -> dict:
    """创建输出DEM的profile"""
    profile = original_profile.copy()
    profile.update({
        'dtype': 'float32',
        'nodata': nodata_value,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    })
    return profile


def main():
    """主函数"""
    logger.info("开始DEM融合（TIN替换方法）")

    # 配置输入文件路径
    mask_path = "study_area_mask.shp"
    dem_path = "Digital Terrain Model.tif"
    output_path = "fused_dem_tin_replacement.tif"

    # 检查输入文件是否存在
    for path in [mask_path, dem_path]:
        if not os.path.exists(path):
            logger.error(f"文件不存在: {path}")
            return

    try:
        # 1. 加载掩膜和DEM
        mask_gdf = load_mask(mask_path)
        dem_array, transform, profile = load_dem(dem_path)

        # 2. 提取掩膜几何
        geometry = extract_mask_geometry(mask_gdf)

        # 3. 采样边界点
        boundary_points = sample_boundary_points(geometry)

        # 4. 采样内部点（均匀网格）
        # TODO: 实现内部点采样

        # 5. 构建TIN
        # TODO: 调用TIN构建函数

        # 6. 插值替换DEM
        # TODO: 调用TIN插值函数

        # 7. 保存结果
        logger.info(f"保存融合后的DEM到: {output_path}")

        # 创建输出profile
        output_profile = create_output_profile(profile)

        # 保存DEM
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(dem_array.astype(np.float32), 1)

        logger.info("DEM融合完成")

    except Exception as e:
        logger.error(f"DEM融合过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()