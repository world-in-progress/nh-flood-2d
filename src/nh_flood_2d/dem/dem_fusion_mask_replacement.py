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