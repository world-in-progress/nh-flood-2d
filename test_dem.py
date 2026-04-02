#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DEM数据类型
"""

import rasterio
import numpy as np
import sys

def check_dem_type():
    """检查DEM数据类型"""
    dem_path = "resource/rebuild_dem/input/study_area_dem.tif"

    with rasterio.open(dem_path) as src:
        print(f"DEM文件: {dem_path}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"NoData值: {src.nodatavals[0]}")

        # 读取一小部分数据
        data = src.read(1, window=((0, 10), (0, 10)))
        print(f"数据示例 (10x10):")
        print(data)
        print(f"数据类型: {data.dtype}")
        print(f"是否有NaN: {np.any(np.isnan(data))}")
        print(f"是否有inf: {np.any(np.isinf(data))}")

        # 检查数值范围
        print(f"数值范围: [{data.min()}, {data.max()}]")

if __name__ == "__main__":
    check_dem_type()