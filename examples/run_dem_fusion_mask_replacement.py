#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM融合脚本使用示例
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nh_flood_2d.dem.dem_fusion_mask_replacement import fuse_dem_with_mask_replacement

def main():
    """示例：融合DEM与高精度点云"""

    # 输入文件路径（根据实际项目结构调整）
    base_dir = "resource/dem_rebuilt"

    inputs = {
        'dem_path': os.path.join(base_dir, 'input', 'study_area_dem.tif'),
        'mask_shp_path': os.path.join(base_dir, 'input', 'study_area_dem_mask.shp'),
        'bay_points_path': os.path.join(base_dir, 'input', 'bay', 'bay', 'bay.txt'),
        'shenzhenhe_points_path': os.path.join(base_dir, 'input', 'bay', 'bay', 'shenzhenhe-fix.csv'),
        'output_path': os.path.join(base_dir, 'output', 'fused_dem_4m.tif')
    }

    # 检查输入文件是否存在
    for key, path in inputs.items():
        if not os.path.exists(path):
            print(f"警告: 输入文件不存在: {path}")
            print(f"请确保文件位于正确位置")
            # 尝试查找文件
            import glob
            possible_files = glob.glob(f"**/{os.path.basename(path)}", recursive=True)
            if possible_files:
                print(f"找到可能的文件: {possible_files[:3]}")
            return

    print("开始DEM融合处理...")
    print(f"原始DEM: {inputs['dem_path']}")
    print(f"掩模文件: {inputs['mask_shp_path']}")
    print(f"bay.txt点云: {inputs['bay_points_path']}")
    print(f"shenzhenhe.csv点云: {inputs['shenzhenhe_points_path']}")
    print(f"输出文件: {inputs['output_path']}")

    try:
        # 执行融合
        fuse_dem_with_mask_replacement(
            dem_path=inputs['dem_path'],
            mask_shp_path=inputs['mask_shp_path'],
            bay_points_path=inputs['bay_points_path'],
            shenzhenhe_points_path=inputs['shenzhenhe_points_path'],
            output_path=inputs['output_path'],
            output_resolution=4.0,
            max_triangle_area=8.0,
            nodata_value=-9999.0
        )

        print("DEM融合处理完成!")

    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()