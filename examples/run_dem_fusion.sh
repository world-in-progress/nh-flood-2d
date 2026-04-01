#!/bin/bash
# run_dem_fusion.sh - DEM融合脚本运行脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 输入文件路径
DEM_FILE="$PROJECT_ROOT/resource/dem_rebuilt/input/study_area_dem.tif"
MASK_SHP="$PROJECT_ROOT/resource/dem_rebuilt/input/study_area_dem_mask.shp"
BAY_POINTS="$PROJECT_ROOT/resource/dem_rebuilt/input/bay/bay/bay.txt"
SHENZHENHE_POINTS="$PROJECT_ROOT/resource/dem_rebuilt/input/bay/bay/shenzhenhe-fix.csv"
OUTPUT_FILE="$PROJECT_ROOT/resource/dem_rebuilt/output/fused_dem_4m.tif"

# 检查输入文件
echo "检查输入文件..."
for file in "$DEM_FILE" "$MASK_SHP" "$BAY_POINTS" "$SHENZHENHE_POINTS"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
done

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 运行DEM融合脚本
echo "运行DEM融合脚本..."
python "$PROJECT_ROOT/src/nh_flood_2d/dem/dem_fusion_mask_replacement.py" \
    --dem "$DEM_FILE" \
    --mask "$MASK_SHP" \
    --bay-points "$BAY_POINTS" \
    --shenzhenhe-points "$SHENZHENHE_POINTS" \
    --output "$OUTPUT_FILE" \
    --resolution 4.0 \
    --max-triangle-area 8.0 \
    --nodata -9999.0

# 检查输出
if [ -f "$OUTPUT_FILE" ]; then
    echo "处理完成! 输出文件: $OUTPUT_FILE"

    # 显示文件信息
    echo "输出文件信息:"
    gdalinfo "$OUTPUT_FILE" | grep -E "(Size|Pixel|Origin|Corner|NoData|Coordinate System)"
else
    echo "错误: 输出文件未生成"
    exit 1
fi