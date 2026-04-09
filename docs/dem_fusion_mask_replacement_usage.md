# DEM融合脚本（分块掩模替换法）使用说明

## 功能概述

本脚本使用分块掩模替换法将低精度DEM与高精度点云融合，生成指定分辨率的输出DEM。

## 算法步骤

1. **DEM重采样**：将原始DEM重采样到目标分辨率（默认4米）
2. **掩模栅格化**：将掩模shapefile栅格化到DEM网格
3. **掩模外像元提取**：提取掩模外像元并保留到最终输出
4. **高精度点云读取**：读取bay.txt和shenzhenhe.csv格式的点云数据
5. **带约束TIN生成**：构建最大三角形面积8平方米的Delaunay三角网
6. **掩模内插值**：对掩模内每个像元进行TIN插值
7. **图层合并**：合并掩模外保留像元和掩模内插值像元
8. **输出DEM**：写入最终4米分辨率DEM

## 使用方法

### 命令行方式

```bash
python src/nh_flood_2d/dem/dem_fusion_mask_replacement.py \
  --dem resource/dem_rebuilt/input/study_area_dem.tif \
  --mask resource/dem_rebuilt/input/study_area_dem_mask.shp \
  --bay-points resource/dem_rebuilt/input/bay/bay/bay.txt \
  --shenzhenhe-points resource/dem_rebuilt/input/bay/bay/shenzhenhe-fix.csv \
  --output resource/dem_rebuilt/output/fused_dem_4m.tif \
  --resolution 4.0 \
  --max-triangle-area 8.0 \
  --nodata -9999.0
```

### Python API方式

```python
from src.nh_flood_2d.dem.dem_fusion_mask_replacement import fuse_dem_with_mask_replacement

fuse_dem_with_mask_replacement(
    dem_path="resource/dem_rebuilt/input/study_area_dem.tif",
    mask_shp_path="resource/dem_rebuilt/input/study_area_dem_mask.shp",
    bay_points_path="resource/dem_rebuilt/input/bay/bay/bay.txt",
    shenzhenhe_points_path="resource/dem_rebuilt/input/bay/bay/shenzhenhe-fix.csv",
    output_path="resource/dem_rebuilt/output/fused_dem_4m.tif",
    output_resolution=4.0,
    max_triangle_area=8.0,
    nodata_value=-9999.0
)
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dem` | 原始DEM文件路径（TIFF格式） | 必填 |
| `--mask` | 掩模shapefile路径 | 必填 |
| `--bay-points` | bay.txt点云文件路径 | 必填 |
| `--shenzhenhe-points` | shenzhenhe.csv点云文件路径 | 必填 |
| `--output` | 输出DEM文件路径 | 必填 |
| `--resolution` | 输出分辨率（米） | 4.0 |
| `--max-triangle-area` | 最大三角形面积（平方米） | 8.0 |
| `--nodata` | 无数据值 | -9999.0 |

## 输入文件格式

### 1. DEM文件
- 格式：GeoTIFF (.tif)
- 坐标系：建议使用投影坐标系
- 无数据值：支持标准NoData值

### 2. 掩模shapefile
- 格式：ESRI Shapefile (.shp)
- 几何类型：多边形（Polygon）
- 坐标系：应与DEM一致
- 内容：掩模区域多边形

### 3. 高精度点云文件
- **bay.txt**: 制表符分隔，第一列为索引，格式：`索引 x y z`
- **shenzhenhe-fix.csv**: 逗号分隔，有标题行，格式：`x,y,z`

## 输出文件

输出为4米分辨率的GeoTIFF文件，包含：
- 掩模外区域：原始DEM重采样值
- 掩模内区域：高精度点云TIN插值
- 无数据区域：使用指定NoData值填充

## 性能优化

- **内存使用**：支持批处理，通过`batch_size`参数控制
- **三角形面积约束**：递归分割大三角形，确保最大面积≤8m²
- **坐标系统一**：自动转换掩模坐标系以匹配DEM
- **错误处理**：详细的日志记录和异常处理