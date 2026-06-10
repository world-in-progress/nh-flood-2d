# nh-flood-2d

`nh-flood-2d` 是一个面向二维浅水洪水模拟的 Python 3.10 建模工程。当前代码支持多分辨率笛卡尔网格上的二维地表水计算，也包含可选的 SWMM 一维管网耦合、DEM 融合工具，以及洪水图和水文过程线后处理。

求解器的主要数值计算由 Taichi kernel 执行，网格、强迫和 UVH 快照数据使用 `fastdb4py` FDB 文件保存。项目将 `fastdb4py` 固定在 `0.1.12`，因为后续 FastDB 变更不默认兼容当前模型使用的数据布局。

## 仓库内容

- 二维地表水预处理与模拟代码：`src/nh_flood_2d/preprocess`、`src/nh_flood_2d/core/solver_compact.py`。
- 可选二维地表水与一维 SWMM 管网耦合代码：`src/nh_flood_2d/core/coupled`。
- 网格、强迫、管网和 UVH 数据的 FDB schema：`src/nh_flood_2d/schema/feature.py`。
- 洪水图、最大淹没范围、视频、水文过程线和对比分析工具：`src/nh_flood_2d/output`。
- DEM 融合和 TIN 辅助代码：`src/nh_flood_2d/dem`、`examples`、`docs/dem_fusion_mask_replacement_usage.md`。
- SWMM `.inp` 转 shapefile 和 warm-start UVH 清理脚本：`tools`。

大型本地输入和模拟输出位于 `resource/`，该目录已被 Git 忽略。全新克隆不包含 `main.py` 中本地场景依赖的 DEM、NE/NS 网格、降雨、潮位、闸门、观测、SWMM 或 UVH 文件。

## 环境要求

- Python `3.10.17`。
- 使用 `uv` 管理依赖。
- 求解运行需要 Taichi 支持的计算后端。
- GIS 和 SWMM 相关依赖由 `pyproject.toml` 安装。
- 本地需要准备与配置 JSON 对应的模型数据文件。

安装依赖：

```bash
uv sync
```

macOS 上如果加载 `swmm-toolkit` 时因为 wheel 内置 dylib 签名问题导致进程退出，可运行：

```bash
uv run fix-macos-codesign
```

## 验证命令

```bash
uv lock --check
uv run pytest tests/test_flood_map_rainfall.py tests/test_flood_map_uvh_validation.py
```

这些 smoke tests 覆盖不依赖本地模型数据的输出工具行为。完整测试收集包含 DEM/GIS 和遗留本地数据测试，只应在具备所需 GMT/GIS 库和项目数据的环境中运行。当前测试不等同于完整洪水模拟验证。

## 数据与配置

模型使用独立的 JSON 文件配置二维计算域、外部强迫和可选管网。配置加载器会验证必要输入路径，并在需要时创建输出目录。

### 计算域配置

通过 `load_domain_config(...)` 加载为 `DomainConfig`。

```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 4547,
  "domain_dir": "path/to/domain-output",
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "duration": -1,
  "yield_step": 300,
  "restart_uvh": "",
  "hydrograph_points": {
    "S4": [827040.3, 843912.8]
  },
  "observation_dir": "path/to/observations"
}
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `ne`, `ns` | 原始网格单元和边文件。 |
| `epsg_code` | 栅格输出使用的坐标参考系统代码。 |
| `domain_dir` | 计算域输出根目录，保存预处理 FDB、UVH 快照、洪水图、水文过程线和最大淹没范围栅格。 |
| `afa` | 自适应时间步计算中的 CFL 系数。 |
| `sita` | 边流量更新中的时间权重系数。 |
| `min_h` | 最小有效水深，单位为米。 |
| `duration` | 模拟时长，单位为秒；`-1` 表示运行到强迫数据结束。 |
| `yield_step` | UVH 输出间隔，单位为秒。 |
| `restart_uvh` | 可选 warm-start UVH `.fdb` 快照路径。 |
| `hydrograph_points` | 水文站名到 `(x, y)` 坐标的映射。 |
| `observation_dir` | 可选观测文件目录，用于水文过程线对比。 |

### 强迫配置

通过 `load_force_config(...)` 加载为 `ForceConfig`。

```json
{
  "gate": "path/to/gate.txt",
  "tide": "path/to/tide.csv",
  "rain": "path/to/rain.csv",
  "force_dir": "path/to/force-output"
}
```

`preprocess(...)` 会将这些文件转换为 `force_dir/preprocessed` 下的 `gate.fdb`、`tide.fdb` 和 `rain.fdb`。

### 管网配置

需要一维二维耦合时，通过 `load_pipe_config(...)` 加载为 `PipeConfig`。

```json
{
  "inp": "path/to/network.inp",
  "pipe_dir": "path/to/pipe-output",
  "coupling_interval": 600.0,
  "exchange_timeout": 600.0,
  "weak_dist_thresh": 50.0
}
```

管网预处理会读取 SWMM `.inp` 中的节点，构建管网 FDB，记录每个节点的主关联二维单元和弱关联二维单元，并将运行时管网文件写入 `pipe_dir`。

## 运行二维求解器

`main.py` 是带有本地硬编码 `resource/` 路径的编排脚本。准备好本地配置文件后可以参考它组织流程，但它不是通用命令行入口。

最小二维流程：

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver

domain_cfg = load_domain_config("path/to/domain.json")
force_cfg = load_force_config("path/to/force.json")

preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

预处理阶段会创建：

- `domain_dir/preprocessed/ne.fdb`
- `domain_dir/preprocessed/ns.fdb`
- `domain_dir/preprocessed/boundary.fdb`
- `force_dir/preprocessed/gate.fdb`
- `force_dir/preprocessed/tide.fdb`
- `force_dir/preprocessed/rain.fdb`

求解器会将 `uvh_*.fdb` 快照写入 `domain_dir/uvh`。

## 运行二维一维耦合

SWMM 管网耦合流程：

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.input.pipe import load_pipe_config
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.preprocess.pipe import prepare_pipe
from src.nh_flood_2d.core.coupled import solver_coupled

domain_cfg = load_domain_config("path/to/domain.json")
force_cfg = load_force_config("path/to/force.json")
pipe_cfg = load_pipe_config("path/to/pipe.json")

preprocess(domain_cfg, force_cfg)
prepare_pipe(pipe_cfg, domain_cfg)
solver_coupled(domain_cfg, force_cfg, pipe_cfg)
```

`solver_coupled(...)` 会启动二维 Taichi 进程；提供 `pipe_cfg` 时，还会启动一维 SWMM 管网进程。两个进程通过 multiprocessing 管理的共享状态按 `coupling_interval` 秒交换排水和溢流数据。

调用 `solver_coupled(domain_cfg, force_cfg, None)` 会运行无一维管网进程的二维耦合代码路径。

## Warm Start

`DomainConfig.restart_uvh` 可以指向已有 UVH 快照。下面的辅助脚本会生成一个清理后的 warm-start 快照，只保留指定单元类型上的水量：

```bash
uv run python tools/clean_uvh_for_warmstart.py \
  --uvh path/to/uvh_20230908-000000.fdb \
  --ne path/to/preprocessed/ne.fdb \
  --out path/to/warmstart.fdb \
  --keep-types 7 8
```

随后在计算域配置中设置 `"restart_uvh": "path/to/warmstart.fdb"`。

## 后处理

常用输出函数：

```python
from src.nh_flood_2d.output.flood_map import (
    generate_flood_map,
    generate_max_inundation_extent_map,
    generate_flood_video,
    plot_spatial_mae_curve,
)
from src.nh_flood_2d.output.hydrograph import (
    draw_hydrograph,
    compare_hydrograph,
    compare_hydrograph_panels,
)

generate_flood_map(domain_cfg)
generate_max_inundation_extent_map(domain_cfg, min_depth=0.05)
generate_flood_video(domain_cfg, output_path="path/to/flood_video.mp4")
draw_hydrograph(domain_cfg, "S4")
```

这些函数依赖求解器生成的预处理网格 FDB 和 UVH 快照。

## DEM 与 GIS 工具

DEM 掩模替换融合命令：

```bash
uv run python src/nh_flood_2d/dem/dem_fusion_mask_replacement.py \
  --dem path/to/study_area_dem.tif \
  --mask path/to/study_area_dem_mask.shp \
  --bay-points path/to/bay.txt \
  --shenzhenhe-points path/to/shenzhenhe-fix.csv \
  --output path/to/fused_dem_4m.tif \
  --resolution 4.0 \
  --nodata -9999.0
```

SWMM `.inp` 转 shapefile：

```bash
uv run python tools/inp2shp.py path/to/network.inp -o path/to/shapefiles --epsg 4547
```

## 数据模型说明

FDB schema 定义在 `src/nh_flood_2d/schema/feature.py`。核心记录包括：

| Feature | 用途 |
| --- | --- |
| `Ne` | 二维网格单元坐标、高程、边数量和类型。 |
| `Ns` | 二维边的几何、高程、长度和属性。 |
| `SideTopoInfo` | 边方向和相邻单元索引。 |
| `Rainfall`, `Tide`, `Gate` | 预处理后的强迫记录。 |
| `UVH` | 每个单元的流速分量和水位。 |
| `Node`, `PipeTopo` | 耦合用 SWMM 节点和管网到二维单元拓扑记录。 |
| `IndexLike`, `U8Value`, `F32Value` | 小型类型化辅助表。 |

二维网格数据中的索引 `0` 是虚拟或哨兵元素/边，真实网格遍历从索引 `1` 开始。

## 代码结构

```text
src/nh_flood_2d/
  input/            Pydantic 配置模型和 JSON 加载器
  preprocess/       计算域、强迫、管网和 warm-start 预处理
  core/
    solver_compact.py
    coupled/        二维一维耦合求解器驱动和交换逻辑
  output/           洪水栅格、视频、水文过程线和对比图
  schema/           fastdb4py Feature 定义
  dem/              DEM 融合和 TIN 工具
  util/             Taichi 初始化和计时辅助函数
tools/              独立维护和转换脚本
examples/           DEM 融合运行示例
docs/               设计说明和使用文档
```
