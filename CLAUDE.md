# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nh-flood-2d` is a 2D hydrodynamic model that simulates shallow water flow on multi-resolution Cartesian grids. It uses GPU-accelerated computation via [Taichi](https://taichi-lang.org/) and stores mesh/simulation data in a custom binary format via [`fastdb4py`](https://github.com/world-in-progress/fastdb) (source: https://github.com/world-in-progress/fastdb).

## Commands

This project uses `uv` for dependency management (Python 3.10.17 required).

```bash
# Install dependencies
uv sync

# Run the simulation
uv run python main.py
```
There is no test suite or linter configured.

## Architecture

The pipeline has three main stages: **Preprocess → Solve → Output**.

### Entry Point

`main.py` — loads `./resource/config.json` as an `InputConfig`, then calls `preprocess(config)`, `solver(config)`, and output functions. Some steps are commented out during development.

### Configuration (`src/nh_flood_2d/input/`)

配置系统现在分为两个独立的部分：`DomainConfig`（域配置）和`ForceConfig`（驱动力配置）。

#### DomainConfig (`src/nh_flood_2d/input/domain.py`)
包含地形和模拟域相关的配置：

- `ne`, `ns` — 水力学元素（NE）和边（NS）的原始数据文件路径
- `epsg_code` — 坐标系EPSG代码
- `domain_dir` — 域输出目录
- `afa` — Courant数（CFL条件）；默认0.5
- `sita` — 时间加权因子；默认1.0
- `min_h` — 最小水深阈值；默认0.02米
- `duration` — 模拟总时长（秒）；`-1`表示自动从输入数据检测
- `yield_step` — 输出间隔（秒）；默认300（5分钟）
- `hydrograph_points` — 观测站点的`名称 -> (x, y)`字典
- `observation_dir` — 观测数据目录，文件名应与`hydrograph_points`键名相同

#### ForceConfig (`src/nh_flood_2d/input/force.py`)
包含驱动力和边界条件相关的配置：

- `gate` — 闸门原始数据文件路径
- `tide` — 潮汐原始数据文件路径
- `rain` — 降雨原始数据文件路径
- `force_dir` — 驱动力输出目录

#### 配置加载函数
- `load_domain_config(config_path)` — 从JSON文件加载DomainConfig
- `load_force_config(config_path)` — 从JSON文件加载ForceConfig

两个配置类都提供派生路径的`@property`方法（如`ne_fdb`, `tide_fdb`等）。

### Preprocessing (`src/nh_flood_2d/preprocess/`)

预处理现在分为两个独立的部分，分别处理域数据和驱动力数据：

#### 主预处理函数 (`__init__.py`)
- `preprocess(domain_cfg, force_cfg)` — 主预处理函数，依次调用`prepare_force`和`prepare_domain`

#### 域数据预处理 (`domain.py`)
- `prepare_domain(domain_cfg)` — 处理地形数据，生成：
  - `ne.fdb` — 水力学元素（Ne），包含侧向索引查找表（CSR格式）
  - `ns.fdb` — 水力学边（Ns），包含拓扑信息（SideTopoInfo）
  - `boundary.fdb` — 边界元素识别，使用Taichi GPU内核

#### 驱动力数据预处理 (`force.py`)
- `prepare_force(force_cfg)` — 处理驱动力数据，生成：
  - `gate.fdb` — 闸门数据
  - `tide.fdb` — 潮汐时间序列
  - `rain.fdb` — 降雨时间序列

可选功能：`_filter_ne_ns()` 移除`z == -9999`的元素并重新编号索引。

### Core Simulation (`src/nh_flood_2d/core/`)

当前有两个实现，但`solver_compact.py`是主要的生产求解器：

#### **`solver_compact.py`** — 紧凑型功能式求解器
- 主求解器函数：`solver(domain_cfg, force_cfg, start_time_step=0)`
- 高程设置函数：`set_elevation(domain_cfg, elevate_meter)` — 将低于指定高程的元素地面高程提升
- 所有状态作为围绕Taichi内核的局部变量/闭包
- 优化了内存管理和性能

#### **`solver.py`** — 原始功能式求解器
- 早期的实现，现已被`solver_compact.py`取代

#### **`domain.py`** — 面向对象的Domain类
- 实验性实现，封装相同逻辑

#### 模拟循环流程（所有实现）：
1. 根据当前时间线性插值潮汐边界条件
2. 从输入CSV计算降雨率
3. 调用`tick()` Taichi内核：
   - 根据上下游水头更新闸门状态（开/关）
   - 使用半隐式Saint-Venant方案在所有边上推进水流
   - 根据土地利用类型（7种）应用Horton下渗模型
   - 更新每个元素的水深；强制执行最小水深
   - 将边界元素水位设置为当前潮汐值
4. 每`yield_step`秒写入UVH（u-速度, v-速度, h-水位高程）数据到`domain_dir/uvh/uvh_<timestamp>.fdb`

#### 物理参数：
- `n = 0.033` — Manning糙率系数
- `g = 9.81` — 重力加速度（m/s²）
- `afa` — Courant数（CFL条件）
- `sita` — 时间加权因子
- `min_h` — 最小水深阈值（米）

### Data Schema (`src/nh_flood_2d/schema/feature.py`)

All stored data uses `fastdb4py` `Feature` subclasses:
- `Ne` — hydro element: index, x, y, z (ground elevation), 4 side counts, type (land use 1–7)
- `Ns` — hydro side: index, length, x, y, z, attr
- `SideTopoInfo` — packed as `[orient, lower_ei, upper_ei]` per side; orient 1=horizontal, 2=vertical
- `Tide`, `Rainfall` — time-series boundary conditions
- `Gate` — packed array of 100 int32 per gate: [upstream_ei, downstream_ei, height, influenced_ei...]
- `UVH` — simulation output per element: u, v, h
- `IndexLike`, `U8Value` — generic index/flag storage

**Virtual element 0 and virtual side 0** are padding; all real data starts at index 1.

### Output (`src/nh_flood_2d/output/`)

#### **`flood_map.py`** — 洪水图生成
- `generate_flood_map(cfg)` — 将UVH快照栅格化为GeoTIFF洪水地图，使用分块GPU处理（块大小4096）
- `generate_max_inundation_extent_map(cfg, min_depth=0.2, invalid_data=-9999.0)` — 生成最大淹没范围地图（所有时间步的最大水深）
- `get_area_meta(ne_fdb_fn, ns_fdb_fn)` — 从ne.fdb和ns.fdb计算区域元数据（边界框、分辨率、元素半尺寸）
- 输出CRS从`config.epsg_code`设置

#### **`hydrograph.py`** — 水文过程线分析
- `draw_hydrograph(cfg, station_name, clampped=True, translation_second=0)` — 在指定站点绘制模拟水位时间序列，可选择与观测数据对比
- `compare_hydrograph(cfgs, station_name, clampped=True, translation_second=0, forward_ignore_second=0, show_obs=True, baseline=None, show=True)` — 比较多个配置在同一站点的水文过程线，返回均方根误差（RMSE）列表
- `_find_ei(ne_fdb_path, ns_fdb_path, x, y)` — 根据坐标查找元素索引的内部函数
- `_extract_data(cfg, station_name)` — 从UVH文件提取站点数据的内部函数

### Utilities (`src/nh_flood_2d/util/`)

- `ti.py` — `init_taichi()` (with singleton guard) and `copy_to_taichi()` for converting numpy arrays to Taichi fields
- `benchmark.py` — timing decorator

## Key Conventions

- **Index 0 is always virtual** (placeholder). Real elements/sides start at index 1 in all arrays and Taichi kernel loops.
- Taichi kernels use `@no_type_check` because Taichi's type inference conflicts with Python type checkers.
- `fastdb4py` (FDB) is a columnar binary store. Access pattern: `db[FeatureClass]['table_name'].column.field_name` returns a numpy array.
- 配置系统已分离为`DomainConfig`（地形/模拟域）和`ForceConfig`（驱动力/边界条件）
- 预处理分为`prepare_domain()`和`prepare_force()`两个独立步骤
- `solver_compact.py`是当前的主要生产求解器，取代了`solver.py`
- `set_elevation()`函数可用于提升特定区域的地面高程
- `evolve_domain(domain_cfg, force_cfg)`是主要的模拟流程封装函数（见`main.py`）
- 输出系统支持洪水地图生成和水文过程线分析/比较
- Raw input file formats: NE/NS are CSV-like text files; tide/rainfall are CSV with datetime headers.
