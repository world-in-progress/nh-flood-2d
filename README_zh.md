# nh-flood-2d

基于多分辨率笛卡尔网格的二维水动力模型，使用GPU加速计算模拟浅水流动。

## 概述

`nh-flood-2d` 是一个高性能的二维水动力模拟框架，专为洪水建模和分析设计。它利用 [Taichi](https://taichi-lang.org/) 实现GPU加速，并使用 [`fastdb4py`](https://github.com/world-in-progress/fastdb) 的自定义二进制格式存储网格/模拟数据。

### 主要特性

- **GPU加速计算**：使用Taichi实现高性能模拟
- **多分辨率笛卡尔网格**：支持灵活的域表示
- **全面的物理模型**，包括：
  - 浅水流动的半隐式Saint-Venant方程
  - 7种土地利用类型的Horton下渗模型
  - 基于水头差的闸门操作逻辑
  - 带线性插值的潮汐边界条件
  - 随时间变化的降雨强迫
- **模块化架构**：域配置和驱动力配置分离
- **多种输出格式**：
  - GeoTIFF洪水地图（栅格化水深）
  - 观测站点的水文过程线时间序列
  - 中间数据存储的二进制FDB文件
- **可配置的模拟参数**：
  - Courant数（CFL条件）
  - 时间加权因子
  - 最小水深阈值
  - 输出间隔和时长

## 安装

本项目使用 `uv` 进行依赖管理（需要Python 3.10.17）。

```bash
# 安装依赖
uv sync

# 运行模拟
uv run python main.py
```

## 项目结构

```
nh-flood-2d/
├── src/nh_flood_2d/
│   ├── input/              # 配置管理
│   │   ├── __init__.py     # 主导入（DomainConfig, ForceConfig）
│   │   ├── domain.py       # 域配置（地形、模拟参数）
│   │   └── force.py        # 驱动力配置（边界条件）
│   ├── preprocess/         # 数据预处理
│   │   ├── __init__.py     # 主预处理函数
│   │   ├── domain.py       # 域数据准备
│   │   ├── force.py        # 驱动力数据准备
│   │   ├── pass_1.py       # 遗留的第1阶段（原始数据转FDB）
│   │   └── pass_2.py       # 遗留的第2阶段（边界识别）
│   ├── core/               # 核心模拟引擎
│   │   ├── solver_compact.py  # 主要生产求解器（GPU加速）
│   │   ├── solver.py          # 遗留的功能式求解器
│   │   └── domain.py          # 实验性面向对象实现
│   ├── output/             # 输出生成
│   │   ├── flood_map.py    # GeoTIFF洪水地图生成
│   │   └── hydrograph.py   # 水文过程线分析和绘图
│   ├── schema/             # 数据模式定义
│   │   └── feature.py      # 数据存储的FDB Feature类
│   └── util/               # 工具函数
│       ├── ti.py           # Taichi初始化和辅助函数
│       └── benchmark.py    # 性能测量的计时装饰器
├── main.py                 # 主入口点，包含使用示例
├── resource/               # 配置和输入数据文件
│   ├── domain_*.json       # 域配置文件
│   ├── df*.json           # 驱动力配置文件
│   └── elevate/           # 高程调整数据
└── CLAUDE.md              # Claude Code开发者指南
```

## API参考

以下函数在 `main.py` 中暴露并可供使用：

### 配置加载

```python
from src.nh_flood_2d.input import load_domain_config, DomainConfig, load_force_config, ForceConfig

# 加载域配置（地形和模拟参数）
domain_cfg = load_domain_config('./resource/domain_mrcg.json')

# 加载驱动力配置（边界条件）
force_cfg = load_force_config('./resource/df7.json')
```

### 主模拟流程

```python
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver

# 完整的模拟工作流
def evolve_domain(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    preprocess(domain_cfg, force_cfg)    # 数据准备
    solver(domain_cfg, force_cfg)        # 核心模拟
```

### 高程调整

```python
from src.nh_flood_2d.core.solver_compact import set_elevation

# 提升低于指定高程的元素地面高程
set_elevation(domain_cfg, elevate_meter=3.0)
```

### 输出生成

```python
from src.nh_flood_2d.output.flood_map import generate_flood_map, generate_max_inundation_extent_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph

# 生成洪水地图
generate_flood_map(domain_cfg)                      # 单时间步洪水地图
generate_max_inundation_extent_map(domain_cfg)      # 最大淹没范围地图

# 分析水文过程线
draw_hydrograph(domain_cfg, 'D74', clampped=True, translation_second=-3600)

# 比较多个模拟
mses = compare_hydrograph(
    [domain_cfg1, domain_cfg2],
    'D74',
    clampped=True,
    show=False,
    show_obs=False,
    baseline=domain_cfg1
)
```

## 配置文件

### 域配置 (`domain_*.json`)
```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 4326,
  "domain_dir": "output/domain_mrcg",
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "duration": -1,
  "yield_step": 300,
  "hydrograph_points": {
    "D74": [827040.3, 843912.8],
    "D75": [827120.5, 843850.2]
  },
  "observation_dir": "path/to/observations"
}
```

### 驱动力配置 (`df*.json`)
```json
{
  "gate": "path/to/gate.txt",
  "tide": "path/to/tide.csv",
  "rain": "path/to/rain.csv",
  "force_dir": "output/force_df7"
}
```

## 使用示例

### 基本模拟
```python
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.input import load_domain_config, load_force_config

# 加载配置
domain_cfg = load_domain_config('./resource/domain_mrcg.json')
force_cfg = load_force_config('./resource/df7.json')

# 运行完整模拟
preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

### 高程调整 + 模拟
```python
from src.nh_flood_2d.core.solver_compact import set_elevation

# 模拟前调整高程
set_elevation(domain_cfg, elevate_meter=3.0)

# 运行模拟
preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

### 后处理分析
```python
# 生成洪水地图
generate_flood_map(domain_cfg)
generate_max_inundation_extent_map(domain_cfg, min_depth=0.2)

# 绘制水文过程线
draw_hydrograph(domain_cfg, 'D74', clampped=True)

# 比较模拟
mses = compare_hydrograph(
    [domain_mrcg, domain_4],
    'D74',
    clampped=True,
    show=True
)
print(f'均方根误差值: {mses}')
```

## 数据模式

模型使用 `fastdb4py` Feature子类进行数据存储：

| Feature | 描述 | 字段 |
|---------|------|------|
| `Ne` | 水力学元素 | `index`, `x`, `y`, `z`, `type` (1-7) |
| `Ns` | 水力学边 | `index`, `length`, `x`, `y`, `z`, `attr` |
| `SideTopoInfo` | 边拓扑 | `[orient, lower_ei, upper_ei]` |
| `Tide` | 潮汐时间序列 | `time`, `level` |
| `Rainfall` | 降雨时间序列 | `time`, `quantity` |
| `Gate` | 闸门信息 | `info[100]`（每闸门） |
| `UVH` | 模拟输出 | 每个元素的 `u`, `v`, `h` |
| `IndexLike` | 索引存储 | `index` |
| `U8Value` | 8位值存储 | `value` |

**注意：** 索引0始终是虚拟的（占位符）。真实的元素/边从索引1开始。

## 物理模型

### 控制方程
模型使用半隐式有限体积格式求解二维浅水方程（Saint-Venant方程）：

1. **连续性方程**: ∂h/∂t + ∇·(hu) = R - I
2. **动量方程**: ∂u/∂t + u·∇u = -g∇h - g∇z - τ/ρ

其中：
- `h`: 水深
- `u`: 流速矢量
- `z`: 地面高程
- `R`: 降雨率
- `I`: 下渗率
- `g`: 重力加速度
- `τ`: 底摩擦（Manning公式）
- `ρ`: 水密度

### 下渗模型
按土地利用类型（7种类型）应用Horton下渗模型：
- 建筑、道路、农业用地、鱼塘、山地、水体、集水区

### 闸门操作
基于上下游水头差控制闸门开/关。

### 边界条件
- 潮汐：域边界随时间变化的水位
- 降雨：域上随时间变化的降水率

## 性能

- **GPU加速**：所有核心计算通过Taichi在GPU上运行
- **内存高效**：FDB格式最小化内存占用
- **可扩展**：大域的分块处理（块大小4096）
- **优化**：CSR类数据布局实现高效邻域访问

## 贡献

请参考 `CLAUDE.md` 获取详细的开发者指南和代码库约定。

## 许可证

[待添加许可证信息]

## 引用

如果您在研究中使用了此软件，请引用：

```
[待添加引用信息]
```

## 致谢

- [Taichi](https://taichi-lang.org/) 提供GPU计算基础设施
- [fastdb4py](https://github.com/world-in-progress/fastdb) 提供高效的二进制数据存储
- [rasterio](https://rasterio.readthedocs.io/) 提供GeoTIFF输出支持