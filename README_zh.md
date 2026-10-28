# nh-flood-2d

`nh-flood-2d` 是一个面向二维浅水洪水模拟的 Python 3.10 建模工程。当前代码支持多分辨率笛卡尔网格上的二维地表水计算，也包含可选的 SWMM 一维管网耦合、DEM 融合工具，以及洪水图和水文过程线后处理。

求解器的主要数值计算由 Taichi kernel 执行，网格、水深帧、极值和完整检查点使用 `fastdb4py` FDB 文件保存。

## 仓库内容

- 二维地表水预处理与模拟代码：`src/nh_flood_2d/preprocess`、`src/nh_flood_2d/core/solver_compact.py`。
- 可选二维地表水与一维 SWMM 管网耦合代码：`src/nh_flood_2d/core/coupled`。
- 网格、强迫和管网数据的 FDB schema：`src/nh_flood_2d/schema/feature.py`；帧与检查点读写：`src/nh_flood_2d/core/output.py`。
- 洪水图、最大淹没范围、视频、水文过程线和对比分析工具：`src/nh_flood_2d/output`。
- DEM 融合和 TIN 辅助代码：`src/nh_flood_2d/dem`、`examples`、`docs/dem_fusion_mask_replacement_usage.md`。
- SWMM `.inp` 转 shapefile 和求解器自检脚本：`tools`。

大型本地输入和模拟输出位于 `resource/`，该目录已被 Git 忽略。全新克隆不包含本地配置所需的 DEM、NE/NS 网格、降雨、潮位、闸门、观测、SWMM 或检查点文件。

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
uv run pytest tests/test_forcing.py tests/test_solver_forcing.py \
  tests/test_clock.py tests/test_device_budget.py tests/test_face_flux.py \
  tests/test_open_boundary.py tests/test_hydraulics.py \
  tests/test_positivity.py tests/test_inventory_rounding.py tests/test_restart.py \
  tests/test_coupling.py tests/test_output_io.py \
  tests/test_output_readers.py tests/test_flood_map_frame_validation.py \
  tests/test_flood_map_rainfall.py
uv run pytest
```

专项检查覆盖强迫、设备时钟与收支、边流量、边界、耦合、检查点读写、水深帧与后处理。全量命令检查仓库其他部分，其中部分测试可能需要本地 DEM 数据或 GMT。

## 数据与配置

compact 求解器使用独立的计算域与强迫 JSON 文件。路径相对于进程工作目录解析，未知字段直接报错。`start`、`duration`、`warmup_duration`、`land_parameters` 必填；模拟时长和输出间隔为正整数秒，预热时长允许 0。`duration + warmup_duration` 必须小于 `2**24` 秒。

### 计算域配置

```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 2326,
  "domain_dir": "path/to/new-run",
  "start": "2023-09-07T16:00:00+08:00",
  "duration": 7200,
  "warmup_duration": 43200,
  "land_parameters": {
    "1": {"manning_n": 0.12, "infiltration": {"f0_mm_h": 76.2, "fc_mm_h": 2.54, "k_per_h": 2}},
    "2": {"manning_n": 0.08, "infiltration": {"f0_mm_h": 76.2, "fc_mm_h": 2.54, "k_per_h": 2}},
    "3": {"manning_n": 0.08, "infiltration": {"f0_mm_h": 20.32, "fc_mm_h": 0.508, "k_per_h": 10}},
    "4": {"manning_n": 0.03, "infiltration": "none"},
    "5": {"manning_n": 0.025, "infiltration": "none"},
    "6": {"manning_n": 0.035, "infiltration": {"f0_mm_h": 101.6, "fc_mm_h": 7.62, "k_per_h": 2}},
    "7": {"manning_n": 0.03, "infiltration": "none"},
    "8": {"manning_n": 0.03, "infiltration": "none"},
    "9": {"manning_n": 0.06, "infiltration": {"f0_mm_h": 152.4, "fc_mm_h": 25.4, "k_per_h": 3}}
  },
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "yield_step": 300,
  "restart_state": "",
  "stations": {"S4": [825187.3806, 841959.8752]},
  "observation_dir": "path/to/observations"
}
```

`start` 必须带时区。所有时间戳按绝对时间对齐，不依赖计算机本地时区。边界序列和降雨时段必须覆盖正式时窗 `start` 到 `start + duration`，覆盖不足即报错，正式模拟不外推或用首末值补齐。预热只使用 `start` 时刻的值，不读取 start 之前的数据。网格、边界线和降雨栅格必须使用同一 EPSG，不执行坐标变换。

`land_parameters` 必须覆盖网格中实际出现的每个地类，缺失即报错；JSON 键为地类编码。上面 1–9 类的数值取自论文附录，仅作为输入示例，程序没有内置默认表。网格如有其他地类，需添加相应配置。每类必填正值 `manning_n`，下渗可设为 `"none"`，或填写 Horton 参数 `f0_mm_h`、`fc_mm_h`、`k_per_h`。下渗率单位为 mm/h，衰减系数单位为 1/h；三者均须非负，且 `f0_mm_h` 不小于 `fc_mm_h`。内边的 n 取两侧单元 n 的算术平均；闭外边沿用内侧值，域外 0 号单元不算地类。

`afa`、`sita`、`min_h` 分别控制 CFL、时间权重和有效水深阈值。θ<1 时，相邻面单宽流量取该面所有子边按边长加权的平均；单边面保持原值。θ 模板只在被边界线选中的 level 或 discharge 外露边上进行线性延拓。外露比例按边长加权；一侧邻面完全缺失时，两邻面的平均等于本边旧流量。真实外边通量为 0，边界交换由指定水位和单元流量源项完成。θ=1 跳过邻面模板。

`stations` 将站名映射到网格坐标：包含坐标的闭矩形单元都是候选，取最小真实单元 ID；坐标落在网格外即报错。`observation_dir` 提供对比观测数据。`yield_step` 默认 300 s，控制站点采样、收支行和步长统计的间隔；水深帧固定每 900 s 输出。

### 强迫配置

```json
{
  "boundaries": [
    {
      "name": "sea",
      "kind": "level",
      "line": "path/to/sea.geojson",
      "series": "path/to/sea.csv"
    },
    {
      "name": "shenzhen_river",
      "kind": "discharge",
      "line": "path/to/river.geojson",
      "series": "path/to/river.csv"
    }
  ],
  "rainfall": {
    "raster": "path/to/rainfall.tif",
    "intervals": "path/to/rainfall_intervals.csv"
  },
  "gate": {"threshold": 1.5, "land_type": 11, "level_boundary": "sea"}
}
```

每条边界给出唯一的 `name`、`kind`、线和序列。`level` 是与网格高程同一垂直基准的水面高程，单位 m；`discharge` 是整条边界的有符号总流量，单位 m³/s，正值流入计算域。边界列表为空时，所有外露边闭合。线文件必须显式声明坐标系，例如：

```json
{
  "type": "LineString",
  "crs": {"type": "name", "properties": {"name": "EPSG:2326"}},
  "coordinates": [[808300, 836950], [808400, 836950]]
}
```

以上坐标仅演示格式，运行时需替换为与目标网格外边共线重合的线。接受 LineString、MultiLineString 及 Feature/FeatureCollection 包装。边界线与真实外露离散边的共线重合长度为正时，选中整条离散边。只接触角点或横穿产生单个交点均不选边。没有选中外边时会报错，并提示在 GIS 中将线吸附到网格外边。同一单元被多条边界占用也会报错。总 Q 只按选中边的长度分配；同一单元选中多条边时仍只得到一个合并份额。

边界 CSV 必须包含 `time`、`value` 两列（额外列忽略）：

```csv
time,value
2023-09-07T16:00:00+08:00,0.6
2023-09-07T17:00:00+08:00,0.8
2023-09-07T18:00:00+08:00,0.9
```

时间采用带时区的 ISO 8601，严格递增，至少两条记录，数值必须有限。河流 Q 采用同一格式，数值单位为 m³/s。序列线性插值，时间步在序列结点处截断；中点 Q 对各线性段精确积分，分配后总量仍是一条 Q。level 在初始时刻和每步末尾施加。

降雨是一份多波段 GeoTIFF，第 k 波段为时段 CSV 第 k 行的累计雨量 mm：

```csv
interval_start,interval_end
2023-09-07T16:00:00+08:00,2023-09-07T17:00:00+08:00
2023-09-07T17:00:00+08:00,2023-09-07T18:00:00+08:00
```

时段长度必须为正，连续、不重叠，记录数与波段数一致。无雨时段显式写 0。本例可使用 2、1 mm 两个波段。载入时按单元中心映射像元；中心落在栅格外，或所采像元为 NoData、非有限值、负雨量时均报错。时段内雨强恒定，等于波段 mm × 0.001 / 时段秒数。一个覆盖全域的像元就是均匀降雨特例。配置的 Horton 下渗能力为 `fc + (f0 - fc) exp(-k t / 3600)`，第一次有雨时容量为 `f0`。各单元仅在本格雨强为正时累计降雨时间 `t`；`none` 对应零下渗。实际 Horton 移除量为容量 × dt 与通量和降雨更新后剩余水量的较小值。

`gate` 可省略或设为 `null`。所参照的边界必须为 `level`；该边界超过 `threshold` 后，将 `land_type` 的全部单元床面高程一次性改为 0 m，同时保持水深和水量不变。阈值与 level 使用同一基准。时间步会在闸门阈值跨越时刻截断，manifest 记录目标单元数和是否已开闸。

冷启动时所有单元水深均为 0，包括低于 0 m 的床面和地类 8；水经配置边界进入，降雨只在正式模拟开始。预热持续 `warmup_duration` 秒，各边界保持 `start` 时刻的值，无降雨，再由同一引擎连续完成正式模拟。预热时长应通过蓄水与代表点水深的稳定过程实测确定。`warmup_solver(domain_cfg, force_cfg)` 对已预处理的计算域只执行预热，到 `start` 停止。

## 运行二维求解器

```bash
uv run python main.py --domain path/to/domain.json --force path/to/force.json
```

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.core.solver_compact import run_simulation

run_simulation(load_domain_config("path/to/domain.json"),
               load_force_config("path/to/force.json"))
```

每次运行使用新的 `domain_dir`，已有 manifest、收支记录或输出时直接报错，不覆盖。命令行入口在预处理前创建 manifest，预处理和输入载入失败均记录为失败运行。需要单独预处理时，仍可调用 `preprocess(domain_cfg, force_cfg)` 后再调用 `solver(domain_cfg, force_cfg)`。

预处理在 `domain_dir/preprocessed` 生成 `ne.fdb`、`ns.fdb` 和 `boundary.fdb`。求解器在 `domain_dir/output/bed.fdb` 一次写入起始工作床面。`output/frames/depth_<elapsed_s>.fdb` 仅含 F32 水深，在初始时刻、此后每 900 s 以及终点输出。`output/stations.csv` 为配置的站点记录 `depth`、`level`、`u`、`v`。`output/extrema.fdb` 保存逐步累计的 `max_depth`、其首次出现的双 U32 时钟字、`max_speed`。极值统计正式模拟阶段；独立预热运行则统计预热阶段。`output/state_start.fdb` 是正式起点的检查点；`output/state_final.fdb` 是完整终点检查点。检查点包含工作床面、水深、边流量 q、Horton 累计降雨时间与下渗能力、时钟、极值和精确收支累计状态。预热与正式模拟共用同一时钟和输出格式。预热帧的经过秒数为负。

网格载入要求正的正方形单元，以满足基于边长的 CFL 尺度。空间水位 MAE 在没有共同湿像元时为 NaN，表示无定义。

闸门打开时床面可能改变。`manifest.json` 的每条 `bed_events` 记录时刻、全部受影响单元 ID 和对应的新床面值。以 `bed.fdb` 为起点回放事件即可恢复任一帧的工作床面；水面高程等于工作床面加水深。

`domain_dir/mass_balance.csv` 每行记录累计名义雨量、实际下渗、各 discharge 边界实际施加的有符号体积、各 level 边界有符号水量变化、`requested_discharge:<name>_m3`、`bed_clamp_nonlevel_m3` 与 `bed_clamp_level_m3`、有符号 `pipe_exchange_m3`（管网流入地表为正）、蓄水量和残差。requested 列积分指定的 Q；`discharge:<name>_m3` 使用 donor 限制后实际施加的体积。非 level 单元截断为净补水，参与收支；level 单元截断是随后被指定水深覆盖的中间诊断量，不计净源项。level 低于床面时水深为 0。开闸保持水深和水量，`gate_bed_m3` 为 0。残差等于当前蓄水减去初始蓄水和已计有符号源项。`relative_residual` 的分母取初始蓄水、累计雨量加正的累计净输入、累计非雨收支项绝对量三者最大值；分母为 0 时留空。

正性限制对每个单元只计算一个 donor 因子：可用水量为步初水深对应存水加正显式源项，统一约束所有外流边和负显式源项。共享边只按供水单元因子缩放一次，两侧单元读取同一通量；同一步流入边的水不能用于支付流出。指定 level 的单元拥有无限 reservoir，在步末设为指定水位。CFL 沿用原有面水深下限 `max(h, 0.01)`，在单元更新后读取当前水深与更新后的 q，真实外边不参与。真实 CFL 小于 `1e-4` s 时明确失败，报告控制边、两格水深、面水深与面流速，不反向放大步长。

设备时钟用两个 U32 字表示，以 `2^-40` s 为量子，以预热起点作为非负原点。设备批量推进，按强迫事件和输出时刻分段。时钟能保留长时运行经过时间的精度，但 F32 水深、流量等主状态仍有舍入。DeviceBudget 保留 F32 诊断项的精确整数累计，主机合并后再转为 F64 写入账本。名义降雨另在 CPU 以 F64 积分。`storage_depth_one_ulp_volume_m3` 单独报告 F32 水深状态一个 ULP 对应的体积尺度，与实测残差分开。

通量与雨深增量先汇总到独立的 F32 设备字段，再更新库存；全局阶段屏障固定批推进前后的加法顺序，不添加补偿水量。

闸门按输入序列的首次上穿阈值事件开启，与强迫共用定点时钟。零时间床面事件及相应水位边界纠正先于输出和下一物理步处理，随后重算 CFL；不执行 dt=0 的物理 tick。从独立预热检查点进入正式模拟时，在经过秒数 0 重置极值。

`domain_dir/step_statistics.csv` 每个输出时刻一行，包含标准步的 `standard_dt_min_s`、`standard_dt_max_s`、`standard_dt_mean_s` 和 `standard_steps`，事件短步的步数、最小值和累计时长，以及最严格 CFL 边和对应两格水深、面水深、面流速。在两个 `yield_step` 时刻之间遇到帧时，也会写入收支、站点和步长统计行。`manifest.json` 包含 git 提交与 dirty 状态、完整配置、输入哈希、模拟和墙钟时间戳、步数与 dt 摘要、预处理/预热/正式模拟耗时、收支回传次数、后端和设备、末步收支及峰值显存（无法测量时为 `unavailable`）。

## 耦合模拟

coupled 驱动与 compact 共用同一水深引擎、强迫载入、时钟、收支与 manifest。管网配置提供 `inp`、`pipe_dir`、正整数秒 `coupling_interval`（默认 600）、`exchange_timeout` 和 `weak_dist_thresh`：

```json
{
  "inp": "path/to/network.inp",
  "pipe_dir": "path/to/new-run/pipe",
  "coupling_interval": 600,
  "exchange_timeout": 600.0,
  "weak_dist_thresh": 50.0
}
```

```bash
uv run python main.py --domain path/to/domain.json --force path/to/force.json --pipe path/to/pipe.json
```

管网交换有独立的源项场，与 discharge 相加后写入 `ssq_t`。共享单元的排水按不同 junction 关系分配，outfall 仅设置水位、不扣除地表水。一个 SWMM actor 完整推进每个交换窗，预热照常推进，在正式起点连续接续、不重启；耦合起点须精确到整秒。INP 必须声明 `FLOW_UNITS CMS`，使流量与水头单位为 m³/s 和 m。运行时 INP、报告和输出写入 `pipe_dir`，原 INP 只读。manifest 包含管网配置、输入哈希与真实 SWMM 时钟和交换窗记录。交换公式保留既有的 3 m 溢流压头上限及 0.5 松弛系数。

## 后端自检

自检涵盖 10,000 m 高床面上的小雨、保持水量的干湿闸门开闸，以及检查点接续。每条命令使用新输出目录，写入 `selfcheck.json`、manifest 与收支账本。指定后端须实际可用，已禁用向 CPU 回退。

```bash
uv run python tools/solver_selfcheck.py --arch cpu --output path/to/new-check-cpu
uv run python tools/solver_selfcheck.py --arch metal --output path/to/new-check-metal
uv run python tools/solver_selfcheck.py --arch cuda --output path/to/new-check-cuda
```

CUDA 设备名通过驱动的 `nvidia-smi` 命令读取，该命令须在 PATH 中。

## 接续运行

将 `restart_state` 指向完整的 `output/state_final.fdb` 检查点，使用新的 `domain_dir`，保持 `start`、网格、闸门及物理参数一致，并将 `warmup_duration` 设为 0。`duration` 可以延长。恢复时刻晚于本次终点时，在写结果文件前报错；等于终点时以零步完成。边界名称、类型、顺序及选中边集合的哈希必须一致。下一 CFL 候选及控制边诊断由恢复的水深、床面和 q 在同一次归约中求出，不再保存下一 dt 字段。求解器恢复工作床面、水深、q、Horton 状态、双字时钟、极值与精确设备收支累计，不重新施加边界初值。边界与降雨输入须覆盖延长后的正式时窗。地表检查点不含 SWMM actor 状态；接续耦合管网运行会明确报错。

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
draw_hydrograph(domain_cfg, "S4", quantity="level")
```

所有水文过程线函数都必须指定 `quantity="level"` 或 `quantity="depth"`。观测文件必须恰好有一列匹配的 Water Level 或 Water Depth；另一种物理量会被拒绝。图例、坐标轴和模拟采样使用同一物理量。

这些函数读取求解器生成的预处理网格、水深帧或极值；水面高程对比根据 manifest 回放床面事件。

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
| `Node`, `PipeTopo` | 耦合用 SWMM 节点和管网到二维单元拓扑记录。 |
| `IndexLike`, `U8Value`, `F32Value` | 小型类型化辅助表。 |

二维网格数据中的索引 `0` 是虚拟或哨兵元素/边，真实网格遍历从索引 `1` 开始。

## 代码结构

```text
src/nh_flood_2d/
  input/            Pydantic 配置模型和 JSON 加载器
  preprocess/       计算域、强迫和管网预处理
  core/
    solver_compact.py  地表求解及检查点接续
    clock.py, device_budget.py, output.py
    coupled/        二维一维耦合求解器驱动和交换逻辑
  output/           洪水栅格、视频、水文过程线和对比图
  schema/           fastdb4py Feature 定义
  dem/              DEM 融合和 TIN 工具
  util/             Taichi 初始化和计时辅助函数
tools/              独立维护和转换脚本
examples/           DEM 融合运行示例
docs/               设计说明和使用文档
```
