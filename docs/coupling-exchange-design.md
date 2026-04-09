# 1D-2D 耦合交换机制设计文档

> **分支**: `feature/stepping-coupling`
> **日期**: 2026-04-09
> **状态**: 设计完成，待实施

---

## 目录

1. [问题背景](#1-问题背景)
2. [根因分析](#2-根因分析)
3. [参考实现对比](#3-参考实现对比)
4. [目标方案：SurDepth=1000 + 双向反馈](#4-目标方案surdepth1000--双向反馈)
5. [实现路径](#5-实现路径)
6. [溢流公式选择](#6-溢流公式选择)
7. [风险与注意事项](#7-风险与注意事项)

---

## 1. 问题背景

### 1.1 系统架构

本项目实现了 2D 浅水方程（nh-flood-2d）与 1D 管网模型（SWMM/pyswmm）的耦合。
两个模型通过 multiprocessing 进程间通信，以固定时间窗口（coupling_interval，默认 600s）
进行数据交换。

```
┌─────────────────────────────────────────────────────────────┐
│                      2D 浅水方程模型                         │
│  (Taichi GPU, solver_compact / flood_2d)                    │
│                                                             │
│  地表水深 h, 流速 u/v, 地面高程 z                             │
│  源汇项 q_source (排水 + 溢流返回)                            │
└────────────┬────────────────────────────────┬───────────────┘
             │ drainage (地表→管)             │ overflow (管→地表)
             │ outfall_stage (水位→出水口)     │ flood_return (溢流)
             ▼                                ▲
┌─────────────────────────────────────────────────────────────┐
│                      1D SWMM 管网模型                        │
│  (pyswmm continuous stepping)                               │
│                                                             │
│  节点水头 HEAD, 溢流 FLOOD, 管道流量                          │
│  1083 junctions + 128 outfalls                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心问题："水爆炸"

在 stepping coupling 方案实施后，模拟运行一段时间后地表会出现大面积积水（"到处是水"），
即使水量收支（NET）始终为负（管网在净排水）。

经过系统性调试（5 次修复迭代），我们发现了两个独立的根因：

| # | 根因 | 修复方案 | 提交 |
|---|------|---------|------|
| 1 | Outfall TOTAL_INFLOW 返回2D造成水循环 | 零化 outfall 返回 | `e429672` |
| 2 | SWMM 内部在 `HEAD > rim` 时就报告 flooding，管内水被"丢弃" | 待实施（本文档） | — |

**第一个根因（已修复）** 消除了水爆炸的直接症状。
**第二个根因（本文档）** 是更深层的物理正确性问题——需要实施 SurDepth=1000 + 双向反馈。

---

## 2. 根因分析

### 2.1 SWMM 的 Flooding 机制

SWMM 的 junction flooding 判断逻辑：

```
if HEAD > rim + SurDepth:
    flood_rate = f(HEAD - rim - SurDepth)   # 水从节点移除
    节点水量减少 flood_rate × dt
```

- `rim` = 地面高程 = invert_elevation + full_depth
- `SurDepth` = 超载深度（允许水头超过 rim 多少才算 flooding）
- 当 `SurDepth = 0` 时，`HEAD > rim` 就触发 flooding

### 2.2 耦合场景下的物理矛盾

在 1D-2D 耦合中，检修井(junction)连接着地下管网和地表。真实物理：

```
            ┌── 实际水面 = rim + h_2d ──┐
            │                           │
   2D地表   │   h_2d = 地表积水深度      │   2D地表
════════════╪═══════════════════════════╪════════════
            │                           │
   管道水头  │   HEAD (SWMM计算)         │
            │                           │
   管底     └── invert_elevation ───────┘
```

- **HEAD > rim + h_2d** → 管道水头超过实际水面 → 水从管涌出 ✓
- **rim < HEAD < rim + h_2d** → 管道已超过井口，但未超过水面 → **不应溢流**
- **HEAD < rim** → 管道水头在地面以下 → 无溢流 ✓

但 SWMM 不知道 2D 模型的存在，当 `SurDepth ≈ 0` 时：
- `HEAD > rim` 就触发 flooding
- SWMM **内部移除**这部分水（管内水量减少）
- 这些水既不在管里，也不（一定）在地表 → **质量不守恒**

### 2.3 三种处理方式对比

#### 方式 A：参考代码的动态 SurDepth（pipe_NH801.py）

```python
# 每个交换窗口执行：
def update_junction_surdepth():
    for j in range(n_junctions):
        SurDepth[j] = h_2d[j]   # 设为当前2D水深
    # → SWMM 只在 HEAD > rim + h_2d 时才 flooding
    # → 物理正确，质量守恒
```

**问题**：SWMM API 禁止在运行时修改 SurDepth（`node_set_parameter()` 报错
`Simulation Already Started`）。

#### 方式 B：当前的 Post-filter（已实施）

```python
# SWMM 在 HEAD > rim 时已经移除了水
flood = solver.node_get_result(idx, NodeResult.FLOOD)
# 我们只返回 "有效" 部分给 2D
if HEAD > rim + h_2d:
    flow = flood * (HEAD - rim - h_2d) / (HEAD - rim)
else:
    flow = 0.0
```

**问题**：
- SWMM 内部已经把水从管网移走了 → 管道水头被人为压低
- 被移除但未返回2D的水 = 系统水量损失
- 管网压力偏低，排水能力计算不准确

#### 方式 C：SurDepth=1000 + 双向反馈（目标方案）

```
.inp 中设所有 junction 的 SurDepth = 1000
→ SWMM 永远不报告 flooding（HEAD 不可能超过 rim + 1000）
→ 管内水量完全守恒
→ 由耦合代码自主计算溢流 + 通过负流量"抽血"
```

| 指标 | 方式 A | 方式 B | 方式 C |
|------|--------|--------|--------|
| 质量守恒 | ✓ | ✗ 水被丢弃 | ✓ |
| 管内压力 | 正确 | 偏低 | 正确 |
| API兼容 | ✗ 运行时不可设 | ✓ | ✓ |
| 实现复杂度 | — | 低 | 中 |

---

## 3. 参考实现对比

### 3.1 PIPE 参考仓库的交换流程

参考代码位于 `Pipe_Flood_fastdb` 仓库：
- 2D 侧：`Flood_new801.py`
- 1D 侧：`pipe_NH801.py`

#### 参考的 1D 初始化（pipe_NH801.py L314-328）

```python
# 1. 清零 .inp 中的 [INFLOWS] baseline
update_start()

# 2. 初始化所有 junction 的 SurDepth = 0
initialize_junction_surdepth()

# 3. 设置 304 个特殊节点 SurDepth = 1000（永不溢流）
update_surdepth_500()
```

#### 参考的每窗口交换（Flood_new801.py L695-769）

```python
# 每个交换窗口：
# 1. 更新动态 SurDepth = 2D 水深（除 304 个特殊节点外）
update_junction_surdepth()   # SurDepth[j] = h_2d[j]
update_surdepth_500()        # 特殊节点保持 1000

# 2. 计算排水量（secondary + primary cells）
for i in range(n_nodes):
    # secondary cells: SET (=), no /nc
    q = min(0.85·π·0.8·d^1.5, d·area/ci)
    # primary cells: accumulate (+=), /nc, node_type gating

# 3. 读取 1D 结果
# junction → FLOOD 流量 (溢流返回2D)
# outfall  → Total Flow Volume (从 .rpt 读取，也返回2D)

# 4. 合并到 q_source
q_source[ei] = -drainage + flood_return
```

#### 参考与当前代码的对齐结果

| 项目 | 参考 (PIPE) | 当前代码 | 一致？ |
|------|------------|---------|--------|
| 排水公式 | `min(0.85·π·0.8·d^1.5, d·area/ci)` | 相同 | ✓ |
| Secondary cells | SET (=), no /nc, all nodes | 相同 | ✓ |
| Primary cells | += , /nc, node_type gating | 相同 | ✓ |
| data_dict flow | primary_drain + total_flow[i] | 相同 | ✓ |
| level 语义 | outfall→WSE, junction→depth | 相同 | ✓ |
| 交换间隔 | 2D=300s, 1D=600s | 两侧=600s | ≈等价 |
| 排水除数 | /300 (hardcoded) | /ci (600) | ✓ 窗口总量一致 |
| SurDepth | 动态设置=2D水深 | post-filter | **✗ 关键差异** |
| Outfall 返回 | 返回2D | 已零化 | ✗ 不同策略 |

### 3.2 .inp 文件中的 SurDepth 分布

```
总节点数：1083 junctions + 128 outfalls = 1211

SurDepth 分布：
  - 304 个 junction: SurDepth = 1000  (特殊节点，永不溢流)
  - 77  个 junction: SurDepth = 0.0
  - 702 个 junction: SurDepth ≈ 0.002 ~ 0.02 (近似为0)
  - 128 个 outfall:  无 SurDepth 属性
```

---

## 4. 目标方案：SurDepth=1000 + 双向反馈

### 4.1 核心思想

1. **屏蔽 SWMM 的 flooding 机制**：将所有 junction 的 SurDepth 设为 1000m，
   使 SWMM 永远不报告 flooding，管内水量完全守恒。
2. **自主计算溢流**：耦合代码根据 `HEAD vs (rim + h_2d)` 判断溢流方向和流量。
3. **双向反馈（"抽血"机制）**：将溢流量通过 `node_set_total_inflow` 的负流量
   从 SWMM 中移除，确保质量守恒。

### 4.2 交换通量 Q_ex 的定义

```
给定：
  HEAD   = SWMM 计算的节点水头 (绝对高程)
  rim    = 节点地面高程 = invert_elevation + full_depth
  h_2d   = 2D 模型在该位置的地表积水深度
  WL_2d  = rim + h_2d = 实际水面绝对高程

交换通量 Q_ex（正值 = 从管到地表，负值 = 从地表到管）：

  if HEAD > rim + h_2d:
      Q_ex = +f(HEAD, rim, h_2d)     # 溢流：管 → 地表
  elif HEAD < rim and h_2d > 0:
      Q_ex = 0                       # 已由 drainage 处理
  else:
      Q_ex = 0                       # 无交换
```

注意：`HEAD < rim + h_2d` 方向的水流（地表→管）已经由 `compute_drainage()` 处理，
不需要在 Q_ex 中重复。

### 4.3 双向反馈的完整闭环

每个交换窗口的数据流：

```
Window N:

  ┌─ 2D 侧 ──────────────────────────────────────────────────────┐
  │                                                               │
  │  ① receive_from_1d()                                          │
  │     → 获取上一窗口的 1D 结果：HEAD_{N-1} for each junction    │
  │                                                               │
  │  ② compute_drainage()                                         │
  │     → 从当前 2D 状态计算 drain_N (m³/s per node)              │
  │     → q_source[ei] = -drain (排水，负源项)                     │
  │                                                               │
  │  ③ compute_overflow() [新增]                                   │
  │     → 用 HEAD_{N-1} vs (rim + h_2d) 计算 Q_ex_{N-1}          │
  │     → q_source[ei] += Q_ex (溢流返回，正源项)                  │
  │                                                               │
  │  ④ apply_sources()                                             │
  │     → ssq_t.from_numpy(q_source)                              │
  │                                                               │
  │  ⑤ send_to_1d()                                               │
  │     → 发送 net_inflow = drain_N - Q_ex_{N-1}                 │
  │                                                               │
  │       drain > Q_ex → 正值 → 管网净接收水                      │
  │       drain < Q_ex → 负值 → 管网净失去水（"抽血"）             │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

  ┌─ 1D 侧 ──────────────────────────────────────────────────────┐
  │                                                               │
  │  ⑥ solver.node_set_total_inflow(idx, net_inflow)              │
  │     → SWMM 用此值作为侧向入流（支持负值）                     │
  │     → 负值 = 从节点抽水 → 降低 HEAD                           │
  │                                                               │
  │  ⑦ sim._model.swmm_step()                                     │
  │     → SWMM 步进，由于 SurDepth=1000，不会内部 flooding        │
  │     → 管内水量精确守恒                                        │
  │                                                               │
  │  ⑧ 读取结果：HEAD, DEPTH → 发回 2D                            │
  │     → 不再读取 FLOOD（因为 SurDepth=1000 下 FLOOD 始终=0）     │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 4.4 SWMM API 对负流量的支持

经查证，`node_set_total_inflow` 底层调用 `swmm_setNodeInflow`，
**EPA SWMM 官方文档确认支持负值**：

- 正值 = 水进入节点（lateral inflow）
- 负值 = 水从节点抽出（lateral extraction）
- SWMM 内部会限制抽取量不超过节点可用水量（不会出现负体积）

这意味着不需要额外的保护逻辑——SWMM 自身保证数值安全。

### 4.5 与当前架构的兼容性

当前 lag-1 异步耦合协议不需要改变：

```
当前：  send_to_1d(data_dict)          → data_dict[name]['flow'] = drainage
目标：  send_to_1d(data_dict)          → data_dict[name]['flow'] = drainage - Q_ex
                                         ↑ 唯一变化：net flow 替代 drainage
```

1D 侧的 `node_set_total_inflow(idx, flow)` 代码**不需要修改**，
只是接收到的 `flow` 值从"纯排水"变成了"净流量"。

---

## 5. 实现路径

### 5.1 Step 1：修改 .inp 预处理 — 设置 SurDepth=1000

**文件**：`src/nh_flood_2d/core/coupled/pipe_1d.py` 的 `_clear_inp_inflows` 之后

新增函数 `_set_all_surdepth(inp_path, value=1000)`:

```python
def _set_all_surdepth(inp_path: str, value: float = 1000.0) -> None:
    """Set SurDepth to a large value for ALL junctions.

    Prevents SWMM from internally flooding — the coupled exchange
    code handles overflow via bidirectional feedback instead.
    """
    # 解析 [JUNCTIONS] 段
    # 每行格式: Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
    # 将 SurDepth 列替换为 value
```

在 `run_1d_pipe()` 中调用：

```python
_clear_inp_inflows(inp_runtime)
_set_all_surdepth(inp_runtime, 1000.0)   # ← 新增
```

### 5.2 Step 2：新增溢流计算 — compute_overflow()

**文件**：`src/nh_flood_2d/core/coupled/exchange.py`

```python
def compute_overflow(
    flood_return: dict,          # 1D 返回的 {name: {'level': HEAD, ...}}
    junction_rim: dict,          # {name: rim_elevation}
    h_2d_at_nodes: dict,         # {name: 2D水深}
    node_names: list,
    node_is_outfall: np.ndarray,
) -> dict:
    """Compute overflow Q_ex based on HEAD vs (rim + h_2d).

    Returns {name: Q_ex} where Q_ex > 0 means pipe → surface.
    """
    overflow = {}
    for i, name in enumerate(node_names):
        if node_is_outfall[i]:
            overflow[name] = 0.0
            continue

        head = flood_return.get(name, {}).get('level', 0.0)
        rim = junction_rim.get(name, 0.0)
        h_2d = h_2d_at_nodes.get(name, 0.0)

        if head > rim + h_2d:
            # 溢流公式（见第6节讨论）
            overflow[name] = overflow_formula(head, rim, h_2d, ...)
        else:
            overflow[name] = 0.0

    return overflow
```

### 5.3 Step 3：修改 apply_sources() — 加入溢流

**文件**：`src/nh_flood_2d/core/coupled/exchange.py`

当前 `apply_sources()` 从 `flood_return` 中读取 `flow`（SWMM FLOOD 结果）。
改为从 `compute_overflow()` 的结果中读取 Q_ex：

```python
def apply_sources(
    q_source: np.ndarray,
    overflow: dict,           # ← 改为 overflow（不再是 flood_return）
    ssq_t,
    primary_ei: np.ndarray,
    name_to_idx: dict,
) -> None:
    for name, q_ex in overflow.items():
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        q_source[ei] += q_ex   # 正值 = 溢流到地表

    q_source[0] = 0.0
    ssq_t.from_numpy(q_source)
```

### 5.4 Step 4：修改 send_to_1d() — 净流量替代纯排水

**文件**：`src/nh_flood_2d/core/coupled/exchange.py` 或 `flood_2d.py` 交换块

在 `compute_drainage()` 返回的 `data_dict` 中，将每个节点的 flow 调整为净流量：

```python
# 在发送前合并
for name in data_dict:
    if name in overflow:
        # net_inflow = drainage - overflow
        data_dict[name]['flow'] -= overflow[name]
```

1D 侧代码不需要改动——它只是用 `node_set_total_inflow(idx, flow)` 设置收到的值。

### 5.5 Step 5：简化 pipe_1d.py 的结果读取

**文件**：`src/nh_flood_2d/core/coupled/pipe_1d.py`

由于 SurDepth=1000，SWMM 的 FLOOD 始终为 0。1D 侧不再需要读取 FLOOD、
做 surcharge post-filter 等复杂逻辑。简化为只返回 HEAD 和 DEPTH：

```python
# 简化后的结果读取
if is_outfall[i]:
    level = solver.node_get_result(swmm_idx, NodeResult.HEAD)
    flow = 0.0   # outfall 不返回水到 2D
else:
    head = solver.node_get_result(swmm_idx, NodeResult.HEAD)
    depth = solver.node_get_result(swmm_idx, NodeResult.DEPTH)
    level = head     # ← 改为返回绝对水头（2D侧需要用于 overflow 计算）
    flow = 0.0       # ← 不再从 SWMM 读取 flow，由 2D 侧自主计算
```

**注意**：1D 返回的 `level` 语义从 "depth" 变为 "HEAD"（绝对水头），
2D 侧的 `compute_overflow()` 需要绝对水头来与 `rim + h_2d` 比较。

### 5.6 Step 6：提取 junction_rim 到共享预计算

当前 `junction_rim` 在 `pipe_1d.py` 中预计算。改为在 `prepare_pipe` 阶段
存入 `pipe.fdb`，让 2D 侧也能访问（因为 overflow 计算在 2D 侧执行）。

或者更简单地：在 2D 侧启动时从 ne.fdb 读取地面高程作为 rim 的近似。

### 5.7 文件变更汇总

| 文件 | 变更 |
|------|------|
| `pipe_1d.py` | 新增 `_set_all_surdepth()`；简化结果读取（删除 FLOOD/surcharge 逻辑）；level 语义改为 HEAD |
| `exchange.py` | 新增 `compute_overflow()`；修改 `apply_sources()` 和 `send_to_1d()` |
| `flood_2d.py` | 交换块中增加 overflow 计算步骤；传递 h_2d 给 overflow |
| `prepare_pipe.py` | 可选：将 junction_rim 存入 pipe.fdb |

---

## 6. 溢流公式选择

### 6.1 候选公式

当 `HEAD > rim + h_2d` 时，需要计算从管到地表的溢流量 Q_ex。

#### 方案 A：孔口出流公式（Orifice）

```
Q_ex = C_d · A_manhole · √(2g · Δh)

其中：
  C_d       = 出流系数 ≈ 0.6
  A_manhole = 检修井口面积 (m²)，可从 .inp 或假设值获取
  g         = 9.81 m/s²
  Δh        = HEAD - (rim + h_2d)  (有效水头差)
```

**优点**：物理意义明确，是工程中标准的管口溢流公式。
**缺点**：需要检修井口面积参数（.inp 中可能没有）。

#### 方案 B：堰流公式（Weir）

```
Q_ex = C_w · L_weir · Δh^1.5

其中：
  C_w     = 堰流系数 ≈ 1.7
  L_weir  = 井口周长 (m)
  Δh      = HEAD - (rim + h_2d)
```

**优点**：适合浅溢流（Δh 较小时更准确）。
**缺点**：同样需要几何参数。

#### 方案 C：简化线性公式

```
Q_ex = α · Δh

其中：
  α  = 经验系数 (m²/s)
  Δh = HEAD - (rim + h_2d)
```

**优点**：简单，易于调试。
**缺点**：物理意义弱。

#### 方案 D：复用 SWMM 的排水公式（推荐起步方案）

```
Q_ex = min(0.85 · π · 0.8 · Δh^1.5,  Δh · A_cell / ci)

其中：
  Δh     = HEAD - (rim + h_2d)
  A_cell = 主格子面积 esl²
  ci     = coupling_interval
```

**优点**：与排水公式对称，无需额外参数，行为可预测。
**缺点**：不是严格物理公式，但对于原型验证足够。

### 6.2 推荐策略

1. **原型阶段**：使用方案 D（复用排水公式），快速验证双向反馈机制的正确性。
2. **精细化阶段**：切换到方案 A（孔口公式），从 .inp 提取或估算井口面积。

---

## 7. 风险与注意事项

### 7.1 lag-1 延迟

overflow 使用的是上一窗口的 HEAD_{N-1}，存在一个 coupling_interval 的延迟。

- coupling_interval = 600s 时，延迟为 10 分钟
- 对缓变流场景可接受
- 若需提高精度，可缩短 coupling_interval 到 300s（与参考一致）

### 7.2 SWMM 负流量的边界行为

当 `node_set_total_inflow(idx, negative_value)` 时：

- SWMM 内部限制抽取量 ≤ 节点可用水量
- 不会出现负体积
- 但如果 overflow 估计过大，可能导致管道快速排空 → 下一窗口 HEAD 骤降
- **建议**：初期添加日志监控 net_inflow 的负值占比

### 7.3 Outfall 节点处理

Outfall 的 SurDepth 不在 [JUNCTIONS] 段，不受 `_set_all_surdepth` 影响。
Outfall 的处理策略不变：

- 设置 outfall stage = 2D 水面高程（`outfall_set_stage`）
- 不返回 TOTAL_INFLOW 到 2D（outfall 是系统出口）

### 7.4 特殊节点（原 SurDepth=1000 的 304 个）

参考代码中 304 个节点的 SurDepth=1000 意味着它们"永不溢流"。
在新方案中所有节点都是 SurDepth=1000，所以这个特殊处理自动消失。

但需要确认：这 304 个节点是否在物理上确实不应该溢流？
如果是，需要在 `compute_overflow()` 中标记它们，跳过溢流计算。

**建议**：从原始 .inp 中提取这 304 个节点的 ID，存入 pipe.fdb 作为 "no_flood" 标记。

### 7.5 drainage 公式与管内水头的关系

当前排水公式 `min(0.85·π·0.8·d^1.5, d·area/ci)` 只考虑了地表水深 `d`，
没有考虑管内水头对排水的抵抗作用。

当管内水头很高（接近 rim）时，地表水即使很深也难以通过重力排入管道。
这是一个独立的改进点，不在本文档范围内，但值得后续关注。

### 7.6 水量守恒验证方法

实施后应添加水量诊断：

```python
# 每个窗口输出：
total_drain    = Σ drainage_i        # 地表→管 (正值)
total_overflow = Σ Q_ex_i            # 管→地表 (正值)
net_exchange   = total_drain - total_overflow  # 正=管净接收
# 长期应 ≈ 降雨输入 - 出水口排出
```

---

## 附录：当前代码的调试历程

### 修复迭代记录

| # | 问题 | 修复 | 提交 |
|---|------|------|------|
| 1 | `sim._isStarted` 不可设 | 使用 `sim.start()` | `8f9f077` |
| 2 | `node_set_parameter` 运行时禁止 | 改用 `node_set_total_inflow` | `3aee06b` |
| 3 | .inp baseline inflows 双重计算 | `_clear_inp_inflows()` | `bc02427` |
| 4 | SWMM flooding 阈值不含2D水深 | Post-filter surcharge correction | `b0d9b1f` |
| 5 | Outfall TOTAL_INFLOW 水循环 | 零化 outfall 返回 | `e429672` |

### 水量诊断数据（修复#5后，70+窗口）

```
窗口  排水(m³/s)  溢流(m³/s)  出水口(m³/s)  NET(m³/s)
w=0    683          0           0            -683
w=10   129          2.0         0            -127
w=20    97          1.2         0             -96
w=30   280          1.4         0            -279
w=40   549          4.2         0            -545
w=50   337          5.8         0            -331
w=60   360         15.5         0            -344
w=70   327         10.9         0            -316
```

- NET 始终为负（管网净排水）
- 溢流仅占排水量的 2-5%
- 模拟稳定运行 42,600s 无水爆炸
