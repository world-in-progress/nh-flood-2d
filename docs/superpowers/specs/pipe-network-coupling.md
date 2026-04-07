# 管网耦合方案设计规范

> 目标：为 `solver_compact` 提供基于 `fastdb4py` 的 1D–2D 并行管网耦合能力，
> 参考 `Pipe_Flood_fastdb` 工程的实现思路，适配本仓库的数据体系与模块结构。

---

## 1. 概述

本方案在不修改 `solver_compact.py` 的前提下，新增：

| 组件 | 路径 | 说明 |
|---|---|---|
| `PipeConfig` | `src/nh_flood_2d/input/pipe.py` | 管网配置 Pydantic 模型 |
| `prepare_pipe()` | `src/nh_flood_2d/preprocess/pipe.py` | 节点–网格拓扑预处理，输出 `pipe.fdb` |
| `solver_coupled()` | `src/nh_flood_2d/core/solver_coupled.py` | 并行耦合主函数 |
| `_run_2d()` | （同上文件，内部函数）| GPU 侧进程，含交换 hook |
| `_run_1d_pipe()` | （同上文件，内部函数）| CPU 侧 SWMM 进程 |

**已有文件需小幅修改：**

| 文件 | 修改内容 |
|---|---|
| `schema/feature.py` | 新增 `PipeTopo` Feature 类 |
| `input/__init__.py` | 导出 `PipeConfig`, `load_pipe_config` |
| `preprocess/__init__.py` | 导出 `prepare_pipe` |

---

## 2. FDB 数据模式

### 2.1 新增 Feature 类（`schema/feature.py`）

```python
class PipeTopo(fdb.Feature):
    """
    CSR-like 节点-网格拓扑（只存 ei，node_id 由 topo_ptr 隐含）。

    索引约定（与 §6.2 保持一致）：
      - Node 表按 0-based 行序存储，共 n_nodes 条；
      - topo_ptr: 0-based CSR offset，长度 = n_nodes + 1；
        节点 i（0..n_nodes-1）对应 topo_ei[ topo_ptr[i] : topo_ptr[i+1] ]；
      - topo_ei / primary_ei 中的 ei 值为 1-based（0 = 无效哨兵，同全仓库约定）；
      - PipeTopo 中**不包含** primary_ei 本身，只存弱相关的额外网格元素。
    """
    ei: fdb.U32        # 对应的 2D 网格元素索引（1-based，0 为无效）
```

### 2.2 `pipe.fdb` 表结构

| FeatureClass | 表名 | 内容 |
|---|---|---|
| `Node` | `Node` | SWMM 节点信息：index, name, x, y, is_outfall |
| `IndexLike` | `'node_primary_ei'` | 每个节点的**主**网格元素索引（1:1，与 Node 等长） |
| `IndexLike` | `'node_count_per_ei'` | 每个 2D 元素上覆盖的节点数（与 ne 等长，用于流量归一化） |
| `PipeTopo` | `PipeTopo` | 弱相关网格展开列表，**不含 primary_ei**（CSR data，只存 ei） |
| `IndexLike` | `'topo_ptr'` | CSR offset，长度 = n_nodes + 1，**0-based**，节点 i 对应 `[topo_ptr[i]:topo_ptr[i+1]]` |
| `IndexLike` | `'node_count'` | 单元素：节点总数 n_nodes |

---

## 3. `PipeConfig`（`src/nh_flood_2d/input/pipe.py`）

```python
class PipeConfig(BaseModel):
    inp: str                              # SWMM .inp 文件路径
    pipe_dir: str                         # pipe.fdb 等产物的输出目录

    coupling_interval: float = 300.0      # 2D-1D 数据交换间隔（秒）
    exchange_timeout: float  = 300.0      # 进程间等待超时（秒）
    weak_dist_thresh: float  = 50.0       # 弱相关搜索半径（m）

    @property
    def pipe_fdb(self) -> str:
        return str(Path(self.pipe_dir) / 'preprocessed' / 'pipe.fdb')

    @property
    def inp_runtime(self) -> str:
        """运行时临时 .inp 副本（避免污染原始文件）"""
        return str(Path(self.pipe_dir) / 'runtime.inp')

    @property
    def hotstart_dir(self) -> str:
        return str(Path(self.pipe_dir) / 'hotstart')

def load_pipe_config(config_path: str) -> PipeConfig:
    return PipeConfig.model_validate_json(Path(config_path).read_text())
```

---

## 4. `prepare_pipe()`（`src/nh_flood_2d/preprocess/pipe.py`）

### 4.1 算法流程

```
prepare_pipe(pipe_cfg, domain_cfg)
│
├─ 1. 解析 SWMM .inp → 节点列表 [{name, x, y, is_outfall}]
│      遍历 [COORDINATES] 段；is_outfall = name.lower().startswith('outfall')
│
├─ 2. 加载 ne.fdb → 获取 xe, ye, ze, type（网格坐标+高程）
│      加载 ns.fdb → 获取 nss.column.x/y（用于半边长计算）
│      加载 isl_ptr_l / isl_ptr_b → 计算每网格的 half_width/height
│      → 得到每网格的 [x_min, x_max, y_min, y_max]
│
├─ 3. 强相关匹配（CPU loop，精确包围盒）
│      对每个非出口节点：找 ze[k] > high_relation_elevation 且坐标落入网格边界的 k
│      对每个出口节点：仅检查坐标落入网格边界（无高程限制）
│      → 记录 primary_ei[node_id] = k，标记 pd_grid[k] = 1
│
├─ 4. 弱相关匹配（Taichi GPU kernel，最近邻）
│      对未被强相关覆盖的网格（pd_grid[k]==0）：
│        找距离 < low_relation_distance 的最近节点
│      → 合并进每个节点的相关网格列表
│
├─ 5. 构建 CSR 拓扑
│      topo_data = 展开的 (node_id, ei) 对列表
│      topo_ptr[i] = 节点 i 的起始偏移
│      node_count_per_ei[k] = 网格 k 上的节点数
│
└─ 6. 写 pipe.fdb
       Node 表 / node_primary_ei / node_count_per_ei / PipeTopo / topo_ptr / node_count
```

### 4.2 关键依赖

- 解析 `.inp` 节点坐标逻辑直接参考 `re_coo.parse_inp_coordinates()`
- 包围盒计算复用 `set_elevation()` 中 `isl_ptr_l/b` → `half_width/height` 模式
- Taichi 初始化复用 `util/ti.py` 的 `init_taichi()`

---

## 5. 耦合架构（`src/nh_flood_2d/core/solver_coupled.py`）

### 5.1 入口函数

```python
def solver_coupled(
    domain_cfg: DomainConfig,
    force_cfg:  ForceConfig,
    pipe_cfg:   PipeConfig,
    start_time_step: int = 0,
):
    import shutil, multiprocessing

    # 复制 .inp 到运行时副本，保留原始文件
    shutil.copy(pipe_cfg.inp, pipe_cfg.inp_runtime)

    # 必须显式使用 spawn，禁止依赖平台默认值。
    # fork 模式下子进程会继承 Taichi 的 _TI_INITIALIZED 标志，导致 GPU 初始化异常。
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    shared = {
        '2d_data':  manager.dict(),    # 2D → 1D：{node_name: {level, flow}}
        '1d_data':  manager.dict(),    # 1D → 2D：{node_name: {level, flow}}
        '2d_ready': manager.Event(),   # 2D 已准备好数据
        '1d_ready': manager.Event(),   # 1D 已准备好数据
        'lock':     manager.Lock(),
        'stop':     manager.Event(),   # 2D 结束时 set，通知 1D 退出
    }

    flood_proc = ctx.Process(
        target=_run_2d,
        args=(shared, domain_cfg, force_cfg, pipe_cfg, start_time_step),
    )
    pipe_proc = ctx.Process(
        target=_run_1d_pipe,
        args=(shared, pipe_cfg),
    )
    flood_proc.start()
    pipe_proc.start()
    flood_proc.join()
    pipe_proc.join()
```

### 5.2 共享内存协议

每次交换周期，双方交换的数据格式：

```python
# 2D → 1D（节点水位 + 流入管网流量 + 实际交换窗口时长）
{
  "SCH1234567": {"level": 0.32,  "flow": 0.015},
  # level 语义：
  #   junction 节点 → 超出地面的水深（h - z，m），用作 surcharge head 输入
  #   outfall  节点 → 绝对水面高程（h，m），用作固定水头边界
  # flow: m³/s（正值=流入管网）
  "Outfall_01": {"level": -0.10, "flow": 0.0},
  "__window_dt__": {"level": 305.2, "flow": 0.0},  # 特殊键：实际交换窗口时长（s）
  ...
}

# 1D → 2D（管网溢出流量）
{
  "SCH1234567": {"level": 1.50,  "flow": 0.008},   # level: SWMM 节点水位；flow: 本窗口溢出增量（m³/s）
  "Outfall_01": {"level": -0.10, "flow": 0.0},
  ...
}
```

---

## 6. `_run_2d()` 设计

### 6.1 与 `solver_compact` 的差异

`_run_2d()` 的核心逻辑与 `solver_compact.solver()` **完全相同**，仅增加以下内容：

**① 启动时额外加载 `pipe.fdb`：**

```python
pipe_fdb = fdb.ORM.load(pipe_cfg.pipe_fdb, from_file=True)
n_nodes = pipe_fdb[IndexLike]['node_count'][0].index       # 节点总数
node_names = [str(pipe_fdb[Node][Node][i].name) for i in range(n_nodes)]  # 显式转 str，防止 fdb.STR 悬空
# 显式 copy：del pipe_fdb 后 FDB 列可能失效，必须持有独立 numpy 数组
node_is_outfall = pipe_fdb[Node][Node].column.is_outfall.copy().astype(bool)
primary_ei = pipe_fdb[IndexLike]['node_primary_ei'].column.index.copy()  # shape (n_nodes,)
# CSR 拓扑（topo_ptr 为 0-based 偏移；topo_ei 中 ei 为 1-based，0 为无效哨兵）
topo_ei   = pipe_fdb[PipeTopo][PipeTopo].column.ei.copy()   # flat element indices
topo_ptr  = pipe_fdb[IndexLike]['topo_ptr'].column.index.copy()  # ptr array
nc_per_ei = pipe_fdb[IndexLike]['node_count_per_ei'].column.index.copy()  # per-element node count
del pipe_fdb
name_to_idx = {name: i for i, name in enumerate(node_names)}
```

**② 主循环新增时间累计变量：**

```python
pipe_exchange_acc = 0.0   # 累计到下次交换的时间
```

**③ 每次 `tick()` 后：**

```python
dt = tick(tide, rainq)
current_time += dt
pipe_exchange_acc += dt

if pipe_exchange_acc >= pipe_cfg.coupling_interval:
    actual_dt = pipe_exchange_acc  # 实际窗口时长（自适应 dt 不整除 coupling_interval）
    _exchange_with_1d(shared, pipe_cfg, actual_dt, h_t, ssq_t, ez_t, esl_t,
                      primary_ei, topo_ei, topo_ptr, nc_per_ei,
                      node_names, node_is_outfall, name_to_idx)
    pipe_exchange_acc -= pipe_cfg.coupling_interval  # 保留余量，避免时间漂移
```

**④ 主循环结束时：**

```python
shared['stop'].set()
```

### 6.2 交换函数 `_exchange_with_1d()`

**关键设计约束（已由源码确认）：**
- `ssq_t[ei]` 单位为 **m³/s**（由 L369-370 推导：`next_h = h + tq*dt/ea`，`ea=m²`，故 `tq` 须为 m³/s）
- `ssq_t` 从不被 `tick()` 重置；**每次交换开始必须先清零，再写入本轮净速率**，否则历史值会持续叠加
- `ssq_t` 在整个 coupling_interval 内持续有效：管网溢出/排水以恒定速率作用于所有内部 tick

```python
def _exchange_with_1d(shared, pipe_cfg, window_dt, h_t, ssq_t, ez_t, esl_t,
                      primary_ei, topo_ei, topo_ptr, nc_per_ei,
                      node_names, node_is_outfall,
                      name_to_idx: dict):           # 预建的 name→index 字典
    import math, numpy as np
    pi = math.pi
    ci = window_dt  # 使用实际窗口时长（非固定 coupling_interval），保证 V/Δt 转换正确

    h_np   = h_t.to_numpy()
    z_np   = ez_t.to_numpy()
    esl_np = esl_t.to_numpy()

    data_dict = {}
    q_drain = np.zeros(len(h_np), dtype=np.float32)  # 本轮排入管网速率（m³/s，负值方向）

    # ── 初始化所有节点的 data_dict entry，并对非出水口节点执行主网格排水 ──
    # 出水口节点（is_outfall=True）只传水位作为边界条件，不从 2D 排水：
    # 若排水则水从 2D 消失但永不进入 1D（_run_1d_pipe 的 node_id_list 不含出水口），
    # 造成静默质量损失。
    for i, name in enumerate(node_names):
        ei = int(primary_ei[i])
        if ei == 0 or node_is_outfall[i]:
            # 出水口或无有效主网格：只记录水位，不排水
            level = float(h_np[ei]) if ei != 0 else 0.0
            data_dict[name] = {'level': level, 'flow': 0.0}
            continue
        depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
        area  = float(esl_np[ei]) ** 2
        nc    = max(int(nc_per_ei[ei]), 1)  # 多节点共享同一主网格时，按节点数均分流量
        q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
        q_drain[ei] -= q
        data_dict[name] = {'level': depth, 'flow': float(q)}

    # ── 弱相关多网格排水（CSR 遍历，PipeTopo 不含 primary_ei，跳过出水口）──
    # topo_ptr 为 0-based CSR offset；topo_ei 中 ei 值 1-based，0 为无效哨兵
    for i, name in enumerate(node_names):
        if node_is_outfall[i] or int(primary_ei[i]) == 0:
            continue
        for j in range(int(topo_ptr[i]), int(topo_ptr[i + 1])):
            ei = int(topo_ei[j])
            if ei == 0:
                continue
            depth = max(float(h_np[ei]) - float(z_np[ei]), 0.0)
            area  = float(esl_np[ei]) ** 2
            nc    = max(int(nc_per_ei[ei]), 1)
            q = min(0.85 * pi * 0.8 * depth ** 1.5, depth * area / ci) / nc
            q_drain[ei] -= q
            data_dict[name]['flow'] += float(q)

    # ── 发送给 1D（含实际交换窗口时长，供 1D 侧 V_f/window_dt 转换）──────────
    data_dict['__window_dt__'] = {'level': float(window_dt), 'flow': 0.0}
    with shared['lock']:
        shared['2d_data'].clear()
        shared['2d_data'].update(data_dict)
        shared['2d_ready'].set()
        shared['1d_ready'].clear()

    # ── 等待 1D 返回溢出数据 ────────────────────────────────────────────
    if shared['stop'].is_set():
        return
    if not shared['1d_ready'].wait(timeout=pipe_cfg.exchange_timeout):
        raise TimeoutError('等待 1D 数据超时')

    with shared['lock']:
        pipe_data = dict(shared['1d_data'])
        shared['1d_ready'].clear()

    # ── 合并 drain + flood，一次性写入 ssq_t ────────────────────────────
    # ssq_t 从不被 tick() 重置，必须先清零再赋新速率
    ssq_np = np.zeros(len(h_np), dtype=np.float32)
    ssq_np += q_drain   # 排水（负值，m³/s）

    for name, d in pipe_data.items():
        idx = name_to_idx.get(name)  # O(1) 查找，替代 O(n) 的 list.index()
        if idx is None:
            continue
        ei = int(primary_ei[idx])
        if ei == 0:
            continue
        ssq_np[ei] += float(d.get('flow', 0.0))  # 溢出（正值，m³/s）

    ssq_t.from_numpy(ssq_np)
```

**启动时预建 `name_to_idx`：**
```python
name_to_idx = {name: i for i, name in enumerate(node_names)}
```

---

## 7. `_run_1d_pipe()` 设计

逻辑直接移植自 `Pipe_Flood_fastdb/pipe_NH801.py`，做以下适配：

### 7.1 启动阶段

```python
def _run_1d_pipe(shared, pipe_cfg):
    from pyswmm import Simulation, Nodes, Output
    from swmm.toolkit import solver as tk_solver
    from swmm.toolkit import shared_enum as tkEnum
    import fastdb4py as fdb
    from ..schema.feature import Node, IndexLike

    # 从 pipe.fdb 读取节点列表（替换 node_index.txt 文件）
    pipe_fdb = fdb.ORM.load(pipe_cfg.pipe_fdb, from_file=True)
    n_nodes = pipe_fdb[IndexLike]['node_count'][0].index
    nodes_table = pipe_fdb[Node][Node]
    node_names    = [str(nodes_table[i].name)        for i in range(n_nodes)]
    node_outfalls = [bool(nodes_table[i].is_outfall) for i in range(n_nodes)]
    node_id_list   = [n for n, o in zip(node_names, node_outfalls) if not o]
    outfall_id_list = [n for n, o in zip(node_names, node_outfalls) if o]
    del pipe_fdb

    inp_file = pipe_cfg.inp_runtime
    # 解析 .inp 各段行号
    st1, st2, st3, en1, en2, en3 = _find_index(inp_file)
    _update_start(inp_file, node_id_list, st2, en2)
    _initialize_junction_surdepth(inp_file, st1, en1)

    out_stem = Path(inp_file).stem
    out_file  = str(Path(pipe_cfg.pipe_dir) / f'{out_stem}.out')
    rpt_file  = str(Path(pipe_cfg.pipe_dir) / f'{out_stem}.rpt')
    step_idx  = 0
    prev_total_flood = {}   # 逐节点累计 flooding volume 差分基准（ML）
```

### 7.2 主循环

```
try:
    while True:
        if shared['stop'].is_set():
            break

        # 短超时轮询：每 2s 检查 stop，超过 exchange_timeout 则 raise
        waited = 0.0
        while not shared['2d_ready'].is_set():
            if shared['stop'].is_set():
                break
            shared['2d_ready'].wait(timeout=2.0)
            waited += 2.0
            if waited >= pipe_cfg.exchange_timeout:
                raise TimeoutError('[1D] Timeout waiting for 2D data')

        ① 读 external_data = shared['2d_data']，clear 2d_ready
           window_dt = external_data.get('__window_dt__', {}).get('level', pipe_cfg.coupling_interval)
        ② _update_junction_surdepth(inp, external_data, st1, en1)
           _update_inflows(inp, external_data, node_id_list, st2, en2)
           _fixed_level(inp, external_data, outfall_id_list, st3, en3)
           _set_inp_duration(inp, window_dt)  # 动态设置 SWMM 单窗口运行时长
        ③ 热启动恢复 + Simulation(inp_file) 运行
        ④ 从 Output() 读节点水位（node_result 的 index 1 = depth/head）
           raw_flood = _total_inflow_from_rpt(rpt_file)  # 逐节点累计量（ML）
           delta = raw_flood[name] - prev_total_flood[name]  # 本窗口增量
           prev_total_flood[name] = raw_flood[name]
        ⑤ 构建 data_dict = {name: {level, flow}}
           level = Output() 读到的节点水位（非 0.0）
           flow = delta * 1000 / window_dt（ML → m³ → m³/s）
        ⑥ shared['1d_data'] ← data_dict，set 1d_ready
        step_idx += 1
finally:
    shared['stop'].set()   # 确保 2D 侧也能感知 1D 的意外退出
```

### 7.3 辅助函数

以下函数直接从 `pipe_NH801.py` 移植（仅调整命名风格）：

| 函数 | 用途 |
|---|---|
| `_find_index(inp)` | 定位 `[JUNCTIONS]`、`[INFLOWS]`、`[OUTFALLS]` 行号 |
| `_update_start(inp, nodes, s, e)` | 初始化所有 inflows 为 0 |
| `_initialize_junction_surdepth(inp, s, e)` | 初始化 surcharge depth 为 0 |
| `_update_junction_surdepth(inp, data, s, e)` | 用 2D 水位更新 surcharge depth |
| `_update_inflows(inp, data, nodes, s, e)` | 用 2D 流量更新 INFLOWS 段 |
| `_fixed_level(inp, data, s, e)` | 固定出水口水位 |
| `_total_inflow_from_rpt(rpt)` | 从 .rpt 报告提取逐节点 flooding volume（ML，cumulative） |
| `_set_inp_duration(inp, window_dt)` | 动态修改 .inp 的 END_DATE/END_TIME，使 SWMM 只运行一个耦合窗口 |

---

## 8. 交换物理公式

### 8.1 2D → 1D（地表水排入管网）

对节点 $i$ 覆盖的网格 $k$：

$$q_{\text{drain},k} = \min\!\left(0.85\,\pi\cdot 0.8\cdot d_k^{1.5},\; \frac{d_k \cdot A_k}{\Delta t_c}\right)$$

- $d_k = \max(h_k - z_k,\; 0)$：地表水深（m）
- $A_k = l_k^2$：网格面积（m²）
- $\Delta t_{\text{exch}}$：本次交换窗口实际时长（`window_dt`，≈`coupling_interval`，但自适应 dt 可能有余量）
- 孔口系数取自 Pipe_Flood_fastdb：$0.85 \times \pi \times 0.8$
- **单位**：`ssq_t` 须为 **m³/s**（由 `tick()` L369-370 推导：`next_h = h + tq*dt/ea`，`ea = esl²`，`tq` 必须为 m³/s）

对 2D 模型：$\text{ssq}\_t[k] \mathrel{-}= q_{\text{drain},k}$（每轮交换以 `np.zeros()` 重建，不累加历史值）

### 8.2 1D → 2D（管网溢出回地面）

SWMM 报告的节点 flooding volume $V_f$（10^6 litres = 1 ML = 1000 m³，SI 模式 .rpt 单位）转换为流率：

$$q_{\text{flood},i} = \frac{\Delta V_{f,i} \times 1000}{\Delta t_{\text{exch}}} \quad \text{(m³/s)}$$

其中 $\Delta V_{f,i}$ 为本窗口内第 $i$ 个节点的累计 flooding volume 增量（ML）；
`prev_total_flood` 为逐节点字典（非标量），跟踪每个节点的累计基准。

对主网格：$\text{ssq}\_t[\text{primary\_ei}[i]] \mathrel{+}= q_{\text{flood},i}$

> **净效果**：每个实际交换窗口（时长 $\Delta t_{\text{exch}}$）开始时，`ssq_t` 被清零并重新写入本轮净速率
> $\text{ssq}\_t[k] = q_{\text{flood},k} - q_{\text{drain},k}$（单位 m³/s），
> 在该 interval 内所有 tick 中持续有效。

---

## 9. 注意事项

| # | 事项 | 处理 |
|---|---|---|
| 1 | `ssq_t` 单位已确认为 m³/s | 无需转换，直接赋值 |
| 2 | `ssq_t` 每次交换**必须先清零**再赋新值，否则历史速率持续叠加 | 代码已用 `np.zeros()` 重建 |
| 3 | `multiprocessing` 必须显式使用 `spawn` 模式，禁止依赖平台默认值（Linux 默认 `fork` 会继承 `_TI_INITIALIZED`，导致子进程跳过 GPU 初始化） | 使用 `multiprocessing.get_context("spawn")` 创建 Manager/Process |
| 4 | `fdb.STR` 字段无 numpy 批量访问，需逐条迭代读取节点 name | 只在预处理/启动阶段执行一次，性能可接受 |
| 5 | 热启动 `.hsf` 路径：`pipe_dir/hotstart/hsf_{step}.hsf` | `_run_1d_pipe()` 内管理 |
| 6 | `_total_inflow_from_rpt()` 返回**逐节点累计量字典**（ML），由 `_run_1d_pipe()` 逐节点差分得本窗口增量 | 见 §7.2 步骤④⑤ |
| 7 | **出水口节点（is_outfall=True）不参与排水**：只向 1D 传递水位作为边界条件；若排水则水从 2D 消失而不进入 1D，造成静默质量损失 | `_exchange_with_1d()` 中出水口节点 `flow=0.0`，`level` 传绝对水位 |

---

## 10. 错误处理

| 情况 | 处理方式 |
|---|---|
| 1D 进程超时未响应 | `_run_2d()` 在 `try/finally` 中抛出 `TimeoutError`，`finally` 块保证 `shared['stop'].set()`，随后 `solver_coupled()` 用限时 `join()` 回收进程 |
| 2D 进程提前退出 | `_run_2d()` 的 `finally` set `stop`；`_run_1d_pipe()` 的 timeout poll 感知到后退出 `try/finally`，同样 set `stop` |
| 双侧 `stop` 保证 | 两个工作进程均以 `try/finally: shared['stop'].set()` 包裹，任意一侧异常都不会导致另一侧永久阻塞 |
| 节点坐标超出网格范围 | `primary_ei[i] = 0`，该节点初始化为零流量 entry，跳过排水和回灌，打 warning 日志 |
| SWMM .inp 解析失败 | `prepare_pipe()` 阶段报错，终止预处理 |
| `_total_inflow_from_rpt()` 返回值 | 必须是**本交换窗口的增量体积**（m³），而非累计总量；由 `_run_1d_pipe()` 负责逐轮差分 |

---

## 11. 使用示例

```python
from nh_flood_2d.input import load_domain_config, load_force_config
from nh_flood_2d.input.pipe import load_pipe_config
from nh_flood_2d.preprocess import preprocess
from nh_flood_2d.preprocess.pipe import prepare_pipe
from nh_flood_2d.core.solver_coupled import solver_coupled

domain_cfg = load_domain_config('./resource/domain.json')
force_cfg  = load_force_config('./resource/force.json')
pipe_cfg   = load_pipe_config('./resource/pipe.json')

# resource/pipe.json 示例：
# {
#   "inp":               "./resource/network.inp",
#   "pipe_dir":          "./resource/pipe",
#   "coupling_interval": 300.0,
#   "exchange_timeout":  300.0
# }

preprocess(domain_cfg, force_cfg)   # 原有预处理
prepare_pipe(pipe_cfg, domain_cfg)  # 新增：生成 pipe.fdb

solver_coupled(domain_cfg, force_cfg, pipe_cfg)
```

---

## 12. 文件变更汇总

### 新增文件

| 路径 | 说明 |
|---|---|
| `src/nh_flood_2d/input/pipe.py` | `PipeConfig`、`load_pipe_config` |
| `src/nh_flood_2d/preprocess/pipe.py` | `prepare_pipe()`，节点–网格拓扑预处理 |
| `src/nh_flood_2d/core/solver_coupled.py` | 耦合主函数，`_run_2d`、`_run_1d_pipe`、辅助函数 |

### 修改文件

| 路径 | 修改内容 |
|---|---|
| `src/nh_flood_2d/schema/feature.py` | 新增 `PipeTopo`、`Node` Feature 类 |
| `src/nh_flood_2d/input/__init__.py` | 导出 `PipeConfig`、`load_pipe_config` |
| `src/nh_flood_2d/preprocess/__init__.py` | 导出 `prepare_pipe` |
| `src/nh_flood_2d/core/__init__.py` | 导出 `solver_coupled` |

### 不修改文件

| 路径 | 原因 |
|---|---|
| `solver_compact.py` | 保持不动，`_run_2d` 复制其逻辑并扩展 |
| `schema/feature.py` 中的 `Node` | 已有字段足够，无需修改 |

