# 闸门控制实现与验收证据（2026-11-05）

实现基于分配检出的 a8fb1e8。`git diff 7648185 a8fb1e8 -- src/nh_flood_2d/core src/nh_flood_2d/input src/nh_flood_2d/preprocess` 为空，确认本任务起点与旧正式闭闸版的求解器一致。新源码按仓库 `src-py-sha256/v1` 规则的 SHA-256 为 `0b6590b22408380c7efcffcb807817963b5650bea1282a853e545be9e67c9f34`。

新增逐闸连通块映射、设备端每步水位触发、不可逆锁存和按绝对时钟匀速降床面；删除旧边界序列触发与切步。检查点保存逐闸下令、首次过水和首次全开采样时钟、开启配置身份、塘区初始库存与精确累计器。manifest 保存回放函数与最终收支，`gate_balance.csv` 保存现有输出时刻的累计交换过程。配置、执行顺序和收支符号见 [参考文档](../gate-control.md)。

## 测试结果

以下为分批实际执行的结果，合计 210 项通过；原有外部资料检查跳过 3 项。香港慢测另由自动比较脚本完整执行并通过。没有将这些结果表述为一次全量 pytest 调用的结果。

| 批次 | 结果 | 实际覆盖 |
| --- | --- | --- |
| 原有回归，排除三个新增测试文件 | 164 通过、3 跳过 | 水动力、CFL、限流、雨洪、Horton、SWMM 耦合、输出、续跑及其他仓库检查 |
| `test_gate_config.py` | 14 通过 | 旧字段、有限数、开门时长、名称、连通块全覆盖/唯一归属、内外点、边界重叠 |
| `test_gate_solver.py` | 30 通过 | 控制、时钟、床动、干湿双向交换、粗细界面、收支、回放和续跑 |
| 公共 kernel AST 比较 | 1 通过 | 禁用新增静态诊断后，10 个数值 kernel 与 7648185 的 AST 一致 |
| 小算例三路独立进程比较 | 1 通过 | 原版、无闸配置、显式禁用；14 步，2 s 预热加 12 s 正式期，非零流量 |
| 香港 1 h 三路独立进程比较 | 脚本通过 | 实际 Bnone-I 网格，550,712 单元；每路 9,054 步；0 s 预热加 3,600 s 正式期 |

原有 3 项跳过原因是外部点云、DEM 和 libgmt 不可用。新增香港 pytest 入口默认需要设置 `NH_GATE_HK_DOMAIN` 与 `NH_GATE_HK_FORCE`，未设置时会额外跳过该慢测；本次通过下面的命令实际执行了相同三路比较。

机制测试包括：各闸独立触发；等于门槛不触发；干/近干控制点无效；初始超阈；短时上穿后回落仍锁存；预热禁用；批大小 1/32 与输出间隔 1/2 s 的规则一致；开到底恰逢输出；固定静水和整体同量床动零流；局部床动引起正确交换及时间步细化收敛；多格干通道先入后出；近干供水、小水位差和反向水头；限流后实际交换；悬挂子边宽度与共享通量守恒，并捕获重构确实读取已改变的床面；开启前/中/后续跑的状态、累计器元数据和后续过程行逐位一致；中途开启床面回放逐位一致；塘区降雨、下渗、漫埂、舍入和水位边界补排闭合。另保留了局部降床 0.1 m 后第一个 0.1 s 水力步约 0.1962 m³/s 的量级检查。

本轮保存了 40 个已完成机制子运行的关键记录，见 [mechanisms.json](mechanisms.json)。其中塘区残差的最大绝对值为 `1.3003713320358656e-5 m³`；每个机制测试使用与其量级匹配的明确容差。两单元/局部床动没有使用“局部降床后仍零流”的错误验收标准。

独立环境位于分配检出的 `tests/.gate-venv`，Python 3.10.17、Taichi 1.7.4、FastDB 0.2.4，CPU 为 Apple M1 Max。作者原环境的 FastDB 为 0.1.12，接口不兼容，因此没有修改或复用它的依赖。SWMM 所需依赖在测试环境补齐，并仅对该环境里的两个 dylib 按仓库现有脚本重新签名。早期尚未补齐依赖及运行中源码变化造成的失败已在稳定源码上重新验证。

执行命令使用 `TI_OFFLINE_CACHE=0 PYTHONDONTWRITEBYTECODE=1` 和 `tests/.gate-venv/bin/python -B`。主要入口如下，pytest 加 `-p no:cacheprovider --basetemp tests/.gate-work/<独立目录>`：

```sh
python -B -m pytest -q tests --ignore=tests/test_gate_solver.py --ignore=tests/test_gate_config.py --ignore=tests/test_gate_compatibility.py
python -B -m pytest -q tests/test_gate_solver.py
python -B -m pytest -q tests/test_gate_config.py tests/test_gate_compatibility.py -k 'not hk'
python -B tests/gate_compatibility.py --case hk \
  --domain <case-dir>/runs_formal/Bnone-I/config/domain.json \
  --force <case-dir>/runs_formal/Bnone-I/config/force.json \
  --output tests/.gate-work/compat-hk
```

## 逐位兼容证据与公共路径

[small.json](small.json) 和 [hong-kong-1h.json](hong-kong-1h.json) 保存三路结果、输入 SHA-256、完整 h/q/床面、dt、完整轨迹、最终预算、质量平衡 CSV 与步长统计 CSV 的 SHA-256。三路每个哈希均完全相同。独立进程使用相同的 CPU 单线程配置，避免多线程块内 F32 原子加法的执行顺序掩盖兼容性判断。

香港 1 h 的公共哈希为：

```text
h       f874fd27f503c948185aa8871bdb159623012a09682f2b54ff95bfd311cdd123
q       b8a36234ecaa63a62a06becf0af32c8754777bd4d53549ff89b92e9bc6203ca5
bed     90c3bb7c71daba58ef4f6bb9c96b6e31bb4918b906d3fbe7ba2ed973fdded090
dt      7c8d3e086188e9c3796fb1572a6a47d11bb3fcf950ede42fa8dac9d693e940cb
trace   ff42cb36f27c3853119bded813f2d67430a9e3ef8637a5bd314d4e02fa9fb667
budget  2ffca8257e343a7d7b1f933e281aa20c78d1a5832d866fc6136fbb1b10df9e62
masscsv 528a33f868375b846fc0dcf817887283e804663a5a58028b8157c281af9a44d2
stepcsv 36f0f7c72bfbe6492d2944298a9d814deeae0c52a95629827ac036f661717280
```

公共数值路径的差异审计显示，ACC 压差/摩阻计算、唯一子边限流、水深更新、边界施加、压力重构和 CFL 的原有表达式未改变。`tick` 只增加 `ti.static(gate_enabled)` 下的只读记账；禁用时这些语句被编译掉。图中的新增阶段、累计器、状态传输与 CSV 均只在启用时存在。时钟、压力重构和全域预算实现文件没有数值改动；旧闸门的边界穿越事件与 Python 开门处理已删除。AST 检查与两类实际比对共同支撑兼容结论。

开启时的顺序为：用上一完成状态的 CFL 选步长，步首按真实时钟检查控制并更新床面，随后做悬挂边压力重构、ACC 通量、供水限流；用最终子边通量记账，再更新水深、降雨、下渗、正性和边界水位；独立测量实际库存舍入，按步末时钟更新床面，计算新 CFL 并提交时钟，最后在现有输出时刻采样。塘区取完整地类 10 单元集合，进出塘与河道侧分别累计，漫埂排除闸口边，床动不产生水量源项。

## 未执行的工作与范围

没有使用 GPU、访问 Windows、启动正式全事件运行、推送或创建 PR。CPU 结果不构成 GPU 位级验证，也不构成整个正式事件或其他网格的验证。香港冒烟使用 Bnone-I 实际输入的第一个小时，未运行 12 h 预热。

未替作者确定实际七个控制点或重新估算最终闸底；这些由运行配置阶段冻结。gridmen 的 `formal_io`、`run_metrics` 和 `provenance` 明确在只读范围，未修改，正式运行前仍需 Host 同步床面回放、闸区水位极值排除及新源码审计条件。根目录 README 的旧闸门示例在本任务 `docs/src/tests` 写入范围之外，未改；新的配置参考已在 `docs` 中完整替换旧接口。作者要求的 Host 验收后 astra max 审计尚未执行；本任务内部只读检查不能替代它。

提交保留在分配的 detached HEAD 检出，供 Host 验收后集成到目标 `feat/gate-control`；未移动作者检出或其他分支/标签。
