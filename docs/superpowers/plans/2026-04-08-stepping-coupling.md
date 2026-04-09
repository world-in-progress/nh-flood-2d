# Stepping-Based Tight Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hotstart-restart SWMM coupling with a continuous stepping approach using pyswmm's runtime solver API, eliminating volume→rate conversion issues, .rpt parsing, and hotstart file management.

**Architecture:** SWMM runs as a single continuous `Simulation` with `step_advance(coupling_interval)`. At each coupling point, the 2D process sends drainage rates and water levels; the 1D process applies them via `solver.node_set_total_inflow()`, `solver.node_set_parameter(SURCHARGE_DEPTH)`, and `solver.outfall_set_stage()`, then reads back flooding/depth directly via `solver.node_get_result()`. The async lag-1 IPC pattern (Events + Manager dict) is preserved unchanged.

**Tech Stack:** pyswmm 2.1.0, swmm.toolkit (solver API), Taichi (2D GPU), fastdb4py

---

## Background

### Current Hotstart-Restart Approach (Problems)

Each coupling window: rewrite `.inp` (INFLOWS/JUNCTIONS/OUTFALLS) → restart SWMM → parse `.rpt` for volumes (ML) → convert `vol * 1000 / duration` to m³/s → manage hotstart files.

1. **Volume→rate conversion ambiguity** — reference uses `10/6 = 1000/600` but .inp is 300s; mismatch causes 2× flooding
2. **`.rpt` column-index fragility** — hardcoded `row[5]`/`row[7]` breaks if SWMM format changes
3. **Hotstart state discontinuity** — each restart rebuilds hydraulic state, potential jumps
4. **Heavy file I/O** — 6 file rewrites + 2 file parses per coupling window
5. **Duration/interval entanglement** — `.inp` END_TIME, coupling_interval, and conversion divisor must all match

### New Stepping Approach (Benefits)

SWMM runs once for the full simulation. At each coupling point, use runtime solver API:

1. **No conversion factor** — set drainage as flow rate (m³/s) directly via `solver.node_set_total_inflow()`; read flooding as flow rate (m³/s) directly via `solver.node_get_result(FLOOD)`
2. **Continuous state** — SWMM internal momentum/pressure preserved across windows
3. **No file I/O per window** — all updates via in-memory API calls
4. **No hotstart files** — single simulation, no save/restore overhead
5. **Decoupled timing** — `.inp` END_TIME = total simulation duration; `coupling_interval` is user-configurable and independent

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nh_flood_2d/core/coupled/pipe_1d.py` | **Rewrite** | Replace hotstart-restart loop with continuous `Simulation` + `step_advance` + solver API |
| `src/nh_flood_2d/core/coupled/exchange.py` | **Modify** | Update `apply_sources()` — 1D return data is now flow rates (m³/s), no volume conversion needed |
| `src/nh_flood_2d/core/coupled/driver.py` | **Simplify** | Remove `.inp` duration derivation of `coupling_interval`; restore user-configurable interval |
| `src/nh_flood_2d/input/pipe.py` | **Modify** | Restore `coupling_interval` as user-configurable field (default 600s); remove `hotstart_dir` property |
| `resource/pipe.json` | **Modify** | Add `coupling_interval: 600` back as user setting |
| `resource/network.inp` | **Modify** | Set END_TIME to cover full simulation duration (36h for current tide data) |

## Tasks

### Task 1: Update PipeConfig and pipe.json

**Files:**
- Modify: `src/nh_flood_2d/input/pipe.py`
- Modify: `resource/pipe.json`

- [ ] **Step 1: Restore `coupling_interval` as user-configurable**

In `src/nh_flood_2d/input/pipe.py`, change `coupling_interval` from auto-derived to user-settable with default 600:

```python
class PipeConfig(BaseModel):
    inp: str
    pipe_dir: str

    coupling_interval: float = 600.0  # 2D↔1D exchange interval (seconds)
    exchange_timeout: float  = 600.0
    weak_dist_thresh: float  = 50.0
```

Remove the `hotstart_dir` property (no longer needed):

```python
    # DELETE this property:
    # @property
    # def hotstart_dir(self) -> str:
    #     ...
```

- [ ] **Step 2: Update pipe.json**

Add `coupling_interval` back:

```json
{
    "inp": "./resource/network.inp",
    "pipe_dir": "./resource/pipe",
    "coupling_interval": 600.0,
    "exchange_timeout": 600.0,
    "weak_dist_thresh": 50.0
}
```

- [ ] **Step 3: Commit**

```bash
git add src/nh_flood_2d/input/pipe.py resource/pipe.json
git commit -m "refactor(pipe): restore user-configurable coupling_interval, remove hotstart_dir"
```

### Task 2: Rewrite pipe_1d.py — continuous stepping

**Files:**
- Rewrite: `src/nh_flood_2d/core/coupled/pipe_1d.py`

This is the core change. All old helper functions (`_find_index`, `_update_start`, `_update_inflows`, `_update_junction_surdepth`, `_fixed_level`, `_total_inflow_from_rpt`, `_get_inp_duration`) are deleted — they are all replaced by solver API calls.

- [ ] **Step 1: Write `_set_inp_end_time()` helper**

Unlike the old approach that set duration to the coupling window, this sets END_TIME to cover the full 2D simulation so SWMM can run continuously:

```python
def _set_inp_end_time(inp_path: str, total_seconds: float) -> None:
    """Set .inp END_DATE/END_TIME to START + total_seconds."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    vals: dict = {}
    for line in lines:
        for key in ('START_DATE', 'START_TIME'):
            m = re.match(rf'^{key}\s+(\S+)', line)
            if m:
                vals[key] = m.group(1)

    start = _dt.strptime(
        f"{vals['START_DATE']} {vals['START_TIME']}", '%m/%d/%Y %H:%M:%S',
    )
    end = start + timedelta(seconds=total_seconds)

    new_lines = []
    for line in lines:
        if re.match(r'^END_DATE\s', line):
            new_lines.append(f'END_DATE             {end.strftime("%m/%d/%Y")}\n')
        elif re.match(r'^END_TIME\s', line):
            new_lines.append(f'END_TIME             {end.strftime("%H:%M:%S")}\n')
        else:
            new_lines.append(line)

    with open(inp_path, 'w') as f:
        f.writelines(new_lines)
```

- [ ] **Step 2: Write `run_1d_pipe()` — initialization and SWMM open**

```python
def run_1d_pipe(shared, pipe_cfg: PipeConfig) -> None:
    """1D SWMM pipe-network loop — continuous stepping mode."""
    from pyswmm import Simulation, Nodes
    from swmm.toolkit import solver, shared_enum as tkEnum

    try:
        inp_runtime = str(pipe_cfg.inp_runtime)
        shutil.copy2(pipe_cfg.inp, inp_runtime)

        # Load node metadata from pipe.fdb
        pipe_fdb = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl = pipe_fdb[Node]['Node']
        node_names = [str(nodes_tbl[i].name) for i in range(n_nodes)]
        is_outfall = [bool(nodes_tbl[i].is_outfall) for i in range(n_nodes)]
        del pipe_fdb

        # Set .inp END_TIME to a large value so SWMM doesn't end early.
        # 2D controls the stop via shared['stop']; we set 7 days as ceiling.
        _set_inp_end_time(inp_runtime, 7 * 24 * 3600)

        coupling_interval = float(pipe_cfg.coupling_interval)

        with Simulation(inp_runtime) as sim:
            sim.step_advance(int(coupling_interval))

            # Build name → SWMM-internal-index map
            nodes_obj = Nodes(sim)
            name_to_swmm_idx: dict[str, int] = {}
            for i, node in enumerate(nodes_obj):
                name_to_swmm_idx[node.nodeid] = i
```

- [ ] **Step 3: Write `run_1d_pipe()` — main stepping loop**

Inside the `with Simulation` block, after initialization:

```python
            window_step = 0

            for _step in sim:
                # ── Wait for 2D data ──────────────────────────────
                if shared['stop'].is_set():
                    break

                waited = 0.0
                while not shared['2d_ready'].is_set():
                    if shared['stop'].is_set():
                        break
                    shared['2d_ready'].wait(timeout=2.0)
                    waited += 2.0
                    if waited >= float(pipe_cfg.exchange_timeout):
                        raise TimeoutError('[1D] Timeout waiting for 2D data')

                if shared['stop'].is_set():
                    break

                with shared['lock']:
                    raw_2d = dict(shared['2d_data'])
                    shared['2d_ready'].clear()

                t_1d_start = time.perf_counter()

                # ── Apply 2D → 1D boundary conditions ────────────
                for i, name in enumerate(node_names):
                    swmm_idx = name_to_swmm_idx.get(name)
                    if swmm_idx is None:
                        continue
                    d = raw_2d.get(name, {})

                    if is_outfall[i]:
                        # Outfall: set stage (absolute water surface elevation)
                        level = float(d.get('level', 0.0))
                        solver.outfall_set_stage(swmm_idx, level)
                    else:
                        # Junction: set surcharge depth + drainage inflow
                        surdepth = float(d.get('level', 0.0))
                        solver.node_set_parameter(
                            swmm_idx,
                            tkEnum.NodeProperty.SURCHARGE_DEPTH,
                            surdepth,
                        )

                    # Set drainage as direct inflow (m³/s) — no conversion!
                    flow = float(d.get('flow', 0.0))
                    solver.node_set_total_inflow(swmm_idx, flow)

                # SWMM steps forward by coupling_interval internally
                # (handled by sim.step_advance + the for loop iterator)

                # ── Read 1D → 2D results ─────────────────────────
                data_1d: dict = {}
                for i, name in enumerate(node_names):
                    swmm_idx = name_to_swmm_idx.get(name)
                    if swmm_idx is None:
                        data_1d[name] = {'level': 0.0, 'flow': 0.0}
                        continue

                    depth = solver.node_get_result(
                        swmm_idx, tkEnum.NodeResult.DEPTH,
                    )
                    head = solver.node_get_result(
                        swmm_idx, tkEnum.NodeResult.HEAD,
                    )
                    flood = solver.node_get_result(
                        swmm_idx, tkEnum.NodeResult.FLOOD,
                    )

                    level = head if is_outfall[i] else depth
                    data_1d[name] = {
                        'level': float(level),
                        'flow': float(flood),  # m³/s — direct, no conversion
                    }

                t_1d_elapsed = time.perf_counter() - t_1d_start

                # ── Send results to 2D ────────────────────────────
                with shared['lock']:
                    shared['1d_data'].clear()
                    shared['1d_data'].update(data_1d)
                    shared['1d_data']['__1d_elapsed__'] = t_1d_elapsed
                    shared['1d_ready'].set()
                    shared['2d_ready'].clear()

                window_step += 1

    finally:
        shared['stop'].set()
```

- [ ] **Step 4: Commit**

```bash
git add src/nh_flood_2d/core/coupled/pipe_1d.py
git commit -m "refactor(pipe_1d): replace hotstart-restart with continuous stepping

Replace entire hotstart-restart mechanism with pyswmm's step_advance +
solver toolkit API. Key changes:
- solver.node_set_total_inflow() replaces .inp INFLOWS rewriting
- solver.node_set_parameter(SURCHARGE_DEPTH) replaces junction editing
- solver.outfall_set_stage() replaces outfall level editing
- solver.node_get_result(FLOOD) replaces .rpt parsing
- No hotstart files, no volume→rate conversion (10/6 eliminated)
- Single continuous SWMM Simulation for full duration"
```

### Task 3: Simplify driver.py

**Files:**
- Modify: `src/nh_flood_2d/core/coupled/driver.py`

- [ ] **Step 1: Remove .inp duration derivation**

Remove the block that imports `_get_inp_duration` and overwrites `coupling_interval`. The driver should simply pass `pipe_cfg` through unchanged:

```python
    if has_pipe:
        print(f'[coupled] coupling_interval = '
              f'{pipe_cfg.coupling_interval:.0f}s')

        mgr = ctx.Manager()
        # ... rest unchanged
```

Also remove the import line `from .pipe_1d import _get_inp_duration` (the function no longer exists).

- [ ] **Step 2: Commit**

```bash
git add src/nh_flood_2d/core/coupled/driver.py
git commit -m "refactor(driver): remove .inp duration derivation, use config directly"
```

### Task 4: Update exchange.py — no conversion in apply_sources

**Files:**
- Modify: `src/nh_flood_2d/core/coupled/exchange.py`

- [ ] **Step 1: Verify apply_sources needs no change**

`apply_sources()` already works with flow rates (m³/s). The 1D return data in the stepping approach is already in m³/s (`solver.node_get_result(FLOOD)` returns m³/s in CMS mode). The function at L163-170 does:

```python
q_source[ei] += float(d.get('flow', 0.0))
```

This is correct — no change needed. Just update the module docstring to reflect the new coupling mode:

```python
"""
2D ↔ 1D exchange protocol (async / lagged coupling).

...
Note: With stepping coupling, 1D return values are already in m³/s
(read directly from SWMM solver API), so no volume→rate conversion
is needed.
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/nh_flood_2d/core/coupled/exchange.py
git commit -m "docs(exchange): update docstring for stepping coupling mode"
```

### Task 5: Verify and run

- [ ] **Step 1: Verify imports**

Run a quick syntax check:

```bash
uv run python -c "from nh_flood_2d.core.coupled.pipe_1d import run_1d_pipe; print('OK')"
```

- [ ] **Step 2: Run simulation**

```bash
uv run python main.py
```

Expected output should show:
- `[coupled] coupling_interval = 600s`
- `[2D] t=1800.x s ...` progress messages
- No hotstart file creation
- No `.rpt` parsing messages

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: complete stepping-based tight coupling refactor"
```

## Key Design Decisions

1. **`NodeResult.FLOOD` returns m³/s** — in CMS flow units, this is the instantaneous flooding rate, not cumulative volume. This eliminates the entire ML→m³/s conversion problem.

2. **`step_advance(coupling_interval)`** — SWMM internally runs its own routing steps (15s per .inp ROUTING_STEP) but only returns control to Python at each coupling interval. This means SWMM does ~40 internal steps per 600s exchange window.

3. **`.inp END_TIME` set to 7 days** — a ceiling that will never be reached. The 2D process controls actual termination via `shared['stop']`. If the simulation ends early, SWMM's `for _step in sim` loop just exits when the stop event is set.

4. **Drainage flow directly as INFLOW** — `solver.node_set_total_inflow(idx, rate_m3s)` accepts the drainage rate in the same units as SWMM's FLOW_UNITS (CMS = m³/s). No unit conversion at all.
