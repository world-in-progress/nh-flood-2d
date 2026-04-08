"""
1D SWMM pipe-network subprocess.

Contains:
  run_1d_pipe()  — main loop (subprocess target for multiprocessing.Process)
  SWMM .inp file manipulation helpers (private)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import fastdb4py as fdb

from ...schema.feature import IndexLike, Node

if TYPE_CHECKING:
    from ...input.pipe import PipeConfig


# ─── SWMM .inp file helpers ──────────────────────────────────────────────────────


def _find_index(inp_path: str):
    """Return (s1, s2, s3, e1, e2, e3): line indices for JUNCTIONS, INFLOWS, OUTFALLS."""
    with open(inp_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    s1 = s2 = s3 = e1 = e2 = e3 = None
    for i, line in enumerate(lines):
        if line == '[JUNCTIONS]':
            s1 = i + 3
        elif s1 is not None and e1 is None and line == '':
            e1 = i
        if line == '[INFLOWS]':
            s2 = i + 3
        elif s2 is not None and e2 is None and line == '':
            e2 = i
        if line == '[OUTFALLS]':
            s3 = i + 3
        elif s3 is not None and e3 is None and line == '':
            e3 = i
    return s1, s2, s3, e1, e2, e3


def _update_start(inp_path: str, node_id_list: list, s2: int, e2: int) -> None:
    """Initialise [INFLOWS] section to zero for all junction nodes."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s2]
    for name in node_id_list:
        new_lines.append(
            f'{name:<15} FLOW             ""               FLOW     1.0      1.0      0\n'
        )
    new_lines.extend(lines[e2:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _initialize_junction_surdepth(inp_path: str, s1: int, e1: int) -> None:
    """Reset surcharge-depth column (index 4) to 0 for all junctions."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s1]
    for i in range(s1, e1):
        parts = lines[i].split()
        if len(parts) > 4:
            parts[4] = '0'
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e1:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _update_junction_surdepth(inp_path: str, level_dict: dict, s1: int, e1: int) -> None:
    """Update surcharge depth (column 4) for each junction from 2D level data."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s1]
    for i in range(s1, e1):
        parts = lines[i].split()
        if len(parts) > 4:
            name = parts[0]
            if name in level_dict:
                parts[4] = str(level_dict[name].get('level', 0.0))
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e1:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _update_inflows(inp_path: str, data: dict, node_id_list: list, s2: int, e2: int) -> None:
    """Rewrite [INFLOWS] section with flow values from 2D exchange data (m³/s)."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s2]
    for name in node_id_list:
        flow = data.get(name, {}).get('flow', 0.0)
        new_lines.append(
            f'{name:<15} FLOW             ""               FLOW     1.0      1.0      {flow}\n'
        )
    new_lines.extend(lines[e2:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _fixed_level(inp_path: str, data_dict: dict, s3: int, e3: int) -> None:
    """Update [OUTFALLS] tide level (column 3) with absolute WSE from 2D."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    new_lines = lines[:s3]
    for i in range(s3, e3):
        parts = lines[i].split()
        if len(parts) > 3:
            name = parts[0]
            if name in data_dict:
                parts[3] = str(data_dict[name].get('level', 0.0))
            new_lines.append(('  '.join(f'{p:<15}' for p in parts)).rstrip() + '\n')
        else:
            new_lines.append(lines[i])
    new_lines.extend(lines[e3:])
    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


def _total_inflow_from_rpt(rpt_path: str) -> dict:
    """Parse SWMM .rpt → {node_name: cumulative_volume_ML}.

    Extracts two sections:
      - "Node Inflow Summary"   → outfall discharge (column 7)
      - "Node Flooding Summary" → junction flooding  (column 5)
    Both volumes are in 10^6 litres (ML) under SI units.
    """
    with open(rpt_path, 'r', encoding='utf-8') as f:
        rows = [ln.strip().split() for ln in f.readlines()]
    result: dict = {}

    # --- Outfall discharge from "Node Inflow Summary" ---
    try:
        fi = rows.index(['Node', 'Inflow', 'Summary']) + 9
        fe = rows.index(['Node', 'Surcharge', 'Summary']) - 4
        for i in range(fi, fe + 1):
            row = rows[i]
            if len(row) > 7 and row[1] == 'OUTFALL':
                result[row[0]] = float(row[7])
    except (ValueError, IndexError):
        pass

    # --- Junction flooding from "Node Flooding Summary" ---
    try:
        fi = rows.index(['Node', 'Flooding', 'Summary']) + 10
        for row in rows[fi:]:
            if not row:
                break
            result[row[0]] = float(row[5])
    except (ValueError, IndexError):
        pass

    return result


def _set_inp_duration(inp_path: str, duration_sec: float) -> None:
    """Rewrite [OPTIONS] END_DATE/END_TIME so SWMM runs exactly `duration_sec` seconds."""
    import re
    from datetime import datetime as _dt, timedelta

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    start_date = start_time = None
    for line in lines:
        m = re.match(r'^START_DATE\s+(\S+)', line)
        if m:
            start_date = m.group(1)
        m = re.match(r'^START_TIME\s+(\S+)', line)
        if m:
            start_time = m.group(1)

    if start_date is None or start_time is None:
        return

    start_dt = _dt.strptime(f'{start_date} {start_time}', '%m/%d/%Y %H:%M:%S')
    end_dt = start_dt + timedelta(seconds=duration_sec)
    end_date_str = end_dt.strftime('%m/%d/%Y')
    end_time_str = end_dt.strftime('%H:%M:%S')

    new_lines = []
    for line in lines:
        if re.match(r'^END_DATE\s', line):
            new_lines.append(f'END_DATE             {end_date_str}\n')
        elif re.match(r'^END_TIME\s', line):
            new_lines.append(f'END_TIME             {end_time_str}\n')
        else:
            new_lines.append(line)

    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


# ─── 1D subprocess entry point ───────────────────────────────────────────────────


def run_1d_pipe(shared, pipe_cfg: PipeConfig) -> None:
    """1D SWMM pipe-network loop; runs in a subprocess spawned by the driver."""
    import shutil as _shutil
    from pyswmm import Simulation, Output

    try:
        inp_runtime = str(pipe_cfg.inp_runtime)
        _shutil.copy2(pipe_cfg.inp, inp_runtime)

        s1, s2, s3, e1, e2, e3 = _find_index(inp_runtime)

        pipe_fdb = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl = pipe_fdb[Node]['Node']
        node_names = [str(nodes_tbl[i].name) for i in range(n_nodes)]
        is_outfall = [bool(nodes_tbl[i].is_outfall) for i in range(n_nodes)]
        del pipe_fdb

        junction_names = [n for n, f in zip(node_names, is_outfall) if not f]

        _update_start(inp_runtime, junction_names, s2, e2)
        _initialize_junction_surdepth(inp_runtime, s1, e1)

        hotstart_dir = Path(pipe_cfg.hotstart_dir)
        hotstart_dir.mkdir(parents=True, exist_ok=True)

        prev_total_flood: dict = {}
        window_step = 0

        while True:
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

            window_dt = float(raw_2d.get('__window_dt__', {}).get('level', pipe_cfg.coupling_interval))

            t_1d_start = time.perf_counter()

            _update_junction_surdepth(inp_runtime, raw_2d, s1, e1)
            _fixed_level(inp_runtime, raw_2d, s3, e3)
            _set_inp_duration(inp_runtime, window_dt)

            hsf_in = hotstart_dir / f'step_{window_step}.hsf'
            hsf_out = hotstart_dir / f'step_{window_step + 1}.hsf'
            with Simulation(inp_runtime) as sim:
                if hsf_in.exists():
                    sim.use_hotstart(str(hsf_in))
                for _ in sim:
                    pass
                sim.save_hotstart(str(hsf_out))

            out_path = str(Path(inp_runtime).with_suffix('.out'))
            node_levels: dict = {}
            try:
                with Output(out_path) as out:
                    for name in node_names:
                        try:
                            result = out.node_result(name, out.end)
                            node_levels[name] = float(list(result.values())[1])
                        except Exception:
                            node_levels[name] = 0.0
            except Exception:
                pass

            rpt_path = str(Path(inp_runtime).with_suffix('.rpt'))
            raw_flood = _total_inflow_from_rpt(rpt_path)

            data_1d: dict = {}
            for name in node_names:
                total_now = float(raw_flood.get(name, 0.0))
                prev_val = prev_total_flood.get(name, 0.0)
                delta = max(total_now - prev_val, 0.0)
                prev_total_flood[name] = total_now
                flow_m3s = delta * 1000.0 / max(window_dt, 1.0)
                data_1d[name] = {
                    'level': node_levels.get(name, 0.0),
                    'flow': flow_m3s,
                }

            t_1d_elapsed = time.perf_counter() - t_1d_start

            with shared['lock']:
                shared['1d_data'].clear()
                shared['1d_data'].update(data_1d)
                shared['1d_data']['__1d_elapsed__'] = t_1d_elapsed
                shared['1d_ready'].set()
                shared['2d_ready'].clear()

            window_step += 1

    finally:
        shared['stop'].set()
