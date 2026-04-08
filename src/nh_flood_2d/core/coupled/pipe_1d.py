"""
1D SWMM pipe-network subprocess — continuous stepping mode.

Contains:
  run_1d_pipe()        — main loop (subprocess target for multiprocessing.Process)
  _set_inp_end_time()  — adjust .inp END to allow continuous simulation
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import fastdb4py as fdb

from ...schema.feature import IndexLike, Node

if TYPE_CHECKING:
    from ...input.pipe import PipeConfig


# ─── .inp helper ─────────────────────────────────────────────────────────────────


def _set_inp_end_time(inp_path: str, total_seconds: float) -> None:
    """Set .inp END_DATE/END_TIME to START + total_seconds."""
    import re
    from datetime import datetime as _dt, timedelta

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


def _clear_inp_inflows(inp_path: str) -> None:
    """Zero out all baseline flow values in the [INFLOWS] section.

    SWMM adds .inp baseline inflows ON TOP of API-set inflows
    (node_set_total_inflow). Clearing them prevents double-counting.
    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    in_section = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '[INFLOWS]':
            in_section = True
            new_lines.append(line)
            continue
        if in_section and stripped.startswith('['):
            in_section = False
        if in_section and stripped and not stripped.startswith(';;'):
            parts = line.split()
            if len(parts) >= 7:
                parts[-1] = '0'
                new_lines.append(
                    f'{parts[0]:<15} {parts[1]:<17}{parts[2]:<17}'
                    f'{parts[3]:<9}{parts[4]:<9}{parts[5]:<9}{parts[6]}\n'
                )
                continue
        new_lines.append(line)

    with open(inp_path, 'w') as f:
        f.writelines(new_lines)


# ─── 1D subprocess entry point ───────────────────────────────────────────────────


def run_1d_pipe(shared, pipe_cfg: PipeConfig) -> None:
    """1D SWMM pipe-network loop — continuous stepping mode."""
    import shutil
    from pyswmm import Simulation, Nodes
    from swmm.toolkit import solver
    from swmm.toolkit.shared_enum import NodeResult

    try:
        # ── Setup ──
        inp_runtime = str(pipe_cfg.inp_runtime)
        shutil.copy2(pipe_cfg.inp, inp_runtime)

        # Load node metadata from pipe.fdb
        pipe_fdb = fdb.ORM.load(str(pipe_cfg.pipe_fdb), from_file=True)
        n_nodes = int(pipe_fdb[IndexLike]['node_count'][0].index)
        nodes_tbl = pipe_fdb[Node]['Node']
        node_names = [str(nodes_tbl[i].name) for i in range(n_nodes)]
        is_outfall = [bool(nodes_tbl[i].is_outfall) for i in range(n_nodes)]
        del pipe_fdb

        # Set END_TIME to 7 days ceiling; 2D controls actual stop
        _set_inp_end_time(inp_runtime, 7 * 24 * 3600)
        # Clear .inp baseline inflows to prevent double-counting with API calls
        _clear_inp_inflows(inp_runtime)

        coupling_interval = float(pipe_cfg.coupling_interval)

        with Simulation(inp_runtime) as sim:
            sim.step_advance(int(coupling_interval))

            # Build name → SWMM-internal-index map
            name_to_swmm_idx: dict = {}
            for i, node in enumerate(Nodes(sim)):
                name_to_swmm_idx[node.nodeid] = i

            # Fail-fast: verify all FDB node names exist in SWMM
            missing = set(node_names) - set(name_to_swmm_idx.keys())
            if missing:
                raise ValueError(
                    f'[1D] {len(missing)} nodes in pipe.fdb not found in '
                    f'SWMM model: {sorted(missing)[:10]}...'
                )

            # Pre-compute junction rim elevations for virtual surcharge
            # correction.  Reference sets SurDepth = 2D_water_depth each
            # window; the SWMM API forbids this during a running sim, so
            # we post-filter flood return using rim + 2D level instead.
            nodes_obj = Nodes(sim)
            junction_rim: dict = {}
            for i, name in enumerate(node_names):
                if is_outfall[i]:
                    continue
                try:
                    nobj = nodes_obj[name]
                    junction_rim[name] = (
                        nobj.invert_elevation + nobj.full_depth
                    )
                except Exception:
                    pass

            # ── Manual start (don't use for/iter — it advances before body) ──
            sim.start()

            window_step = 0

            while True:
                # ── Wait for 2D data ──
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

                # ── Apply 2D → 1D boundary conditions (BEFORE advancing) ──
                for i, name in enumerate(node_names):
                    swmm_idx = name_to_swmm_idx.get(name)
                    if swmm_idx is None:
                        continue
                    d = raw_2d.get(name, {})

                    if is_outfall[i]:
                        level = float(d.get('level', 0.0))
                        solver.outfall_set_stage(swmm_idx, level)
                    else:
                        # Drainage inflow (m³/s) — no conversion needed
                        flow = float(d.get('flow', 0.0))
                        solver.node_set_total_inflow(swmm_idx, flow)

                # ── Advance SWMM by one coupling interval ──
                elapsed = sim._model.swmm_step()
                if elapsed <= 0:
                    break  # SWMM reached END_TIME

                # ── Read 1D → 2D results (AFTER advancing) ──
                data_1d: dict = {}
                for i, name in enumerate(node_names):
                    swmm_idx = name_to_swmm_idx.get(name)
                    if swmm_idx is None:
                        data_1d[name] = {'level': 0.0, 'flow': 0.0}
                        continue

                    if is_outfall[i]:
                        level = solver.node_get_result(
                            swmm_idx, NodeResult.HEAD,
                        )
                        flow = solver.node_get_result(
                            swmm_idx, NodeResult.TOTAL_INFLOW,
                        )
                    else:
                        depth_1d = solver.node_get_result(
                            swmm_idx, NodeResult.DEPTH,
                        )
                        head = solver.node_get_result(
                            swmm_idx, NodeResult.HEAD,
                        )
                        flood = solver.node_get_result(
                            swmm_idx, NodeResult.FLOOD,
                        )

                        # Virtual surcharge correction: suppress flood
                        # return that is below the 2D water surface.
                        rim = junction_rim.get(name, 0.0)
                        level_2d = float(
                            raw_2d.get(name, {}).get('level', 0.0),
                        )
                        effective_rim = rim + level_2d

                        if flood <= 0.0 or head <= effective_rim:
                            flow = 0.0
                        else:
                            excess = head - effective_rim
                            swmm_excess = head - rim
                            if swmm_excess > 1e-6:
                                flow = flood * (excess / swmm_excess)
                            else:
                                flow = 0.0

                        level = depth_1d

                    data_1d[name] = {
                        'level': float(level),
                        'flow': float(flow),
                    }

                t_1d_elapsed = time.perf_counter() - t_1d_start

                # ── Send results to 2D ──
                with shared['lock']:
                    shared['1d_data'].clear()
                    shared['1d_data'].update(data_1d)
                    shared['1d_data']['__1d_elapsed__'] = t_1d_elapsed
                    shared['1d_ready'].set()
                    shared['2d_ready'].clear()

                window_step += 1

    finally:
        shared['stop'].set()
