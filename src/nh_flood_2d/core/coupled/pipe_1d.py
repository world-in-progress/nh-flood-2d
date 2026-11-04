"""A single, continuous SWMM actor driven by explicit exchange windows."""

from __future__ import annotations

import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing.connection import Connection


def _prepare_inp(source: str, runtime: str, start: datetime, end: datetime) -> None:
    """Write all SWMM runtime changes into pipe_dir; leave the source intact."""
    if Path(source).resolve() == Path(runtime).resolve():
        raise ValueError('pipe.inp and inp_runtime must be different files')
    shutil.copy2(source, runtime)
    lines = Path(runtime).read_text().splitlines(keepends=True)
    options = {
        'START_DATE': start.strftime('%m/%d/%Y'),
        'START_TIME': start.strftime('%H:%M:%S'),
        'REPORT_START_DATE': start.strftime('%m/%d/%Y'),
        'REPORT_START_TIME': start.strftime('%H:%M:%S'),
        'END_DATE': end.strftime('%m/%d/%Y'),
        'END_TIME': end.strftime('%H:%M:%S'),
    }
    seen = set()
    section = ''
    flow_units = None
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('['):
            section = stripped.lower()
        parts = stripped.split()
        if section == '[options]' and len(parts) >= 2 and parts[0].upper() == 'FLOW_UNITS':
            flow_units = parts[1].upper()
        if section == '[options]' and parts and parts[0].upper() in options:
            key = parts[0].upper()
            result.append(f'{key:<20} {options[key]}\n')
            seen.add(key)
        elif section == '[junctions]' and parts and not stripped.startswith(';') and len(parts) >= 5:
            parts[4] = '1000.0'
            result.append('  '.join(parts) + '\n')
        elif section == '[inflows]' and parts and not stripped.startswith(';') and len(parts) >= 7:
            parts[6] = '0'  # Baseline, followed by optional Pattern
            result.append('  '.join(parts) + '\n')
        else:
            result.append(line)
    if flow_units != 'CMS':
        raise ValueError('Coupled SWMM FLOW_UNITS must be CMS (m3/s and metres)')
    missing = options.keys() - seen
    if missing:
        raise ValueError(f'SWMM [OPTIONS] missing {sorted(missing)}')
    Path(runtime).write_text(''.join(result))


def _heads(node_names: list[str], indexes: dict[str, int]) -> dict[str, float]:
    from swmm.toolkit import solver
    from swmm.toolkit.shared_enum import NodeResult

    return {name: float(solver.node_get_result(indexes[name], NodeResult.HEAD))
            for name in node_names}


def run_1d_pipe(
    conn: Connection,
    source_inp: str,
    runtime_inp: str,
    node_names: list[str],
    is_outfall: list[bool],
    start: datetime,
    end: datetime,
) -> None:
    """Actor protocol: ready(heads), advance(boundaries, seconds), close."""
    try:
        from pyswmm import Simulation, Nodes
        from swmm.toolkit import solver

        _prepare_inp(source_inp, runtime_inp, start, end)
        with Simulation(runtime_inp) as sim:
            # SWMM's Julian-day stop comparison can discard the final routing
            # second for starts away from a minute boundary. The .inp retains
            # the requested end; this in-memory guard lets the last stride
            # reach that exact end without running a second beyond it.
            sim.end_time = end.replace(tzinfo=None) + timedelta(seconds=1)
            indexes = {node.nodeid: i for i, node in enumerate(Nodes(sim))}
            missing = set(node_names) - indexes.keys()
            if missing:
                raise ValueError(f'pipe.fdb nodes missing from SWMM: {sorted(missing)}')
            sim.start()
            conn.send(('ready', {'heads': _heads(node_names, indexes)}))
            last_seconds = 0.0
            while True:
                command, payload = conn.recv()
                if command == 'close':
                    conn.send(('closed', {'swmm_seconds': last_seconds}))
                    break
                if command != 'advance':
                    raise ValueError(f'Unknown SWMM actor command: {command}')
                seconds = int(payload['seconds'])
                if seconds <= 0:
                    raise ValueError('SWMM window must contain positive whole seconds')
                boundaries = payload['boundaries']
                for name, outfall in zip(node_names, is_outfall):
                    index = indexes[name]
                    if outfall:
                        solver.outfall_set_stage(index, float(boundaries[name]['level']))
                    else:
                        solver.node_set_total_inflow(index, float(boundaries[name]['flow']))
                sim._model.swmm_stride(seconds)
                elapsed = (sim.current_time - start.replace(tzinfo=None)).total_seconds()
                expected = last_seconds + seconds
                if abs(elapsed - expected) > 0.05:
                    raise RuntimeError(
                        f'SWMM advanced {elapsed - last_seconds:.6f}s, expected {seconds}s '
                        f'(cumulative {elapsed:.6f}s vs {expected:.6f}s)'
                    )
                last_seconds = elapsed
                conn.send(('advanced', {'heads': _heads(node_names, indexes),
                                        'swmm_seconds': elapsed}))
    except BaseException as error:
        try:
            conn.send(('error', {'traceback': traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError) as delivery_error:
            raise error from delivery_error
        raise
    finally:
        conn.close()
