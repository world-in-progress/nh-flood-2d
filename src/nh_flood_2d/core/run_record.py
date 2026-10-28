"""Host float64 water accounting and reproducible per-run metadata."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import taichi as ti


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RunRecord:
    def __init__(self, domain, force, pipe=None):
        root = Path(__file__).resolve().parents[3]
        self.path = Path(domain.domain_dir) / 'manifest.json'
        output = Path(domain.domain_dir) / 'output'
        if self.path.exists() or (self.path.parent / 'mass_balance.csv').exists() or (output.exists() and any(output.iterdir())):
            raise FileExistsError(f'Use a new output directory: {domain.domain_dir}')
        self.clock = time.perf_counter()
        self.data = {
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
            'config': {'domain': domain.model_dump(mode='json'), 'force': force.model_dump(mode='json'), 'pipe': pipe.model_dump(mode='json') if pipe else None},
            'input_sha256': {},
            'simulation_start': domain.start.isoformat(),
            'warmup_start': (domain.start - timedelta(seconds=domain.warmup_duration)).isoformat(),
            'simulation_end': (domain.start + timedelta(seconds=domain.duration)).isoformat(),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'steps': 0, 'warmup_steps': 0, 'backend': None, 'device_name': None,
            'peak_device_memory_bytes': 'unavailable', 'status': 'running',
            'budget_accumulator': {
                'device_step_reduction': 'float32 atomic sums in blocks of 256 cells',
                'device_time_reduction': 'exact int32 sign-separated base-65536 words, quantum 2^-149',
                'host_reduction': 'exact integer merge across blocks and outputs, then float64 conversion',
            },
        }
        self.write()
    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.data['error'] = str(error)
        self.finish('failed' if error_type else 'completed')

    def hash_inputs(self, domain, force, pipe=None):
        for path in [domain.ne, domain.ns, *force.input_paths()] + ([domain.restart_state] if domain.restart_state else []) + ([pipe.inp] if pipe else []):
            self.data['input_sha256'][str(Path(path).resolve())] = sha256(path)
        self.write()

    def device(self):
        arch = ti.lang.impl.current_cfg().arch
        self.data['backend'] = arch.name
        if arch == ti.cuda:
            # CUDA logical device 0 is the first visible physical device.
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
            self.data['device_name'] = subprocess.check_output(
                ['nvidia-smi', f'--id={visible}', '--query-gpu=name', '--format=csv,noheader'],
                text=True).strip()
        elif arch == ti.metal or arch == ti.cpu:
            self.data['device_name'] = (subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                if platform.system() == 'Darwin' else platform.processor() or platform.machine())
        else:
            # Taichi does not expose a public device-name query for these backends.
            self.data['device_name'] = 'unavailable'

    def write(self):
        self.path.write_text(json.dumps(self.data, indent=2) + '\n')

    def finish(self, status):
        self.data.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(),
                         wall_seconds=time.perf_counter() - self.clock)
        self.write()


class WaterBudget:
    def __init__(self, path, boundaries, initial_storage):
        self.boundaries = boundaries
        self.initial = float(initial_storage)
        self.rain = np.float64(0)
        self.discharge = np.zeros(len(boundaries), dtype=np.float64)
        # Each level boundary keeps five physical source components separate.
        # Its intermediate floor is diagnostic only, since level overwrites it.
        self.device_terms = np.zeros(4 + 6 * len(boundaries), dtype=np.float64)
        self.file = open(path, 'w', newline='')
        fields = ['time', 'elapsed_seconds', 'phase', 'rainfall_m3', 'infiltration_m3']
        fields += [f'{b.kind}:{b.name}_m3' for b in boundaries]
        fields += [f'requested_discharge:{b.name}_m3' for b in boundaries if b.kind == 'discharge']
        fields += ['bed_clamp_nonlevel_m3', 'bed_clamp_level_m3', 'gate_bed_m3', 'pipe_exchange_m3',
                   'initial_storage_m3', 'storage_m3', 'residual_m3', 'relative_residual']
        self.writer = csv.DictWriter(self.file, fields)
        self.writer.writeheader()
        self.last_row = None

    def add_forcing(self, dt, rain_rate_volume, boundary_values):
        self.rain += np.float64(dt) * rain_rate_volume
        for i, b in enumerate(self.boundaries):
            if b.kind == 'discharge':
                self.discharge[i] += np.float64(dt) * boundary_values[i]

    def set_device_terms(self, cumulative_terms):
        self.device_terms = cumulative_terms

    def output(self, date, seconds, storage, phase):
        terms = self.device_terms
        levels = terms[4:4+5*len(self.boundaries)].reshape(len(self.boundaries), 5).sum(axis=1, dtype=np.float64)
        floor, level_floor, pipe = terms[1], terms[2], terms[3]
        inputs = self.rain - terms[0] + floor + pipe
        actual_discharge = terms[4+5*len(self.boundaries):]
        inputs += actual_discharge.sum(dtype=np.float64) + levels.sum(dtype=np.float64)
        residual = float(storage - self.initial - inputs)
        total_in = self.rain + np.maximum(actual_discharge, 0).sum() + np.maximum(levels, 0).sum() + max(pipe, 0.)
        scale = max(self.initial, float(total_in), float(np.abs(actual_discharge).sum() + np.abs(terms[[0, 1, 3]]).sum() + np.abs(levels).sum()))
        row = dict(time=date.isoformat(), elapsed_seconds=seconds, phase=phase,
                   rainfall_m3=float(self.rain), infiltration_m3=float(terms[0]),
                   bed_clamp_nonlevel_m3=float(floor), bed_clamp_level_m3=float(level_floor), gate_bed_m3=0.,
                   pipe_exchange_m3=float(pipe), initial_storage_m3=self.initial,
                   storage_m3=float(storage), residual_m3=residual,
                   relative_residual=residual / scale if scale else None)
        for i, b in enumerate(self.boundaries):
            row[f'{b.kind}:{b.name}_m3'] = float(actual_discharge[i] if b.kind == 'discharge' else levels[i])
            if b.kind == 'discharge':
                row[f'requested_discharge:{b.name}_m3'] = float(self.discharge[i])
        self.writer.writerow(row)
        self.file.flush()
        self.last_row = row
        return row

    def close(self):
        self.file.close()
