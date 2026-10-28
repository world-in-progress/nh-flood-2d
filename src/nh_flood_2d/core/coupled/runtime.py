"""Surface-side lifetime and window protocol for a continuous SWMM actor."""

from __future__ import annotations

import math
import multiprocessing
import time
from datetime import timedelta

import fastdb4py as fdb
import numpy as np

from ...schema.feature import F32Value, IndexLike, Node, PipeTopo
from .exchange import compute_exchange
from .pipe_1d import run_1d_pipe


class PipeCoupling:
    """Own one SWMM process from warmup start through official simulation end.

    `begin_window` plans source rates from current 1D heads. `end_window`
    pairs measured 2D exchange with node inflow, then advances SWMM and
    refreshes heads before the next window can begin.
    """

    def __init__(self, domain_cfg, pipe_cfg):
        if domain_cfg.start.microsecond:
            raise ValueError('SWMM coupling requires domain.start at a whole second')
        self.domain_cfg = domain_cfg
        self.pipe_cfg = pipe_cfg
        self.absolute_start = domain_cfg.start - timedelta(seconds=domain_cfg.warmup_duration)
        self.absolute_end = domain_cfg.start + timedelta(seconds=domain_cfg.duration)
        self._elapsed_next = -float(domain_cfg.warmup_duration)
        self._conn = None
        self._process = None
        self._heads: dict[str, float] = {}
        self._pending: dict | None = None
        self._pending_plan: tuple | None = None
        self._windows: list[dict] = []
        self._swmm_seconds = 0.0
        self._actor_failed = False
        self._mesh_validated = False
        self._load_topology()

    def _load_topology(self) -> None:
        db = fdb.ORM.load(str(self.pipe_cfg.pipe_fdb), from_file=True)
        try:
            count = int(db[IndexLike]['node_count'][0].index)
            nodes = db[Node]['Node']
            self.node_names = [str(nodes[i].name) for i in range(count)]
            self.is_outfall = np.array([bool(nodes[i].is_outfall) for i in range(count)], dtype=bool)
            self.primary_ei = db[IndexLike]['node_primary_ei'].column.index.copy()
            self.topo_ei = db[PipeTopo]['PipeTopo'].column.ei.copy()
            self.topo_ptr = db[IndexLike]['topo_ptr'].column.index.copy()
            self.rims = db[F32Value]['node_rim'].column.value.copy()
        finally:
            del db
        if len(self.primary_ei) != count or len(self.rims) != count or len(self.topo_ptr) != count + 1:
            raise ValueError('pipe topology and node arrays have inconsistent sizes')
        if self.topo_ptr[0] != 0 or np.any(np.diff(self.topo_ptr.astype(np.int64)) < 0) or self.topo_ptr[-1] > len(self.topo_ei):
            raise ValueError('pipe topology pointer is invalid')
        if len(set(self.node_names)) != len(self.node_names):
            raise ValueError('pipe.fdb contains duplicate node names')

    def _receive(self, expected: str) -> dict:
        deadline = time.monotonic() + float(self.pipe_cfg.exchange_timeout)
        while not self._conn.poll(0.2):
            if self._process is not None and not self._process.is_alive():
                raise RuntimeError(f'SWMM actor exited with code {self._process.exitcode}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timeout waiting for SWMM {expected}')
        try:
            kind, payload = self._conn.recv()
        except EOFError as exc:
            raise RuntimeError('SWMM actor closed its pipe unexpectedly') from exc
        if kind == 'error':
            self._actor_failed = True
            raise RuntimeError(f"SWMM actor failed:\n{payload['traceback']}")
        if kind != expected:
            raise RuntimeError(f'Expected SWMM {expected}, received {kind}')
        return payload

    def __enter__(self):
        if self._process is not None:
            raise RuntimeError('PipeCoupling can only be entered once')
        ctx = multiprocessing.get_context('spawn')
        self._conn, child = ctx.Pipe(duplex=True)
        self._process = ctx.Process(
            target=run_1d_pipe,
            args=(child, str(self.pipe_cfg.inp), str(self.pipe_cfg.inp_runtime),
                  self.node_names, self.is_outfall.tolist(),
                  self.absolute_start, self.absolute_end),
            name='solver-swmm', daemon=False,
        )
        try:
            self._process.start()
            child.close()
            self._heads = self._receive('ready')['heads']
            return self
        except BaseException:
            child.close()
            self._stop_actor()
            raise

    def begin_window(
        self,
        elapsed_seconds_from_start: float,
        window_dt: float,
        depth_np: np.ndarray,
        bed_np: np.ndarray,
        area_np: np.ndarray,
        land_types_np: np.ndarray,
    ) -> np.ndarray:
        if self._process is None or self._conn is None:
            raise RuntimeError('PipeCoupling is not entered')
        if self._pending is not None:
            raise RuntimeError('end_window must finish before begin_window')
        if not self._mesh_validated:
            if not (depth_np.shape == bed_np.shape == area_np.shape == land_types_np.shape):
                raise ValueError('Surface depth, bed, area and land types must share one shape')
            linked = np.concatenate((self.primary_ei, self.topo_ei[:int(self.topo_ptr[-1])]))
            if np.any(linked >= len(depth_np)):
                raise ValueError('pipe topology cell index is outside the surface mesh')
            self._mesh_validated = True
        elapsed = float(elapsed_seconds_from_start)
        if abs(elapsed - self._elapsed_next) > 1e-5:
            raise ValueError(f'Pipe window starts at {elapsed}, expected {self._elapsed_next}')
        seconds = round(float(window_dt))
        if seconds <= 0 or not math.isclose(float(window_dt), seconds, abs_tol=1e-5):
            raise ValueError('SWMM exchange windows must contain positive whole seconds')
        if elapsed + seconds > float(self.domain_cfg.duration) + 1e-5:
            raise ValueError('Pipe window extends past simulation end')
        if elapsed < 0 < elapsed + seconds:
            raise ValueError('Pipe window must end at zero before official simulation')
        source, boundaries, diagnostic, relationships = compute_exchange(
            seconds, depth_np, bed_np, area_np, land_types_np,
            self.node_names, self.is_outfall, self.primary_ei,
            self.topo_ei, self.topo_ptr, self.rims, self._heads,
        )
        self._pending_plan = (source, boundaries, relationships)
        self._pending = {
            'elapsed_start_s': elapsed,
            'elapsed_end_s': elapsed + seconds,
            'absolute_start': (self.domain_cfg.start + timedelta(seconds=elapsed)).isoformat(),
            'absolute_end': (self.domain_cfg.start + timedelta(seconds=elapsed + seconds)).isoformat(),
            'window_seconds': seconds,
            'swmm_before_s': self._swmm_seconds,
            **diagnostic,
        }
        return source

    def end_window(self, actual_surface_volumes: np.ndarray) -> None:
        """Advance SWMM after pairing measured 2D volume with node inflow."""
        if self._pending is None or self._pending_plan is None:
            raise RuntimeError('No open pipe exchange window')
        planned_source, planned_boundaries, relationships = self._pending_plan
        actual = np.asarray(actual_surface_volumes, dtype=np.float64)
        if actual.shape != planned_source.shape or not np.all(np.isfinite(actual)):
            raise ValueError('actual_surface_volumes must be finite and match the surface mesh')
        seconds = self._pending['window_seconds']
        requested = planned_source.astype(np.float64) * seconds
        delta = actual - requested
        boundaries = {name: dict(values) for name, values in planned_boundaries.items()}
        returned = 0.0
        for ei, change in enumerate(delta):
            if change == 0.:
                continue
            attached = relationships[ei]
            if not attached:
                raise ValueError(f'Unpaired surface/pipe exchange at cell {ei}')
            correction = change / (seconds * len(attached))
            for node in attached:
                boundaries[self.node_names[node]]['flow'] -= correction
            if requested[ei] < 0. and change > 0.:
                returned += change
        requested_swmm = sum(v['flow'] for v in planned_boundaries.values()) * seconds
        applied_swmm = sum(v['flow'] for v in boundaries.values()) * seconds
        balance = float(actual.sum(dtype=np.float64) + applied_swmm)
        tolerance = 1e-9 + 1e-12 * float(np.abs(actual).sum(dtype=np.float64))
        if abs(balance) > tolerance:
            raise ArithmeticError(f'Surface/SWMM exchange mismatch {balance} m3')
        self._pending.update(
            requested_surface_exchange_m3=float(requested.sum(dtype=np.float64)),
            applied_surface_exchange_m3=float(actual.sum(dtype=np.float64)),
            requested_swmm_inflow_m3=float(requested_swmm),
            applied_swmm_inflow_m3=float(applied_swmm),
            limited_drainage_return_m3=float(returned),
            surface_swmm_balance_error_m3=balance,
        )
        self._conn.send(('advance', {'seconds': seconds, 'boundaries': boundaries}))
        payload = self._receive('advanced')
        actual = float(payload['swmm_seconds'])
        expected = self._swmm_seconds + self._pending['window_seconds']
        if abs(actual - expected) > 0.05:
            raise RuntimeError(f'SWMM cumulative clock {actual}s differs from {expected}s')
        self._heads = payload['heads']
        self._swmm_seconds = actual
        self._pending['swmm_after_s'] = actual
        self._windows.append(self._pending)
        self._elapsed_next = self._pending['elapsed_end_s']
        self._pending = None
        self._pending_plan = None

    def metadata(self) -> dict:
        warmup = sum(w['window_seconds'] for w in self._windows if w['elapsed_end_s'] <= 0)
        official = sum(w['window_seconds'] for w in self._windows if w['elapsed_start_s'] >= 0)
        zero_index = next((i for i, w in enumerate(self._windows)
                           if w['elapsed_end_s'] == 0), None)
        continuous = (zero_index is None or zero_index + 1 == len(self._windows) or
                      abs(self._windows[zero_index + 1]['swmm_before_s'] -
                          self._windows[zero_index]['swmm_after_s']) < 0.05)
        return {
            'absolute_start': self.absolute_start.isoformat(),
            'absolute_end': self.absolute_end.isoformat(),
            'swmm_cumulative_seconds': self._swmm_seconds,
            'warmup_completed_seconds': warmup,
            'official_completed_seconds': official,
            'continuous_clock_at_zero': continuous,
            'window_count': len(self._windows),
            'pending_window': self._pending is not None,
            'windows': [dict(w) for w in self._windows],
        }

    def _stop_actor(self) -> None:
        process = self._process
        conn = self._conn
        if process is None:
            return
        error = None
        try:
            if process.is_alive() and conn is not None and not self._actor_failed:
                conn.send(('close', {}))
                while True:
                    kind = self._receive_any()
                    if kind == 'closed':
                        break
            process.join(timeout=float(self.pipe_cfg.exchange_timeout))
            if process.is_alive():
                raise TimeoutError('SWMM actor did not exit after close')
            if process.exitcode not in (0, None) and not self._actor_failed:
                raise RuntimeError(f'SWMM actor exited with code {process.exitcode}')
        except BaseException as exc:
            error = exc
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        finally:
            if conn is not None:
                conn.close()
            self._conn = None
            self._process = None
        if error is not None:
            raise error

    def _receive_any(self) -> str:
        deadline = time.monotonic() + float(self.pipe_cfg.exchange_timeout)
        while not self._conn.poll(0.2):
            if self._process is not None and not self._process.is_alive():
                raise RuntimeError(f'SWMM actor exited with code {self._process.exitcode}')
            if time.monotonic() >= deadline:
                raise TimeoutError('Timeout waiting for SWMM actor to close')
        kind, payload = self._conn.recv()
        if kind == 'error':
            raise RuntimeError(f"SWMM actor failed:\n{payload['traceback']}")
        return kind

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._stop_actor()
        except BaseException as close_error:
            if exc_value is not None:
                raise close_error from exc_value
            raise
        return False
