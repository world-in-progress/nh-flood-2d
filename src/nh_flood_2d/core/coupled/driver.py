"""
Coupled 2D surface-water / 1D pipe-network solver — process driver.

Orchestrates subprocess spawning and lifecycle:
  - 2D GPU hydrodynamic process  (flood_2d.run_2d)
  - 1D CPU SWMM pipe process     (pipe_1d.run_1d_pipe)

When pipe_cfg is None, runs 2D-only mode with no IPC infrastructure.
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from .flood_2d import run_2d
from .pipe_1d import run_1d_pipe

if TYPE_CHECKING:
    from ...input import DomainConfig, ForceConfig
    from ...input.pipe import PipeConfig


def solver_coupled(
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None = None,
    start_time_step: int = 0,
) -> None:
    """
    Coupled 2D/1D solver entry point.

    When *pipe_cfg* is provided, spawns two subprocesses using the 'spawn'
    context (required for Taichi GPU):
      run_2d()       – GPU 2D hydrodynamic simulation
      run_1d_pipe()  – CPU SWMM pipe-network simulation
    The two processes exchange data every coupling_interval seconds via shared
    multiprocessing.Manager dicts and Events.

    When *pipe_cfg* is None, runs 2D-only mode — a single subprocess with no
    IPC infrastructure.
    """
    has_pipe = pipe_cfg is not None
    ctx = multiprocessing.get_context('spawn')

    if has_pipe:
        print(f'[coupled] coupling_interval = '
              f'{pipe_cfg.coupling_interval:.0f}s')

        mgr = ctx.Manager()
        shared = {
            '2d_data':  mgr.dict(),
            '1d_data':  mgr.dict(),
            '2d_ready': mgr.Event(),
            '1d_ready': mgr.Event(),
            'lock':     mgr.Lock(),
            'stop':     mgr.Event(),
        }
    else:
        shared = {
            '2d_data':  {},
            '1d_data':  {},
            'stop':     ctx.Event(),
        }

    p2d = ctx.Process(
        target=run_2d,
        args=(shared, domain_cfg, force_cfg, pipe_cfg, start_time_step),
        daemon=False,
        name='solver-2d',
    )

    p1d = None
    if has_pipe:
        p1d = ctx.Process(
            target=run_1d_pipe,
            args=(shared, pipe_cfg),
            daemon=False,
            name='solver-1d',
        )

    p2d.start()
    if p1d is not None:
        p1d.start()
        print(f'[coupled] 2D pid={p2d.pid}  1D pid={p1d.pid}')
    else:
        print(f'[2D-only] pid={p2d.pid}')

    try:
        p2d.join()
        if has_pipe:
            shared['stop'].set()
        if p1d is not None:
            p1d.join(timeout=120.0)
    except KeyboardInterrupt:
        print('[coupled] KeyboardInterrupt – signalling stop.')
        if has_pipe:
            shared['stop'].set()
        p2d.join(timeout=30.0)
        if p1d is not None:
            p1d.join(timeout=30.0)
    finally:
        if p2d.is_alive():
            p2d.terminate()
        if p1d is not None and p1d.is_alive():
            p1d.terminate()
        if has_pipe:
            mgr.shutdown()
        print('[coupled] Both processes terminated.')
