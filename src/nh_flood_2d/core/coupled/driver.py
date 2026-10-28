"""Compatibility entry point into the single 2D evolution engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...input import DomainConfig, ForceConfig
    from ...input.pipe import PipeConfig


def solver_coupled(
    domain_cfg: DomainConfig,
    force_cfg: ForceConfig,
    pipe_cfg: PipeConfig | None = None,
) -> None:
    """Run an already preprocessed domain with optional SWMM coupling."""
    from ..solver_compact import solver

    return solver(domain_cfg, force_cfg, pipe_cfg)
