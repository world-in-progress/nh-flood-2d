from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def absolute_time(value: str) -> datetime:
    """Parse an ISO 8601 timestamp without consulting the machine's timezone."""
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f'Timestamp must include a timezone: {value}')
    return result


class InputModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class BoundaryConfig(InputModel):
    name: str = Field(min_length=1)
    kind: Literal['level', 'discharge']
    line: str
    series: str


class RainfallConfig(InputModel):
    raster: str
    intervals: str


class GateConfig(InputModel):
    threshold: float
    land_type: int = Field(ge=0, le=255)
    level_boundary: str


class ForceConfig(InputModel):
    boundaries: list[BoundaryConfig]
    rainfall: RainfallConfig
    gate: GateConfig | None = None

    @model_validator(mode='after')
    def validate_names(self):
        names = [b.name for b in self.boundaries]
        if len(set(names)) != len(names):
            raise ValueError('Boundary names must be unique')
        if self.gate is not None and not any(
            b.name == self.gate.level_boundary and b.kind == 'level'
            for b in self.boundaries
        ):
            raise ValueError('gate.level_boundary must name a level boundary')
        return self

    def input_paths(self) -> list[str]:
        return [p for b in self.boundaries for p in (b.line, b.series)] + [
            self.rainfall.raster, self.rainfall.intervals,
        ]


def load_force_config(config_path: str) -> ForceConfig:
    return ForceConfig.model_validate_json(Path(config_path).read_text())
