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


class GatePointConfig(InputModel):
    name: str = Field(min_length=1)
    inside_point: tuple[float, float]
    control_point: tuple[float, float]


class GateConfig(InputModel):
    land_type: int = Field(ge=0, le=255)
    pond_land_type: int = Field(ge=0, le=255)
    enabled: bool
    trigger_level: float
    opening_seconds: float = Field(gt=0)
    open_elevation: float
    gates: list[GatePointConfig] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_gates(self):
        if self.pond_land_type == self.land_type:
            raise ValueError('pond_land_type must differ from gate land_type')
        names = [gate.name for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValueError('Gate names must be unique')
        return self


class ForceConfig(InputModel):
    boundaries: list[BoundaryConfig]
    rainfall: RainfallConfig
    gate: GateConfig | None = None

    @model_validator(mode='after')
    def validate_names(self):
        names = [b.name for b in self.boundaries]
        if len(set(names)) != len(names):
            raise ValueError('Boundary names must be unique')
        return self

    def input_paths(self) -> list[str]:
        return [p for b in self.boundaries for p in (b.line, b.series)] + [
            self.rainfall.raster, self.rainfall.intervals,
        ]


def load_force_config(config_path: str) -> ForceConfig:
    return ForceConfig.model_validate_json(Path(config_path).read_text())
