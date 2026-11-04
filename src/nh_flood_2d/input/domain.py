from pathlib import Path
from typing import Annotated, Literal
from pydantic import AwareDatetime, Field, field_validator, model_validator
from .force import InputModel

def load_domain_config(config_path: str) -> 'DomainConfig':
    """Load input configuration from a JSON file"""
    return DomainConfig.model_validate_json(Path(config_path).read_text())


class HortonInfiltration(InputModel):
    f0_mm_h: float = Field(ge=0)
    fc_mm_h: float = Field(ge=0)
    k_per_h: float = Field(ge=0)

    @model_validator(mode='after')
    def validate_capacity(self):
        if self.f0_mm_h < self.fc_mm_h:
            raise ValueError('f0_mm_h must be at least fc_mm_h')
        return self


class LandParameters(InputModel):
    manning_n: float = Field(ge=0)
    infiltration: Literal['none'] | HortonInfiltration


class DomainConfig(InputModel):
    ne: str
    ns: str
    epsg_code: int
    domain_dir: str
    
    afa: float = Field(default=0.5, gt=0, le=1)        # Courant number (CFL condition)
    sita: float = Field(default=1.0, ge=0, le=1)       # time weighting factor
    min_h: float = Field(default=0.02, gt=0)     # minimum water depth (m)
    
    start: AwareDatetime = Field(strict=True)
    duration: int = Field(gt=0)  # simulation seconds
    warmup_duration: int = Field(ge=0)
    land_parameters: dict[Annotated[int, Field(ge=0, le=255)],
                          LandParameters] = Field(min_length=1)
    yield_step: int = Field(default=300, gt=0)  # output interval in seconds
    
    restart_state: str = ''  # complete checkpoint, empty for a cold start
    
    stations: dict[str, tuple[float, float]] = {}  # name -> (x, y)
    observation_dir: str = ''  # directory for observation data, file_name should be the same as stations keys
    
    @property
    def _domain_path(self) -> Path:
        return Path(self.domain_dir)
    
    @property
    def _preprocessed_path(self) -> Path:
        path = self._domain_path / 'preprocessed'
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def tmp_ne(self) -> str:
        return str(self._domain_path / 'temp_ne.txt')
    
    @property
    def tmp_ns(self) -> str:
        return str(self._domain_path / 'temp_ns.txt')
    
    @property
    def ne_fdb(self) -> str:
        ne_fdb_fn = self._preprocessed_path / 'ne.fdb'
        return str(ne_fdb_fn)
    
    @property
    def ns_fdb(self) -> str:
        ns_fdb_fn = self._preprocessed_path / 'ns.fdb'
        return str(ns_fdb_fn)
    
    @property
    def boundary_fdb(self) -> str:
        boundary_fdb_fn = self._preprocessed_path / 'boundary.fdb'
        return str(boundary_fdb_fn)
    
    @property
    def output_dir(self) -> str:
        dir_path = self._domain_path / 'output'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)
    
    @property
    def frame_dir(self) -> str:
        path = Path(self.output_dir) / 'frames'
        path.mkdir(exist_ok=True)
        return str(path)

    @model_validator(mode='after')
    def validate_clock_range(self):
        if self.duration + self.warmup_duration >= 2**24:
            raise ValueError('Duration plus warmup must fit the two-U32 clock (less than 2**24 seconds)')
        return self

    @property
    def flood_map_dir(self) -> str:
        dir_path = self._domain_path / 'flood_maps'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)

    @property
    def max_inundation_dir(self) -> str:
        dir_path = self._domain_path / 'max_inundation'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)

    def clean_tmp_files(self):
        for tmp_file in [self.tmp_ne, self.tmp_ns]:
            tmp_path = Path(tmp_file)
            if tmp_path.exists():
                tmp_path.unlink()
    
    @field_validator('ne', 'ns')
    def validate_paths(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Path does not exist: {v}')
        return v
    
    @field_validator('domain_dir')
    def validate_domain_dir(cls, v):
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return v
