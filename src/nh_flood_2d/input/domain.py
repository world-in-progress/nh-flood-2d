from pathlib import Path
from pydantic import BaseModel, field_validator

def load_domain_config(config_path: str) -> 'DomainConfig':
    """Load input configuration from a JSON file"""
    return DomainConfig.model_validate_json(Path(config_path).read_text())

class DomainConfig(BaseModel):
    ne: str
    ns: str
    epsg_code: int
    domain_dir: str
    
    afa: float = 0.5        # Courant number (CFL condition)
    sita: float = 1.0       # time weighting factor
    min_h: float = 0.02     # minimum water depth (m)
    
    duration: int = -1      # total simulation duration in seconds (default: -1 means auto-detect from input data)
    yield_step: int = 300   # output every 5 minutes (300 seconds) as default
    
    hydrograph_points: dict[str, tuple[float, float]] = {}  # name -> (x, y)
    observation_dir: str = ''  # directory for observation data, file_name should be the same as hydrograph_points keys
    
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
    def uvh_dir(self) -> str:
        dir_path = self._domain_path / 'uvh'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)
    
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

    @property
    def hydrograph_dir(self) -> str:
        dir_path = self._domain_path / 'hydrographs'
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