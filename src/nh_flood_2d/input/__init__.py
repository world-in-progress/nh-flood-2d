from pathlib import Path
from pydantic import BaseModel, field_validator

def load_input_config(config_path: str) -> 'InputConfig':
    """Load input configuration from a JSON file"""
    return InputConfig.model_validate_json(Path(config_path).read_text())

class InputConfig(BaseModel):
    ne: str
    ns: str
    gate: str
    tide: str
    rain: str
    epsg_code: int
    output_dir: str
    
    afa: float = 0.5        # Courant number (CFL condition)
    sita: float = 1.0       # time weighting factor
    min_h: float = 0.02     # minimum water depth (m)
    
    duration: int = -1      # total simulation duration in seconds (default: -1 means auto-detect from input data)
    yield_step: int = 300   # output every 5 minutes (300 seconds) as default
    
    hydrograph_points: dict[str, tuple[float, float]] = {}  # name -> (x, y)
    observation_dir: str = ''  # directory for observation data, file_name should be the same as hydrograph_points keys
    
    @property
    def tmp_ne(self) -> str:
        return str(Path(self.ne).parent / 'temp_ne.txt')
    
    @property
    def tmp_ns(self) -> str:
        return str(Path(self.ns).parent / 'temp_ns.txt')
    
    @property
    def ne_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'ne.fdb')
    
    @property
    def ns_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'ns.fdb')
    
    @property
    def boundary_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'boundary.fdb')
    
    @property
    def node_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'node.fdb')
    
    @property
    def gate_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'gate.fdb')
    
    @property
    def tide_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'tide.fdb')
    
    @property
    def rain_fdb(self) -> str:
        return str(Path(self.output_dir) / 'preprocessed' / 'rain.fdb')
    
    @property
    def uvh_dir(self) -> str:
        dir_path = Path(self.output_dir) / 'uvh'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)
    
    @property
    def flood_map_dir(self) -> str:
        dir_path = Path(self.output_dir) / 'flood_maps'
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        return str(dir_path)
    
    @property
    def hydrograph_dir(self) -> str:
        dir_path = Path(self.output_dir) / 'hydrographs'
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
    
    @field_validator('output_dir')
    def validate_output_dir(cls, v):
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        preprocess_dir = path / 'preprocessed'
        if not preprocess_dir.exists():
            preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        return v