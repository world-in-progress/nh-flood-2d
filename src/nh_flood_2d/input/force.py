from pathlib import Path
from pydantic import BaseModel, field_validator

def load_force_config(config_path: str) -> 'ForceConfig':
    return ForceConfig.model_validate_json(Path(config_path).read_text())

class ForceConfig(BaseModel):
    gate: str
    tide: str
    rain: str
    force_dir: str
    
    @property
    def _force_path(self) -> Path:
        return Path(self.force_dir)
    
    @property
    def _preprocessed_path(self) -> Path:
        path = self._force_path / 'preprocessed'
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def gate_fdb(self) -> str:
        gate_fdb_fn = self._preprocessed_path / 'gate.fdb'
        return str(gate_fdb_fn)
    
    @property
    def tide_fdb(self) -> str:
        tide_fdb_fn = self._preprocessed_path / 'tide.fdb'
        return str(tide_fdb_fn)
    
    @property
    def rain_fdb(self) -> str:
        rain_fdb_fn = self._preprocessed_path / 'rain.fdb'
        return str(rain_fdb_fn)