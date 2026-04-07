from pathlib import Path
from pydantic import BaseModel, field_validator

def load_pipe_config(config_path: str) -> 'PipeConfig':
    return PipeConfig.model_validate_json(Path(config_path).read_text())

class PipeConfig(BaseModel):
    inp: str                          # SWMM .inp 原始文件路径
    pipe_dir: str                     # 管网预处理/运行输出目录

    coupling_interval: float = 300.0  # 交换周期（s）
    exchange_timeout: float  = 300.0  # 等待对端数据的超时（s）

    @property
    def _pipe_path(self) -> Path:
        p = Path(self.pipe_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def pipe_fdb(self) -> str:
        return str(self._pipe_path / 'pipe.fdb')

    @property
    def inp_runtime(self) -> str:
        """运行时 .inp 副本路径（预处理/求解器写入此文件，不修改原始文件）"""
        return str(self._pipe_path / Path(self.inp).name)

    @property
    def hotstart_dir(self) -> str:
        p = self._pipe_path / 'hotstart'
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    @field_validator('inp')
    def validate_inp(cls, v):
        if not Path(v).exists():
            raise ValueError(f'SWMM .inp not found: {v}')
        return v

    @field_validator('pipe_dir')
    def validate_pipe_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
