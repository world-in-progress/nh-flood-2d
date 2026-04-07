from .force import prepare_force
from .domain import prepare_domain
from .pipe import prepare_pipe
from ..input import ForceConfig, DomainConfig
from ..input.pipe import PipeConfig

def preprocess(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    prepare_force(force_cfg)
    prepare_domain(domain_cfg)