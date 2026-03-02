from .pass_1 import build_fdbs
from .pass_2 import build_boundary_fdb

def preprocess(cfg):
    build_fdbs(cfg)
    build_boundary_fdb(cfg)