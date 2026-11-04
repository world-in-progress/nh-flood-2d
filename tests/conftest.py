"""Each independent physics test owns its Taichi runtime and fields."""
import pytest
import taichi as ti

from src.nh_flood_2d.util import ti as ti_runtime


@pytest.fixture
def taichi_cpu():
    ti.reset()
    ti_runtime._TI_INITIALIZED = False
    ti_runtime.init_taichi(use_gpu=False)
    yield
    ti.reset()
    ti_runtime._TI_INITIALIZED = False
