import numpy as np
import taichi as ti

_TI_INITIALIZED = False
def init_taichi(use_gpu: bool = True, profiler: bool = False):
    global _TI_INITIALIZED
    if not _TI_INITIALIZED:
        arch = ti.gpu if use_gpu else ti.cpu
        ti.init(arch = arch, kernel_profiler=profiler, log_level=ti.ERROR)
        _TI_INITIALIZED = True
    
def copy_to_taichi(np_array: np.ndarray, dtype: any, shape: any) -> ti.MatrixField | ti.ScalarField:
    """
    Copy a numpy array to a Taichi field.
    This function can facilitate the conversion from column data in FastDB table to Taichi field.
    
    Note: tichi not initialized inside this function, Taichi should be initialized before calling this function.
    
    e.g.:
        column_data = nes.column.x  # numpy array from FastDB
        field = copy_to_taichi(column_data, ti.f32, None)
    """
    if shape is None:
        shape = np_array.shape
    else:
        if np_array.shape != shape:
            np_array = np_array.reshape(shape)
        
    field = ti.field(dtype=dtype, shape=np_array.shape if shape is None else shape)
    field.from_numpy(np_array)
    return field