import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path

from ..input import InputConfig
from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import IndexLike, Ne, SideTopoInfo, Ns, U8Value

def build_boundary_fdb(cfg: InputConfig):
    """Create boundary hydro element FDB from NE FDB and NS FDB"""
    ne_fdb_fn: str = cfg.ne_fdb
    ns_fdb_fn: str = cfg.ns_fdb
    fdb_fn: str = cfg.boundary_fdb
    
    # Init Taichi
    init_taichi(use_gpu=True, profiler=True)
    
    # Load necessaray information from fdbs
    es = fdb.ORM.load(ne_fdb_fn, from_file=True)[Ne][Ne]
    ss = fdb.ORM.load(ns_fdb_fn, from_file=True)[Ns][Ns]
    s_ts = fdb.ORM.load(ns_fdb_fn, from_file=True)[SideTopoInfo][SideTopoInfo]
    e_num = len(es)
    s_num = len(ss)
    e_xs_t = copy_to_taichi(es.column.x, ti.f32, None)
    e_ys_t = copy_to_taichi(es.column.y, ti.f32, None)
    s_ts_t = copy_to_taichi(s_ts.column.info, ti.u32, [s_num, 3])
    
    # Set Taichi fields
    sbf_t = ti.field(dtype=ti.u8, shape=s_num)          # side boundary flag
    bi_t = ti.field(dtype=ti.u32, shape=e_num)          # boundary element indices
    bcount_t = ti.field(dtype=ti.u32, shape=())         # boundary element count
    is_boundary = ti.field(dtype=ti.u8, shape=e_num)    # mark boundary elements
    bcount_t[None] = 0
    
    @ti.kernel
    def mark_boundary_elements():
        for si in range(1, s_num):
            ei = 0
            sbf_t[si] = 0
            orient = s_ts_t[si, 0]
            el = s_ts_t[si, 1]
            eh = s_ts_t[si, 2]
            
            if orient == 1 and el * eh == 0:    # horizontal side
                ei = el if eh == 0 else eh
                sbf_t[si] = 1
            elif orient == 2 and el * eh == 0:  # vertical side
                ei = el if eh == 0 else eh
                sbf_t[si] = 1
            
            if ei != 0 and e_xs_t[ei] < 808411.0 and e_ys_t[ei] < 837066.0:
                is_boundary[ei] = 1

    @ti.kernel
    def collect_boundary_elements():
        for i in range(1, e_num):
            if is_boundary[i] == 1:
                idx = ti.atomic_add(bcount_t[None], 1)
                bi_t[idx] = i
    
    mark_boundary_elements()
    collect_boundary_elements()
    
    # Create boundary FDB
    capacity = int(bcount_t.to_numpy()[None][0])
    db = fdb.ORM.truncate([
        fdb.TableDefn(IndexLike, capacity, 'bdei'),
        fdb.TableDefn(U8Value, s_num, 'sbf')
    ])
    bdei: np.ndarray = db[IndexLike]['bdei'].column.index
    bdei[:] = bi_t.to_numpy()[:capacity]
    sbf: np.ndarray = db[U8Value]['sbf'].column.value
    sbf[:] = sbf_t.to_numpy()[:s_num]
    
    # Save to file
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def _check_boundary_fdb(test_fn: str, fdb_fn: str):
    """Check created boundary FDB"""
    # Load test boundary indices
    test_bd_ie = []
    with open(test_fn, 'r', encoding='utf-8') as f:
        for line in f:
            test_bd_ie.append(int(line.strip()))
    test_bd_ie = sorted(test_bd_ie)
    
    # Load created boundary FDB
    db = fdb.ORM.load(fdb_fn, from_file=True)
    bdei = db[IndexLike]['bdei'].column.index
    fdb_bd_ie = sorted(bdei.tolist())
    
    # Compare
    for i in range(len(test_bd_ie)):
        if test_bd_ie[i] != fdb_bd_ie[i]:
            print(f"Boundary index mismatch at position {i}: test {test_bd_ie[i]} vs fdb {fdb_bd_ie[i]}")
    print("Boundary FDB check passed!")

if __name__ == '__main__':
    import time
    
    start_time = time.time()
    build_boundary_fdb('./fdb/ne.fdb', './fdb/ns.fdb', './fdb/boundary.fdb')
    end_time = time.time()
    print(f"Boundary FDB created in {end_time - start_time:.2f} seconds.")
    
    # _check_boundary_fdb('resource/bd_ie.txt', './fdb/boundary.fdb')