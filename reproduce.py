import numpy as np
from pathlib import Path
from python import fastdb4py as fdb

class Ne(fdb.Feature):
    index: fdb.U32
    x: fdb.F32
    y: fdb.F32
    z: fdb.F32
    l_side_num: fdb.U32
    r_side_num: fdb.U32
    b_side_num: fdb.U32
    t_side_num: fdb.U32
    type: fdb.U8
    
class IndexLike(fdb.Feature):
    index: fdb.U32

def create_ne_fdb(ne_fn: str, fdb_fn: str):
    """Create NE FDB from NE file"""
    ne_path = Path(ne_fn)
    if not ne_path.exists():
        raise FileNotFoundError(f'NE file not found: {ne_path}')
    
    # Get element count
    ne_f = open(ne_path, 'r', encoding='utf-8')
    element_count = sum(1 for _ in ne_f) + 1  # including virtual element 0
    ne_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(IndexLike, element_count * 10, 'isl1'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl2'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl3'),
        fdb.TableDefn(IndexLike, element_count * 10, 'isl4'),
        fdb.TableDefn(Ne, element_count)
    ])
    
    nes = db[Ne][Ne]
    e_xs = nes.column.x
    e_ys = nes.column.y
    e_zs = nes.column.z
    e_types = nes.column.type
    e_indices = nes.column.index
    e_lcount = nes.column.l_side_num
    e_rcount = nes.column.r_side_num
    e_bcount = nes.column.b_side_num
    e_tcount = nes.column.t_side_num
    
    isl1 = db[IndexLike]['isl1'].column.index
    isl2 = db[IndexLike]['isl2'].column.index
    isl3 = db[IndexLike]['isl3'].column.index
    isl4 = db[IndexLike]['isl4'].column.index
    
    lines = open(ne_path, 'r', encoding='utf-8').readlines()
    
    for idx in range(1, element_count):
        ne_record = lines[idx - 1]
        data = ne_record.split(',')
        indices_array = np.array([int(v) for v in data[5:-4]], dtype=np.uint32)
        
        # Set hydro element data directly to np arrays
        e_indices[idx] = int(data[0])
        e_xs[idx] = float(data[-4])
        e_ys[idx] = float(data[-3])
        e_zs[idx] = float(data[-2])
        e_types[idx] = int(data[-1])
        e_lcount[idx] = int(data[1])
        e_rcount[idx] = int(data[2])
        e_bcount[idx] = int(data[3])
        e_tcount[idx] = int(data[4])
        
        # Set side indices
        si_offset = idx * 10
        l_count = e_lcount[idx]
        r_count = e_rcount[idx]
        b_count = e_bcount[idx]
        t_count = e_tcount[idx]
        isl1[si_offset:si_offset + l_count] = indices_array[0:l_count]
        isl2[si_offset:si_offset + r_count] = indices_array[l_count:l_count + r_count]
        isl3[si_offset:si_offset + b_count] = indices_array[l_count + r_count:l_count + r_count + b_count]
        isl4[si_offset:si_offset + t_count] = indices_array[l_count + r_count + b_count:l_count + r_count + b_count + t_count]
    
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))

if __name__ == '__main__':
    ne_fdb_path = Path('./ne.fdb')
    if ne_fdb_path.exists():
        ne_fdb_path.unlink()
    
    create_ne_fdb('./ne.txt', str(ne_fdb_path))
    es = fdb.ORM.load(str(ne_fdb_path), from_file=True)[Ne][Ne]
    print(f"Element count: {len(es)}") 
