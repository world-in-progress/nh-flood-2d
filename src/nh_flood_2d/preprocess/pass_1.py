import os
import linecache
import numpy as np
import fastdb4py as fdb
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from functools import partial

from ..input import DomainConfig
from ..schema.feature import Tide, Rainfall, IndexLike, SideTopoInfo, Ne, Ns, Node, Gate

def create_node_fdb(inp_fn: str, fdb_fn: str):
    """Create node FDB from inp file"""
    inp_path = Path(inp_fn)
    if not inp_path.exists():
        raise FileNotFoundError(f'Inp file not found: {inp_path}')
    db = fdb.ORM.create()
    
    # Add virtual node 0
    virtual_node = Node()
    virtual_node.index = 0
    virtual_node.name = 'virtual_node_0'
    virtual_node.x = 0.0
    virtual_node.y = 0.0
    virtual_node.is_outfall = False
    db.push(virtual_node)
    
    node_index = 0
    in_coordinates_section = False
    with open(inp_path, 'r', encoding='utf-8') as f:
        for l in f:
            stripped = l.strip()
            
            # Handle the start and end of Coordinates section
            if '[coordinates]' in stripped.lower(): # start of coordinates section
                in_coordinates_section = True
                continue
            elif stripped.startswith('[') and in_coordinates_section: # end of coordinates section
                break
            
            # Focus on the content of Coordinates section
            # Skip other sections, the header of Coordinates section and empty lines
            if not in_coordinates_section or stripped.startswith(';;') or not stripped:
                continue
            
            # Parse coordinate line
            parts = stripped.split()
            if len(parts) < 3:
                continue # skip invalid lines
            node_index += 1
            node_name = parts[0]
            is_outfall = node_name.lower().startswith('outfall')
            
            node = Node()
            node.index = node_index
            node.name = node_name
            node.x = float(parts[1])
            node.y = float(parts[2])
            node.is_outfall = is_outfall
            
            db.push(node)
    
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))

def create_gate_fdb(g_fn: str, fdb_fn: str):
    """Create gate FDB from gate file"""
    g_path = Path(g_fn)
    if not g_path.exists():
        raise FileNotFoundError(f'Gate file not found: {g_path}')

    db = fdb.ORM.truncate([
        fdb.TableDefn(IndexLike, 1, 'gate_count'),
        fdb.TableDefn(Gate, 100 * 100)
    ])  # assuming max 100 gates
    gates = db[Gate][Gate].column.info
    
    count = 0
    with open(g_path, 'r', encoding='utf-8') as f:
        for idx, l in enumerate(f):
            count += 1
            gate_array = np.array([int(v) for v in l.strip().split(',')], dtype=np.uint32)
            gates[idx * 100:idx * 100 + gate_array.shape[0]] = gate_array
    db[IndexLike]['gate_count'][0].index = count
    
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))

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
    
    for idx in range(1, element_count):
        ne_record = linecache.getline(str(ne_path), idx)
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

def create_ne_fdb_compact(ne_fn: str, fdb_fn: str):
    """Create NE FDB using a compact (CSR-like) side-index layout.

    The original layout allocates 4 fixed arrays of size (e_num × 10) for side
    indices, wasting ~9x memory on typical Cartesian grids where most elements
    have only 1 side per direction.

    Compact layout:
      - ``isl_data``  : flat array of every side index, all elements concatenated
                        [l-sides of e1, r-sides of e1, b-sides of e1, t-sides of e1,
                         l-sides of e2, ...]
      - ``isl_ptr_l`` : isl_ptr_l[ei] = index in isl_data where element ei's l-sides begin
      - ``isl_ptr_r`` : isl_ptr_r[ei] = index in isl_data where element ei's r-sides begin
      - ``isl_ptr_b`` : isl_ptr_b[ei] = index in isl_data where element ei's b-sides begin
      - ``isl_ptr_t`` : isl_ptr_t[ei] = index in isl_data where element ei's t-sides begin

    Memory: (total_actual_sides + 4 × e_num) × 4 bytes vs 160 × e_num bytes (old).
    """
    ne_path = Path(ne_fn)
    if not ne_path.exists():
        raise FileNotFoundError(f'NE file not found: {ne_path}')

    with open(ne_path, 'r', encoding='utf-8') as f:
        ne_lines = f.readlines()

    element_count = len(ne_lines) + 1  # +1 for virtual element 0

    # Count total side indices for pre-allocation
    total_sides = 0
    for line in ne_lines:
        data = line.split(',')
        total_sides += int(data[1]) + int(data[2]) + int(data[3]) + int(data[4])

    db = fdb.ORM.truncate([
        fdb.TableDefn(IndexLike, total_sides,   'isl_data'),
        fdb.TableDefn(IndexLike, element_count, 'isl_ptr_l'),
        fdb.TableDefn(IndexLike, element_count, 'isl_ptr_r'),
        fdb.TableDefn(IndexLike, element_count, 'isl_ptr_b'),
        fdb.TableDefn(IndexLike, element_count, 'isl_ptr_t'),
        fdb.TableDefn(Ne, element_count)
    ])

    nes       = db[Ne][Ne]
    e_xs      = nes.column.x
    e_ys      = nes.column.y
    e_zs      = nes.column.z
    e_types   = nes.column.type
    e_indices = nes.column.index
    e_lcount  = nes.column.l_side_num
    e_rcount  = nes.column.r_side_num
    e_bcount  = nes.column.b_side_num
    e_tcount  = nes.column.t_side_num

    isl_data  = db[IndexLike]['isl_data'].column.index
    isl_ptr_l = db[IndexLike]['isl_ptr_l'].column.index
    isl_ptr_r = db[IndexLike]['isl_ptr_r'].column.index
    isl_ptr_b = db[IndexLike]['isl_ptr_b'].column.index
    isl_ptr_t = db[IndexLike]['isl_ptr_t'].column.index

    data_offset = 0
    for idx, line in enumerate(ne_lines, start=1):
        data    = line.split(',')
        l_count = int(data[1])
        r_count = int(data[2])
        b_count = int(data[3])
        t_count = int(data[4])
        total   = l_count + r_count + b_count + t_count

        indices_array = np.array([int(v) for v in data[5:5 + total]], dtype=np.uint32)

        e_indices[idx] = int(data[0])
        e_xs[idx]      = float(data[-4])
        e_ys[idx]      = float(data[-3])
        e_zs[idx]      = float(data[-2])
        e_types[idx]   = int(data[-1])
        e_lcount[idx]  = l_count
        e_rcount[idx]  = r_count
        e_bcount[idx]  = b_count
        e_tcount[idx]  = t_count

        isl_data[data_offset:data_offset + total] = indices_array

        isl_ptr_l[idx] = data_offset
        isl_ptr_r[idx] = data_offset + l_count
        isl_ptr_b[idx] = data_offset + l_count + r_count
        isl_ptr_t[idx] = data_offset + l_count + r_count + b_count

        data_offset += total

    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))

def create_ne_fdb_parallel(ne_fn: str, fdb_fn: str):
    """Create NE FDB from NE file in parallel"""
    shared_name = 'shared_ne'
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
    
    db.share(shared_name, close_after=False)
    
    # Add actual hydro elements in parallel
    batch_size = 50000
    batch_args = [i for i in range(1, element_count, batch_size)]
    batch_func = partial(
        _batch_ne_worker,
        ne_count=element_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        ne_file=ne_fn
    )
    
    # num_procs = min(mp.cpu_count(), len(batch_args))
    num_procs = 1
    with mp.Pool(processes=num_procs) as pool:
        pool.map(batch_func, batch_args)
    
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def _filter_ne_ns(ne_fn: str, ns_fn: str, out_ne_fn: str, out_ns_fn: str):
    """
    Filter NE and NS files:
    1. Remove NE elements with z == -9999.
    2. Remove NS sides that are dangling (connected to removed NEs on both relevant ends).
    3. Renumber NE and NS indices.
    """
    print(f'Filtering NE/NS...\n  NE: {ne_fn}\n  NS: {ns_fn}')
    
    # --- Process NE ---
    ne_map = {}  # old_index -> new_index
    valid_ne_lines = []
    
    # First pass NE: read and determine valid elements
    with open(ne_fn, 'r', encoding='utf-8') as f:
        ne_lines = f.readlines()
        
    new_ne_idx = 0
    for line in ne_lines:
        parts = line.strip().split(',')
        if not parts: continue
        
        # Format: id, l, r, b, t, [indices...], x, y, z, type
        # z is at index -2
        z_val = float(parts[-2])
        old_idx = int(parts[0])
        
        if z_val != -9999:
            new_ne_idx += 1
            ne_map[old_idx] = new_ne_idx
            valid_ne_lines.append(line)
            
    print(f'  NE: Reduced from {len(ne_lines)} to {len(valid_ne_lines)} elements.')
            
    # --- Process NS to build ns_map and filter ---
    ns_map = {} # old_side_id -> new_side_id
    
    with open(ns_fn, 'r', encoding='utf-8') as f:
        ns_lines = f.readlines()
        
    new_ns_idx = 0
    temp_valid_ns_data = [] # Store parts for valid NS lines
    
    for line in ns_lines:
        parts = line.strip().split(',')
        if len(parts) < 6: continue
        
        # NS Format: id, orient, left, right, bottom, top, length, x, y, z, attr
        old_ns_id = int(parts[0])
        orient = int(parts[1])
        left = int(parts[2])
        right = int(parts[3])
        bottom = int(parts[4])
        top = int(parts[5])
        
        # Remap neighbors. If not in map, it means it was removed (mapped to 0).
        new_left = ne_map.get(left, 0)
        new_right = ne_map.get(right, 0)
        new_bottom = ne_map.get(bottom, 0)
        new_top = ne_map.get(top, 0)
        
        # Check validity
        # Horizontal (1): connects top/bottom. Vertical (2): connects left/right.
        keep = False
        if orient == 1: # horizontal
            if new_bottom != 0 or new_top != 0:
                keep = True
        else: # vertical
            if new_left != 0 or new_right != 0:
                keep = True
                
        if keep:
            new_ns_idx += 1
            ns_map[old_ns_id] = new_ns_idx
            
            # Update columns 0, 2, 3, 4, 5
            parts[0] = str(new_ns_idx)
            parts[2] = str(new_left)
            parts[3] = str(new_right)
            parts[4] = str(new_bottom)
            parts[5] = str(new_top)
            
            temp_valid_ns_data.append(",".join(parts))
            
    print(f'  NS: Reduced from {len(ns_lines)} to {len(temp_valid_ns_data)} sides.')
            
    # Write valid NS file
    Path(out_ns_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(out_ns_fn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(temp_valid_ns_data))
        if temp_valid_ns_data: f.write('\n') # trailing newline
        
    # --- Rewrite NE file with updated IDs and updated Side IDs ---
    updated_ne_lines = []
    
    for line in valid_ne_lines:
        parts = line.strip().split(',')
        
        # Update ID (Col 0)
        old_id = int(parts[0])
        parts[0] = str(ne_map[old_id])
        
        # Update Side Indices (Col 5 to ...)
        # Counts are at 1, 2, 3, 4
        # counts = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
        # total_sides = sum(counts)
        # side indices start at 5, go to 5 + total_sides
        
        l_cnt = int(parts[1])
        r_cnt = int(parts[2])
        b_cnt = int(parts[3])
        t_cnt = int(parts[4])
        total_sides = l_cnt + r_cnt + b_cnt + t_cnt
        
        side_idx_start = 5
        indices_end = side_idx_start + total_sides
        
        # Update indices in place
        for i in range(side_idx_start, indices_end):
            old_s_id = int(parts[i])
            parts[i] = str(ns_map.get(old_s_id, 0))
            
        updated_ne_lines.append(",".join(parts))
        
    with open(out_ne_fn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(updated_ne_lines))
        if updated_ne_lines: f.write('\n')

def create_ns_fdb_parallel(ns_fn: str, fdb_fn: str):
    """Create NS FDB from NS file in parallel"""
    shared_name = 'shared_ns'
    ns_path = Path(ns_fn)
    if not ns_path.exists():
        raise FileNotFoundError(f'NS file not found: {ns_path}')
    
    # Get side count
    ns_f = open(ns_path, 'r')
    side_count = sum(1 for _ in ns_f) + 1  # including virtual side 0
    ns_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(Ns, side_count),
        fdb.TableDefn(SideTopoInfo, side_count * 3)
    ])
    
    db.share(shared_name, close_after=False)
    
    # Add actual sides in parallel
    batch_size = 50000000
    batch_args = [i for i in range(1, side_count, batch_size)]
    batch_func = partial(
        _batch_ns_worker,
        ns_count=side_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        ns_file=ns_fn
    )
    
    num_procs = min(mp.cpu_count(), len(batch_args))
    num_procs = 1
    with mp.Pool(processes=num_procs) as pool:
        pool.map(batch_func, batch_args)
        
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def create_tide_fdb_parallel(t_fn: str, fdb_fn: str):
    """Create tide FDB from tide file in parallel"""
    shared_name = 'shared_tide'
    tide_path = Path(t_fn)
    if not tide_path.exists():
        raise FileNotFoundError(f'Tide file not found: {tide_path}')
    
    # Get tide record counts
    t_f = open(tide_path, 'r', encoding='utf-8')
    t_count = sum(1 for _ in t_f) - 1  # exclude header
    t_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(Tide, t_count)        
    ])
    db.share(shared_name, close_after=False)
    
    # Add tide records in parallel
    batch_size = 10000
    batch_args = [i for i in range(0, t_count, batch_size)]
    batch_func = partial(
        _batch_tide_worker,
        t_count=t_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        t_fn=str(tide_path)
    )
    
    # num_procs = min(mp.cpu_count(), len(batch_args))
    num_procs = 1
    with mp.Pool(processes=num_procs) as pool:
        pool.map(batch_func, batch_args)
    
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()

def create_rainfall_fdb_parallel(r_fn: str, fdb_fn: str):
    """Create rainfall FDB from rainfall file in parallel"""
    shared_name = 'shared_rainfall'
    rainfall_path = Path(r_fn)
    if not rainfall_path.exists():
        raise FileNotFoundError(f'Rainfall file not found: {rainfall_path}')
    
    # Get rainfall record counts
    r_f = open(rainfall_path, 'r', encoding='utf-8')
    r_count = sum(1 for _ in r_f) - 1  # exclude header
    r_f.close()
    
    db = fdb.ORM.truncate([
        fdb.TableDefn(Rainfall, r_count)        
    ])
    db.share(shared_name, close_after=False)
    
    # Add rainfall records in parallel
    batch_size = 10000
    batch_args = [i for i in range(0, r_count, batch_size)]
    batch_func = partial(
        _batch_rainfall_worker,
        r_count=r_count,
        fdb_fn=shared_name,
        batch_size=batch_size,
        r_fn=str(rainfall_path)
    )
    
    num_procs = min(mp.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_procs) as pool:
        pool.map(batch_func, batch_args)
    
    # Save to file and remove shared database
    fdb_path = Path(fdb_fn)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(fdb_path))
    db.unlink()
    
def _batch_ne_worker(ne_si: int, ne_count: int, fdb_fn: str, batch_size: int, ne_file: str):
    try:
        db = fdb.ORM.load(fdb_fn)
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
        
        for idx in range(ne_si, min(ne_si + batch_size, ne_count)):
            ne_record = linecache.getline(ne_file, idx)
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
    finally:
        db.close() 

def _batch_ns_worker(ns_si: int, ns_count: int, fdb_fn: str, batch_size: int, ns_file: str):
    try:
        db = fdb.ORM.load(fdb_fn)
        nss = db[Ns][Ns]
        s_indices = nss.column.index
        s_lengths = nss.column.length
        s_xs = nss.column.x
        s_ys = nss.column.y
        s_zs = nss.column.z
        s_attrs = nss.column.attr
        
        sts = db[SideTopoInfo][SideTopoInfo].column.info
        
        for idx in range(ns_si, min(ns_si + batch_size, ns_count)):
            ns_record = linecache.getline(ns_file, idx)
            data = ns_record.split(',')
            topo = np.array([int(v) for v in data[1:6]], dtype=np.uint32)
            sts_s = idx * 3
            if topo[0] == 1: # horizontal (connects bottom/top)
                sts[sts_s:sts_s + 3] = np.array([topo[0], topo[3], topo[4]], dtype=np.uint32)
            else: # vertical (connects left/right)
                sts[sts_s:sts_s + 3] = np.array([topo[0], topo[1], topo[2]], dtype=np.uint32)
            
            # Set side data directly to np arrays
            s_indices[idx] = int(data[0])
            s_lengths[idx] = float(data[6])
            s_xs[idx] = float(data[7])
            s_ys[idx] = float(data[8])
            s_zs[idx] = float(data[9])
            s_attrs[idx] = int(data[10])
    finally:
        db.close()

def _batch_tide_worker(t_si: int, t_count: int, fdb_fn: str, batch_size: int, t_fn: str):
    try:
        db = fdb.ORM.load(fdb_fn)
        ts = db[Tide][Tide]
        times = ts.column.time
        levels = ts.column.level
        
        for l_idx in range(t_si, min(t_si + batch_size, t_count)):
            line = linecache.getline(t_fn, l_idx + 2) # +2 to skip header and 0-based index
            data = line.strip().split(',')
            time_str = f'{data[0]} {data[1]}'
            date = datetime.strptime(time_str, '%d/%m/%Y %H:%M:%S').timestamp()
            
            times[l_idx] = date
            levels[l_idx] = float(data[2])
    finally:
        db.close()

def _batch_rainfall_worker(r_si: int, r_count: int, fdb_fn: str, batch_size: int, r_fn: str):
    try:
        db = fdb.ORM.load(fdb_fn)
        rs = db[Rainfall][Rainfall]
        times = rs.column.time
        quantities = rs.column.quantity
        
        for r_idx in range(r_si, min(r_si + batch_size, r_count)):
            l = linecache.getline(r_fn, r_idx + 2) # +2 to skip header and 0-based index
            data = l.strip().split(',')
            date = datetime.strptime(data[0], '%Y/%m/%d %H:%M').timestamp()
            times[r_idx] = date
            quantities[r_idx] = float(data[2])
    finally:
        db.close()

def _check_node_fdb(inp_fn: str, fdb_fn: str):
    inp_path = Path(inp_fn)
    fdb_path = Path(fdb_fn)
    db = fdb.ORM.load(str(fdb_path), from_file=True)
    nodes = db[Node][Node]
    
    node_index = 1
    in_coordinates_section = False
    with open(inp_path, 'r', encoding='utf-8') as f:
        for l in f:
            stripped = l.strip()
            
            # Handle the start and end of Coordinates section
            if '[coordinates]' in stripped.lower(): # start of coordinates section
                in_coordinates_section = True
                continue
            elif stripped.startswith('[') and in_coordinates_section: # end of coordinates section
                break
            
            # Focus on the content of Coordinates section
            # Skip other sections, the header of Coordinates section and empty lines
            if not in_coordinates_section or stripped.startswith(';;') or not stripped:
                continue
            
            # Parse coordinate line
            parts = stripped.split()
            if len(parts) < 3:
                continue # skip invalid lines
            
            node = nodes[node_index]
            
            node_index += 1
            node_name = parts[0]
            is_outfall = node_name.lower().startswith('outfall')
            assert node is not None, f'Node index {node_index} not found in FDB'
            assert node.name == node_name, f'Node name mismatch at index {node_index}: {node.name} != {node_name}'
            assert abs(node.x - float(parts[1])) < 0.1, f'Node x mismatch at index {node_index}: {node.x} != {parts[1]}'
            assert abs(node.y - float(parts[2])) < 0.1, f'Node y mismatch at index {node_index}: {node.y} != {parts[2]}'
            assert node.is_outfall == is_outfall, f'Node is_outfall mismatch at index {node_index}: {node.is_outfall} != {is_outfall}'
    
    print('FDB node data verification passed.')

def _check_gate_fdb(gate_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    gates = db[Gate][Gate].column.info
    
    with open(gate_fn, 'r', encoding='utf-8') as f:
        for idx, l in enumerate(f):
            gate_array = np.array([int(v) for v in l.strip().split(',')], dtype=np.uint32)
            fdb_gate_array = gates[idx * 100:idx * 100 + gate_array.shape[0]]
            if not np.array_equal(gate_array, fdb_gate_array):
                raise ValueError(f'Gate data mismatch at index {idx}')
    
    print('FDB gate data verification passed.')

def _check_ne_fdb(ne_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    nes = db[Ne][Ne]
    isl1 = db[IndexLike]['isl1'].column.index
    isl2 = db[IndexLike]['isl2'].column.index
    isl3 = db[IndexLike]['isl3'].column.index
    isl4 = db[IndexLike]['isl4'].column.index
    
    with open(ne_fn, 'r', encoding='utf-8') as f:
        for line in f:
            record = line.split(',')
            idx = int(record[0])
            left_edge_num = int(record[1])
            right_edge_num = int(record[2])
            bottom_edge_num = int(record[3])
            top_edge_num = int(record[4])
            start = 5
            left_edges = [int(edge_idx) for edge_idx in record[start:start + left_edge_num]]
            start += left_edge_num
            right_edges = [int(edge_idx) for edge_idx in record[start:start + right_edge_num]]
            start += right_edge_num
            bottom_edges = [int(edge_idx) for edge_idx in record[start:start + bottom_edge_num]]
            start += bottom_edge_num
            top_edges = [int(edge_idx) for edge_idx in record[start:start + top_edge_num]]

            ne_element = nes[idx]
            assert ne_element.index == idx, f'Index mismatch: {ne_element.index} != {idx}'
            assert ne_element.l_side_num == left_edge_num, f'Left edge count mismatch at index {idx}'
            assert ne_element.r_side_num == right_edge_num, f'Right edge count mismatch at index {idx}'
            assert ne_element.b_side_num == bottom_edge_num, f'Bottom edge count mismatch at index {idx}'
            assert ne_element.t_side_num == top_edge_num, f'Top edge count mismatch at index {idx}'

            si_offset = idx * 10
            l_edges_db = isl1[si_offset:si_offset + left_edge_num].tolist()
            r_edges_db = isl2[si_offset:si_offset + right_edge_num].tolist()
            b_edges_db = isl3[si_offset:si_offset + bottom_edge_num].tolist()
            t_edges_db = isl4[si_offset:si_offset + top_edge_num].tolist()

            assert l_edges_db == left_edges, f'Left edges mismatch at index {idx}'
            assert r_edges_db == right_edges, f'Right edges mismatch at index {idx}'
            assert b_edges_db == bottom_edges, f'Bottom edges mismatch at index {idx}'
            assert t_edges_db == top_edges, f'Top edges mismatch at index {idx}'

    print('FDB NE data verification passed.')

def _check_ns_fdb(ns_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    nss = db[Ns][Ns]
    
    with open(ns_fn, 'r', encoding='utf-8') as f:
        for line in f:
            record = line.split(',')
            idx = int(record[0])
            orient = int(record[1])
            left = int(record[2])
            right = int(record[3])
            bottom = int(record[4])
            top = int(record[5])
            length = float(record[6])
            x = float(record[7])
            y = float(record[8])
            z = float(record[9])
            attr = int(record[10])

            ns_element = nss[idx]
            ns_topo = db[SideTopoInfo][SideTopoInfo].column.info[idx * 3:(idx + 1) * 3]
            assert ns_element.index == idx, f'Index mismatch: {ns_element.index} != {idx}'
            if orient == 1:  # horizontal
                assert ns_topo[0] == orient, f'Orient mismatch at index {idx}'
                assert ns_topo[1] == bottom, f'Bottom mismatch at index {idx}'
                assert ns_topo[2] == top, f'Top mismatch at index {idx}'
            else:  # vertical
                assert ns_topo[0] == orient, f'Orient mismatch at index {idx}'
                assert ns_topo[1] == left, f'Left mismatch at index {idx}'
                assert ns_topo[2] == right, f'Right mismatch at index {idx}'
            assert abs(ns_element.length - length) < 1e-3, f'Length mismatch at index {idx}'
            assert abs(ns_element.x - x) < 0.1, f'X mismatch at index {idx}'
            assert abs(ns_element.y - y) < 0.1, f'Y mismatch at index {idx}'
            assert abs(ns_element.z - z) < 1e-3, f'Z mismatch at index {idx}'
            assert ns_element.attr == attr, f'Attr mismatch at index {idx}'

    print('FDB NS data verification passed.')

def _check_tide_fdb(tide_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    tides = db[Tide][Tide]
    times = tides.column.time
    levels = tides.column.level
    
    with open(tide_fn, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for idx, line in enumerate(f):
            data = line.strip().split(',')
            time = data[0] + ' ' + data[1]
            date = datetime.strptime(time, '%d/%m/%Y %H:%M:%S').timestamp()
            level = float(data[2])
            
            fdb_time = times[idx]
            fdb_level = levels[idx]
            
            if abs(fdb_time - date) > 1e-6 or abs(fdb_level - level) > 1e-6:
                raise ValueError(f'Mismatch at record {idx}: file({date}, {level}) != fdb({fdb_time}, {fdb_level})')
    
    print('FDB tide data verification passed.')
    
def _check_rainfall_fdb(r_fn: str, fdb_fn: str):
    db = fdb.ORM.load(fdb_fn, from_file=True)
    rainfalls = db[Rainfall][Rainfall]
    times = rainfalls.column.time
    quantities = rainfalls.column.quantity
    
    with open(r_fn, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for idx, line in enumerate(f):
            data = line.strip().split(',')
            time = data[0]
            date = datetime.strptime(time, '%Y/%m/%d %H:%M').timestamp()
            quantity = float(data[2])
            
            fdb_time = times[idx]
            fdb_quantity = quantities[idx]
            
            if abs(fdb_time - date) > 1e-6 or abs(fdb_quantity - quantity) > 1e-6:
                raise ValueError(f'Mismatch at record {idx}: file({date}, {quantity}) != fdb({fdb_time}, {fdb_quantity})')
    
    print('FDB rainfall data verification passed.')

def _create_worker(cfg: DomainConfig, idx: int):
    if idx == 0:
        create_gate_fdb(cfg.gate, cfg.gate_fdb)
    elif idx == 1:
        create_ne_fdb_compact(cfg.tmp_ne, cfg.ne_fdb)
    elif idx == 2:
        create_ns_fdb_parallel(cfg.tmp_ns, cfg.ns_fdb)
    elif idx == 3:
        create_tide_fdb_parallel(cfg.tide, cfg.tide_fdb)
    elif idx == 4:
        create_rainfall_fdb_parallel(cfg.rain, cfg.rain_fdb)

def build_fdbs(cfg: DomainConfig):
    
    # _filter_ne_ns(cfg.ne, cfg.ns, cfg.tmp_ne, cfg.tmp_ns)
    # create_ne_fdb_compact(cfg.tmp_ne, cfg.ne_fdb)
    create_ns_fdb_parallel(cfg.tmp_ns, cfg.ns_fdb)
    # try:
    #     # Generate filtered temporary NE/NS files
    #     _filter_ne_ns(cfg.ne, cfg.ns, cfg.tmp_ne, cfg.tmp_ns)
        
    #     # Create FDBs in parallel
    #     processes = []
    #     for i in range(5):
    #         p = mp.Process(target=_create_worker, args=(cfg, i))
    #         p.start()
    #         processes.append(p)
        
    #     for p in processes:
    #         p.join()
    # finally:
    #     # Cleanup temporary files
    #     if Path(cfg.tmp_ne).exists(): os.remove(cfg.tmp_ne)
    #     if Path(cfg.tmp_ns).exists(): os.remove(cfg.tmp_ns)

if __name__ == '__main__':
    import time
    start_time = time.time()
    
    # Generate filtered temporary NE/NS files
    _filter_ne_ns('test_data/ne.txt', 'test_data/ns.txt', 'temp_ne.txt', 'temp_ns.txt')
    
    # Create FDBs in parallel
    processes = []
    for i in range(5):
        p = mp.Process(target=_create_worker, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    # Cleanup temporary files
    if Path('temp_ne.txt').exists(): os.remove('temp_ne.txt')
    if Path('temp_ns.txt').exists(): os.remove('temp_ns.txt')

    end_time = time.time()
    print(f'FDB creation completed in {end_time - start_time:.2f} seconds.')
    
    # _check_node_fdb('test_data/0610.inp', 'fdb/node.fdb')
    # _check_gate_fdb('test_data/max_gate7_ne.txt', 'fdb/gate.fdb')
    # _check_ne_fdb('test_data/ne.txt', 'fdb/ne.fdb')
    # _check_ns_fdb('test_data/ns.txt', 'fdb/ns.fdb')
    # _check_tide_fdb('test_data/test_tide.csv', 'fdb/tide.fdb')
    # _check_rainfall_fdb('test_data/test_rain.csv', 'fdb/rain.fdb')