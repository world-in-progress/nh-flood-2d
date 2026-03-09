import os
import linecache
import numpy as np
import fastdb4py as fdb
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from functools import partial

from ..input import ForceConfig
from ..schema.feature import Tide, Rainfall, IndexLike, SideTopoInfo, Ne, Ns, Node, Gate

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

def _create_worker(cfg: ForceConfig, idx: int):
    if idx == 0:
        create_gate_fdb(cfg.gate, cfg.gate_fdb)
    elif idx == 1:
        create_tide_fdb_parallel(cfg.tide, cfg.tide_fdb)
    elif idx == 2:
        create_rainfall_fdb_parallel(cfg.rain, cfg.rain_fdb)

def prepare_force(cfg: ForceConfig):
    processes = []
    for i in range(3):
        p = mp.Process(target=_create_worker, args=(cfg, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()