import time
import logging
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from typing import no_type_check
from dataclasses import dataclass

from ..util.ti import init_taichi, copy_to_taichi
from ..schema.feature import Ne, IndexLike, Ns, SideTopoInfo, Node

logger = logging.getLogger(__name__)

@dataclass
class ElementBounds:
    min_x: np.ndarray
    max_x: np.ndarray
    min_y: np.ndarray
    max_y: np.ndarray

@dataclass
class NodeInfo:
    index: int
    name: str
    x: float
    y: float
    is_outfall: bool = False

@dataclass
class CouplingConfig:
    max_nodes: int = 15000
    low_relation_distance: float = 600.0    # distance threshold (m) of low relation
    high_relation_elevation: float = 3.0    # elevation threshold (m) of high relation
    medium_relation_elevation: float = 2.6  # elevation threshold (m) of medium relation

def compute_grid_bounds(
    e_num: int,
    e_xs: np.ndarray, e_ys: np.ndarray,
    e_sns1: np.ndarray, e_sns2: np.ndarray,
    e_sns3: np.ndarray, e_sns4: np.ndarray,
    isl1: np.ndarray, isl2: np.ndarray, isl3: np.ndarray, isl4: np.ndarray,
    s_lens: np.ndarray, verbose: bool = False
) -> ElementBounds:
    bounds = ElementBounds(
        min_x = np.zeros(e_num, dtype=np.float32),
        max_x = np.zeros(e_num, dtype=np.float32),
        min_y = np.zeros(e_num, dtype=np.float32),
        max_y = np.zeros(e_num, dtype=np.float32)
    )
    
    # Reshape to 2D (Shape: [e_num, 10])
    isl1 = isl1.reshape([e_num, 10])
    isl2 = isl2.reshape([e_num, 10])
    isl3 = isl3.reshape([e_num, 10])
    isl4 = isl4.reshape([e_num, 10])
    
    for ei in range(1, e_num):  # ignore virtual hydro elemnt `0`
        len1 = sum(s_lens[isl1[ei, i]] for i in range(e_sns1[ei]))
        len2 = sum(s_lens[isl2[ei, i]] for i in range(e_sns2[ei]))
        len3 = sum(s_lens[isl3[ei, i]] for i in range(e_sns3[ei]))
        len4 = sum(s_lens[isl4[ei, i]] for i in range(e_sns4[ei]))
        
        if verbose:
            if abs(len1 - len2) > 1e-6:
                logger.warning(f'Hydro element {ei} left/right side length mismatch: {len1} vs {len2}')
            if abs(len3 - len4) > 1e-6:
                logger.warning(f'Hydro element {ei} bottom/top side length mismatch: {len3} vs {len4}')
        
        bounds.min_x[ei] = e_xs[ei] - len1 / 2.0
        bounds.max_x[ei] = e_xs[ei] + len1 / 2.0
        bounds.min_y[ei] = e_ys[ei] - len3 / 2.0
        bounds.max_y[ei] = e_ys[ei] + len3 / 2.0
    
    return bounds

def find_high_relations(
    nodes: fdb.Table,
    bounds: ElementBounds,
    e_zs: np.ndarray,
    ei_1235: np.ndarray,
    config: CouplingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_num = len(e_zs)
    n_num = len(nodes)
    
    n_topos = np.zeros((n_num, 2), dtype=np.int32)   # topo info for each node: [node_idx, high_relation_element_idx]
    e_relations = np.zeros(e_num, dtype=np.int32)    # 1: has high relation, 0: no high relation
    e_contains = np.zeros(e_num, dtype=np.int32)     # node nums contained in each element
    
    n_is = nodes.column.index
    n_xs = nodes.column.x
    n_ys = nodes.column.y
    for i in range(1, n_num):  # skip virtual node `0`
        # Get node info
        idx = n_is[i]
        x = n_xs[i]
        y = n_ys[i]
        
        # Register node topo info
        n_topos[idx, 0] = idx
        
        # Handle outfall nodes (only consider spatial relation)
        if nodes[i].is_outfall:
            for ei in range(1, e_num): # skip virtual hydro element `0`
                if (
                    x >= bounds.min_x[ei] and x <= bounds.max_x[ei] and
                    y >= bounds.min_y[ei] and y <= bounds.max_y[ei]
                ):
                    n_topos[idx, 1] = ei
                    e_relations[ei] = 1
                    break
        # Handle common nodes (consider spatial relation, elevation and element type)
        else:
            for ei in ei_1235:
                if (
                    x >= bounds.min_x[ei] and x <= bounds.max_x[ei] and
                    y >= bounds.min_y[ei] and y <= bounds.max_y[ei] and
                    e_zs[ei] >= config.high_relation_elevation
                ):
                    n_topos[idx, 1] = ei
                    e_relations[ei] = 1
                    e_contains[ei] += 1
                    break
    return n_topos, e_relations, e_contains

@ti.kernel
@no_type_check
def find_low_relations(
    n_common_ids: ti.template(),
    e_xs_t: ti.template(),
    e_ys_t: ti.template(),
    n_xs_t: ti.template(),
    n_ys_t: ti.template(),
    ei_12345_t: ti.template(),
    en_lrs_t: ti.template(),
    lr_num_t: ti.template(),
    e_relations_t: ti.template(),
    low_relation_distance: float
):
    for i in range(ei_12345_t.shape[0]):
        n_lr_idx = 0
        ei = ei_12345_t[i]
        min_dis = 100000000.0
        if e_relations_t[ei] == 0:
            for k in range(n_common_ids.shape[0]):
                ni = n_common_ids[k]
                this_dis = ti.sqrt((e_xs_t[ei]-n_xs_t[ni])**2 + (e_ys_t[ei]-n_ys_t[ni])**2)
                if this_dis < min_dis:
                    min_dis = this_dis
                    n_lr_idx = ni
            if min_dis <= low_relation_distance:
                lr_num = ti.atomic_add(lr_num_t[None], 1)
                en_lrs_t[lr_num, 0] = ei
                en_lrs_t[lr_num, 1] = n_lr_idx
            
def re_coo(
    ne_fn: str,
    ns_fn: str,
    node_fn: str,
    resource_dir: str,
    config: CouplingConfig | None = None
):
    # Initialize coupling configuration
    if config is None:
        config = CouplingConfig()
    
    # Initialize Taichi
    init_taichi(use_gpu=True, profiler=True)
    
    # Validate provided paths
    node_path = Path(node_fn)
    if not node_path.exists():
        raise FileNotFoundError(f'Pipeline node file not found: {node_path}')
    ne_path = Path(ne_fn)
    if not ne_path.exists():
        raise FileNotFoundError(f'NE file not found: {ne_path}')
    ns_path = Path(ns_fn)
    if not ns_path.exists():
        raise FileNotFoundError(f'NS file not found: {ns_path}')
    
    resource_path = Path(resource_dir)
    resource_path.mkdir(parents=True, exist_ok=True)
    
    # Load ne, ns and node data from fdb
    ne = fdb.ORM.load(str(ne_path), from_file=True)
    es = ne[Ne][Ne]
    e_num = len(es)
    e_xs = es.column.x
    e_ys = es.column.y
    e_zs = es.column.z
    e_ts = es.column.type
    e_sns1 = es.column.l_side_num   # sns: side numbers, 1: left, 2: right, 3: bottom, 4: top
    e_sns2 = es.column.r_side_num
    e_sns3 = es.column.b_side_num
    e_sns4 = es.column.t_side_num
    e_isl1 = ne[IndexLike]['isl1'].column.index   # isl: index side list
    e_isl2 = ne[IndexLike]['isl2'].column.index
    e_isl3 = ne[IndexLike]['isl3'].column.index
    e_isl4 = ne[IndexLike]['isl4'].column.index
    e_zs[e_ts == 1] += 10.0 # increase elevation by 10m for hydro elements with type 1 (building)
    
    ns = fdb.ORM.load(str(ns_path), from_file=True)
    ss = ns[Ns][Ns]
    s_lens = ss.column.length
    s_ts = ns[SideTopoInfo][SideTopoInfo].column.info    # sts: side topo infos
    
    nodes = fdb.ORM.load(str(node_path), from_file=True)[Node][Node]
    n_num = len(nodes)
    
    # Compute element bounds
    bounds = compute_grid_bounds(
        e_num, e_xs, e_ys,
        e_sns1, e_sns2, e_sns3, e_sns4,
        e_isl1, e_isl2, e_isl3, e_isl4, s_lens, verbose=True
    )
    
    # Remove hydro elements with type 4 (fishpond), 6 (water) or 7 (catch basin)
    ei_1235 = np.where(~np.isin(e_ts, [4, 6, 7]))[0]
    ei_12345 = np.where(~np.isin(e_ts, [6, 7]))[0]
    # Remove virtual hydro element `0`
    ei_1235 = ei_1235[ei_1235 != 0]     
    ei_12345 = ei_12345[ei_12345 != 0]
    
    # Find high relation between hydro elements and nodes
    n_topos, e_relations, e_contains = find_high_relations(
        nodes, bounds,
        e_zs, ei_1235,
        config
    )
    
    # Copy data to taichi
    e_xs_t = copy_to_taichi(e_xs, ti.f32, None)
    e_ys_t = copy_to_taichi(e_ys, ti.f32, None)
    n_xs_t = copy_to_taichi(nodes.column.x, ti.f32, None)
    n_ys_t = copy_to_taichi(nodes.column.y, ti.f32, None)
    ei_12345_t = copy_to_taichi(ei_12345, ti.i32, None)
    e_relations_t = copy_to_taichi(e_relations, ti.i32, None)
    
    # Get common node ids (non-outfall nodes)
    n_common_ids = np.array([node.index for node in nodes if not node.is_outfall], dtype=np.int32)
    n_common_ids = n_common_ids[n_common_ids != 0]  # remove virtual node 0
    n_common_ids_t = copy_to_taichi(n_common_ids, ti.i32, None)
    
    # Find low relation between hydro elements and nodes
    en_lrs_t = ti.field(dtype=ti.i32, shape=(e_num, 2)) # [hydro_element_idx, node_idx]
    lr_num_t = ti.field(dtype=ti.i32, shape=()) # scalar field to store number of low relations found
    lr_num_t[None] = 0
    find_low_relations(
        n_common_ids_t,
        e_xs_t, e_ys_t,
        n_xs_t, n_ys_t,
        ei_12345_t, en_lrs_t, lr_num_t,
        e_relations_t, config.low_relation_distance
    )
    en_lrs = en_lrs_t.to_numpy()[:lr_num_t[None]]
    
    # Build low relation topo list
    topo_list_l = []
    for ni in range(1, n_num):
        lr_list = []
        sub_list = [
            int(n_topos[ni, 0]),
            int(n_topos[ni, 1])
        ]
        for k in range(en_lrs.shape[0]):
            if en_lrs[k, 1] == ni:
                lr_list.append(int(en_lrs[k, 0]))
        sub_list.append(len(lr_list))
        sub_list.extend(lr_list)
        topo_list_l.append(sub_list)
    
    # Find medium relation between hydro elements and nodes
    topo_list_m = []
    for ni in range(len(topo_list_l)):
        mr_list = []
        sub_list = [
            int(n_topos[ni, 0]),
            int(n_topos[ni, 1])
        ]
        lr_num = topo_list_l[ni][2]
        if lr_num != 0:
            for k in range(3, 3 + lr_num):
                ei_lr = topo_list_l[ni][k]
                
                # Skip element represting fishpond
                if e_ts[ei_lr] == 4:
                    continue
                # Skip element has low elevation
                if e_zs[ei_lr] < config.medium_relation_elevation:
                    continue
                
                # Check if all neighboring elements of ei_lr has higer elevation
                is_medium = True
                # Check all 4 directions: left(1), right(2), bottom(3), top(4)
                # Iterate over all connected sides and identify the neighbor element dynamically
                for sns, isl in [
                    (e_sns1, e_isl1), (e_sns2, e_isl2), 
                    (e_sns3, e_isl3), (e_sns4, e_isl4)
                ]:
                    for count in range(sns[ei_lr]):
                        si = isl[ei_lr * 10 + count]
                        # s_ts structure: [type, e1, e2]
                        e1 = s_ts[si * 3 + 1]
                        e2 = s_ts[si * 3 + 2]
                        
                        # The neighbor is the element that is not the current one (ei_lr)
                        neighbor_ei = e1 if e2 == ei_lr else e2
                        
                        if e_zs[neighbor_ei] < e_zs[ei_lr]:
                            is_medium = False
                            break
                    if not is_medium:
                        break
                
                if is_medium:
                    mr_list.append(ei_lr)
        sub_list.append(len(mr_list))
        sub_list.extend(mr_list)
        topo_list_m.append(sub_list)

    # Persist results
    # ne.save('fdb/ne.fdb')  # save modified ne fdb with updated elevations
                
    with open(resource_path / 'node_num_per_grid.txt', 'w', encoding='utf-8') as f:
        for num in e_contains:
            f.write(f'{num}\n')
    
    with open(resource_path / 'high_relation_xy.txt', 'w', encoding='utf-8') as f:
        for ni in range(1, n_num):
            ei = int(n_topos[ni, 1])
            if ei != 0:
                f.write(f'{e_xs[ei]},{e_ys[ei]},{ni}\n')
    
    with open(resource_path / 'low_relation_xy.txt', 'w', encoding='utf-8') as f:
        for ei, ni in en_lrs:
            f.write(f'{e_xs[ei]},{e_ys[ei]},{ni}\n')
    
    with open(resource_path / 'medium_relation_xy.txt', 'w', encoding='utf-8') as f:
        for sub_list in topo_list_m:
            mr_num = sub_list[2]
            for k in range(3, 3 + mr_num):
                ei_mr = sub_list[k]
                f.write(f'{e_xs[ei_mr]},{e_ys[ei_mr]},{sub_list[0]}\n')
    
    with open(resource_path / 'topo_H&L.txt', 'w', encoding='utf-8') as f:
        for sub_list in topo_list_l:
            f.write(','.join(map(str, sub_list)) + '\n')
    
    with open(resource_path / 'topo_H&M.txt', 'w', encoding='utf-8') as f:
        for sub_list in topo_list_m:
            f.write(','.join(map(str, sub_list)) + '\n')
    
    with open(resource_path / 'single-node.txt', 'w', encoding='utf-8') as f:
        for sub_list in topo_list_m:
            ni = sub_list[0]
            if sub_list[2] == 0:
                f.write(f'{nodes[ni].name}')
      
if __name__ == '__main__':
    if True:
        start_time = time.time()
        re_coo(
            ne_fn = 'fdb/ne.fdb',
            ns_fn = 'fdb/ns.fdb',
            node_fn = 'fdb/node.fdb',
            resource_dir = 'resource',
        )
        print(f'Preprocessing time: {time.time() - start_time:.2f} seconds')
    
    else:
        from old_model.main import get_ne, get_ns
        from old_model.re_coo_new import re_coo as r1
        
        start_time = time.time()
        ne_data = get_ne()
        ns_data = get_ns()
        r1(
            'test_data/0610.inp',
            'resource',
            ne_data,
            ns_data,
            re_coo_index = 1
        )
        print(f'Re-COO (from main) time: {time.time() - start_time:.2f} seconds')
    