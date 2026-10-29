"""Map validated land-class parameters to mesh cells and sides."""
import numpy as np
from networkx.utils import UnionFind
from ..input.domain import HortonInfiltration, LandParameters


def connected_still_water(bed, side_crest, closed_sides, side_topology, boundaries, area):
    """Fill only water bodies reachable from their prescribed starting levels.

    Bed, crest and closed-side masks are the same working state used by the
    momentum update, including any gate that opens at the warmup start.
    """
    bed = bed.astype(np.float64)
    side_crest = side_crest.astype(np.float64)
    levels = {}
    for boundary in boundaries:
        if boundary.kind == 'level':
            levels.setdefault(boundary.series.at(0.), []).append(boundary)
    depth = np.zeros(len(bed), dtype=np.float32)
    left, right = side_topology[:, 1], side_topology[:, 2]
    crest = np.maximum(np.maximum(bed[left], bed[right]), side_crest)
    internal = (left != 0) & (right != 0) & ~closed_sides.astype(bool)
    bodies = []
    for level, group in levels.items():
        seeds = np.unique(np.concatenate([b.cells for b in group]))
        seeds = seeds[bed[seeds] < level]
        if not len(seeds):
            continue
        edges = internal & (crest < level)
        components = UnionFind(range(len(bed)))
        for a, b in zip(left[edges], right[edges]):
            components.union(int(a), int(b))
        labels = np.fromiter((components[i] for i in range(len(bed))), dtype=np.intp)
        selected = np.unique(labels[seeds])
        wet = np.isin(labels, selected)
        for other_level, other_group in levels.items():
            if other_level != level:
                for boundary in other_group:
                    if np.any(wet[boundary.cells]):
                        names = ', '.join(b.name for b in group)
                        raise ValueError('Conflicting initial levels on connected water body: '
                                         f'{names} ({level}) and {boundary.name} ({other_level})')
        depth[wet] = level - bed[wet].astype(np.float64)
        bodies.append(dict(level_m=float(level), boundaries=[b.name for b in group],
                           connected_components=len(selected), wet_cells=int(np.count_nonzero(wet))))
    return depth, dict(wet_cells=int(np.count_nonzero(depth)),
                       storage_m3=float(np.dot(depth.astype(np.float64), area)), levels=bodies)


def open_boundary_face_fractions(selected_sides, side_lengths, side_topology,
                                 face_ptrs, face_counts, flat_side_ids):
    """Selected exterior-side length divided by total length of each cell face."""
    result = np.zeros((len(face_ptrs), 4), dtype=np.float32)
    cells = np.unique(side_topology[selected_sides, 1:].sum(axis=1))
    for cell in cells:
        for direction in range(4):
            begin = face_ptrs[cell, direction]
            ids = flat_side_ids[begin:begin + face_counts[cell, direction]]
            lengths = side_lengths[ids].astype(np.float64)
            result[cell, direction] = lengths[selected_sides[ids]].sum() / lengths.sum()
    return result

def map_land_parameters(
    table: dict[int, LandParameters], land_types: np.ndarray, side_topology: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return side Manning n and cell Horton f0/fc (m/s), k (1/h)."""
    classes = np.unique(land_types[1:])
    missing = sorted(set(classes.tolist()) - table.keys())
    if missing:
        raise ValueError(f'land_parameters missing land types: {missing}')

    cell_n = np.zeros(len(land_types), dtype=np.float32)
    cell_f0 = np.zeros_like(cell_n)
    cell_fc = np.zeros_like(cell_n)
    cell_k = np.zeros_like(cell_n)
    mm_h_to_m_s = np.float32(1. / 3_600_000.)
    for land_type in classes:
        parameters = table[int(land_type)]
        mask = land_types == land_type
        cell_n[mask] = parameters.manning_n
        if isinstance(parameters.infiltration, HortonInfiltration):
            cell_f0[mask] = parameters.infiltration.f0_mm_h * mm_h_to_m_s
            cell_fc[mask] = parameters.infiltration.fc_mm_h * mm_h_to_m_s
            cell_k[mask] = parameters.infiltration.k_per_h

    # Element 0 denotes the exterior even if the mesh uses land-class code 0.
    cell_n[0] = cell_f0[0] = cell_fc[0] = cell_k[0] = 0.
    left, right = side_topology[:, 1], side_topology[:, 2]
    sides = (cell_n[left] + cell_n[right]) * np.float32(.5)
    exterior = (left == 0) | (right == 0)
    # A closed exterior side uses the interior value (a mirrored ghost cell).
    sides[exterior] = cell_n[left[exterior]] + cell_n[right[exterior]]
    return sides, cell_f0, cell_fc, cell_k
