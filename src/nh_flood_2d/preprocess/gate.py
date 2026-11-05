"""Resolve gate components by shared subedges, including hanging interfaces."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GateLayout:
    cell_gate: np.ndarray
    controls: np.ndarray
    cells: list[np.ndarray]

    @classmethod
    def load(cls, config, mesh, land_types, topology, boundaries):
        selected = land_types == config.land_type
        selected[0] = False
        adjacency = [[] for _ in land_types]
        for _, left, right in topology[1:]:
            if selected[left] and selected[right]:
                adjacency[left].append(right)
                adjacency[right].append(left)
        component = np.full(len(land_types), -1, dtype=np.int32)
        groups = []
        for seed in np.flatnonzero(selected):
            if component[seed] >= 0:
                continue
            pending, cells = [int(seed)], []
            component[seed] = len(groups)
            while pending:
                cell = pending.pop()
                cells.append(cell)
                for neighbour in adjacency[cell]:
                    if component[neighbour] < 0:
                        component[neighbour] = len(groups)
                        pending.append(int(neighbour))
            groups.append(np.array(sorted(cells), dtype=np.int32))

        def locate(point, label):
            x, y = point
            b = mesh.bounds
            hits = np.flatnonzero((b[:, 0] <= x) & (x <= b[:, 2]) &
                                  (b[:, 1] <= y) & (y <= b[:, 3]))
            hits = hits[hits != 0]
            if not len(hits):
                raise ValueError(f'{label}: point lies outside the mesh')
            # Same deterministic closed-rectangle convention as station sampling.
            return int(hits[0])

        cell_gate = np.full(len(land_types), -1, dtype=np.int32)
        controls, cells, claimed = [], [], set()
        for index, gate in enumerate(config.gates):
            inside = locate(gate.inside_point, f'{gate.name}.inside_point')
            group = int(component[inside])
            if group < 0:
                raise ValueError(f'{gate.name}: inside_point is not on a gate cell')
            if group in claimed:
                raise ValueError('Each gate component must correspond to exactly one gate')
            claimed.add(group)
            control = locate(gate.control_point, f'{gate.name}.control_point')
            if selected[control]:
                raise ValueError(f'{gate.name}: control_point is on a gate cell')
            controls.append(control)
            cells.append(groups[group])
            cell_gate[groups[group]] = index
        if len(claimed) != len(groups):
            raise ValueError('Each gate component must correspond to exactly one gate')
        for boundary in boundaries:
            if boundary.kind == 'level' and np.any(selected[boundary.cells]):
                raise ValueError('Gate cells must not overlap a level boundary')
        return cls(cell_gate, np.array(controls, dtype=np.int32), cells)
