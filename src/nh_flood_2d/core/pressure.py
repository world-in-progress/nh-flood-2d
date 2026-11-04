"""Tangential water-level reconstruction on hanging Cartesian subedges."""
from dataclasses import dataclass

import numpy as np
import taichi as ti

from ..util.ti import copy_to_taichi


@dataclass
class HangingGeometry:
    edge_index: np.ndarray
    offsets: np.ndarray
    gradient_slots: np.ndarray
    cells: np.ndarray
    axes: np.ndarray


def hanging_geometry(x, y, side_x, side_y, topology):
    """Build sparse reconstruction indices once from the loaded mesh geometry.

    Axis 0 is x and axis 1 is y. Exterior sides never reconstruct. A mesh with
    aligned centres returns None and needs no device fields or per-step work.
    """
    lower, higher = topology[:, 1], topology[:, 2]
    internal = (lower != 0) & (higher != 0)
    vertical = topology[:, 0] == 2
    tangent = np.where(vertical, side_y, side_x)
    centres = np.where(vertical[:, None], y[topology[:, 1:]], x[topology[:, 1:]])
    offsets = tangent[:, None] - centres
    edges = np.flatnonzero(internal & np.any(offsets != 0., axis=1))
    if not len(edges):
        return None
    offsets = offsets[edges].astype(np.float32)
    neighbours = topology[edges, 1:]
    displaced = offsets != 0.
    cells = np.unique(neighbours[displaced]).astype(np.int32)
    slots = np.full(neighbours.shape, -1, dtype=np.int32)
    slots[displaced] = np.searchsorted(cells, neighbours[displaced])
    axes = np.zeros((len(cells), 2), dtype=np.int32)
    edge_axes = np.broadcast_to(vertical[edges, None], neighbours.shape)
    axes[slots[displaced], edge_axes[displaced].astype(np.int32)] = 1
    edge_index = np.full(len(topology), -1, dtype=np.int32)
    edge_index[edges] = np.arange(len(edges), dtype=np.int32)
    return HangingGeometry(edge_index, offsets, slots, cells, axes)


@ti.data_oriented
class PressureReconstruction:
    """Cache each required cell gradient once before the momentum update."""

    def __init__(self, geometry, *, bed, depth, x, y, lengths, topology,
                 face_ptr, face_count, sides, min_h):
        self.edge_index = copy_to_taichi(geometry.edge_index, ti.i32, None)
        self.offsets = copy_to_taichi(geometry.offsets, ti.f32, None)
        self.slots = copy_to_taichi(geometry.gradient_slots, ti.i32, None)
        self.cells = copy_to_taichi(geometry.cells, ti.i32, None)
        self.axes = copy_to_taichi(geometry.axes, ti.i32, None)
        self.gradients = ti.field(ti.f32, (len(geometry.cells), 2))
        self.bed, self.depth, self.x, self.y = bed, depth, x, y
        self.lengths, self.topology = lengths, topology
        self.face_ptr, self.face_count, self.sides = face_ptr, face_count, sides
        self.min_h = min_h

    @ti.func
    def face_difference(self, cell, direction, axis):
        value, distance, total = 0., 0., 0.
        usable = 1
        eta = self.bed[cell] + self.depth[cell]
        start = self.face_ptr[cell, direction]
        for j in range(self.face_count[cell, direction]):
            side = self.sides[start + j]
            other = self.topology[side, 1] + self.topology[side, 2] - cell
            if other == 0 or self.depth[other] <= self.min_h:
                usable = 0
            else:
                weight = self.lengths[side]
                # Difference first: a common F32 water level stays exactly zero.
                value += weight * ((self.bed[other] + self.depth[other]) - eta)
                delta = self.x[other] - self.x[cell]
                if axis == 1:
                    delta = self.y[other] - self.y[cell]
                distance += weight * delta
                total += weight
        if total > 0.:
            value /= total
            distance /= total
        else:
            usable = 0
        return value, distance, usable

    @ti.func
    def gradient(self, cell, axis):
        minus, plus = 0, 1
        if axis == 1:
            minus, plus = 2, 3
        dm, xm, okm = self.face_difference(cell, minus, axis)
        dp, xp, okp = self.face_difference(cell, plus, axis)
        result = 0.
        if okm == 1 and okp == 1:
            result = (dp - dm) / (xp - xm)
        elif okm == 1:
            result = dm / xm
        elif okp == 1:
            result = dp / xp
        return result

    @ti.func
    def update_cell(self, slot):
        cell = self.cells[slot]
        for axis in ti.static(range(2)):
            if self.axes[slot, axis] == 1:
                value = 0.
                if self.depth[cell] > self.min_h:
                    value = self.gradient(cell, axis)
                self.gradients[slot, axis] = value

    @ti.func
    def difference(self, side, lower, higher, raw):
        result = raw
        index = self.edge_index[side]
        if index >= 0:
            if self.depth[lower] > self.min_h and self.depth[higher] > self.min_h:
                axis = ti.cast(self.topology[side, 0] == 2, ti.i32)
                if self.offsets[index, 0] != 0.:
                    result -= self.gradients[self.slots[index, 0], axis] * self.offsets[index, 0]
                if self.offsets[index, 1] != 0.:
                    result += self.gradients[self.slots[index, 1], axis] * self.offsets[index, 1]
        return result
