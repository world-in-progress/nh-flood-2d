"""Physical invariants of the hanging-interface pressure reconstruction."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import taichi as ti
from rasterio.transform import from_origin

from src.nh_flood_2d.core.output import read_checkpoint, write_checkpoint
from src.nh_flood_2d.core.pressure import PressureReconstruction, hanging_geometry
from src.nh_flood_2d.core.run_record import RunRecord, sha256
from src.nh_flood_2d.core.solver_compact import _solve
from src.nh_flood_2d.input import DomainConfig, ForceConfig
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.util.ti import copy_to_taichi
from test_solver_forcing import START

pytestmark = pytest.mark.usefixtures('taichi_cpu')


def square_mesh(path, cells, bed_at):
    """Write complete square-cell topology, splitting coarse faces at neighbours."""
    breaks = [{}, {}]
    for x0, x1, y0, y1 in cells:
        for x in (x0, x1):
            breaks[0].setdefault(x, set()).update((y0, y1))
        for y in (y0, y1):
            breaks[1].setdefault(y, set()).update((x0, x1))
    owners, faces = {}, []
    for cell, (x0, x1, y0, y1) in enumerate(cells, 1):
        groups = []
        for orient, coordinate, lo, hi, owner in (
                (2, x0, y0, y1, 2), (2, x1, y0, y1, 1),
                (1, y0, x0, x1, 2), (1, y1, x0, x1, 1)):
            points = sorted(p for p in breaks[2-orient][coordinate] if lo <= p <= hi)
            group = []
            for a, b in zip(points, points[1:]):
                key = orient, coordinate, a, b
                owners.setdefault(key, {})[owner] = cell
                group.append(key)
            groups.append(group)
        faces.append(groups)
    side_ids = {key: i for i, key in enumerate(sorted(owners), 1)}
    xy = np.array([(0, 0)] + [((a+b)/2, (c+d)/2) for a, b, c, d in cells], dtype='float32')
    bed = np.r_[0, [bed_at(x, y, i) for i, (x, y) in enumerate(xy[1:], 1)]].astype('float32')
    ne = [','.join(map(str, [i, *map(len, ff), *[side_ids[k] for group in ff for k in group],
                           *xy[i], bed[i], 10])) for i, ff in enumerate(faces, 1)]
    topo = np.zeros((len(side_ids)+1, 3), dtype='int32')
    side_xy = np.zeros((len(topo), 2), dtype='float32')
    ns = []
    for (orient, coordinate, a, b), i in side_ids.items():
        lower, higher = owners[(orient, coordinate, a, b)].get(1, 0), owners[(orient, coordinate, a, b)].get(2, 0)
        x, y = (coordinate, (a+b)/2) if orient == 2 else ((a+b)/2, coordinate)
        neighbours = (lower, higher, 0, 0) if orient == 2 else (0, 0, lower, higher)
        ns.append(','.join(map(str, [i, orient, *neighbours, b-a, x, y, 0, 0])))
        topo[i], side_xy[i] = (orient, lower, higher), (x, y)
    (path/'ne.txt').write_text('\n'.join(ne)+'\n')
    (path/'ns.txt').write_text('\n'.join(ns)+'\n')
    return xy, bed, topo, side_xy


def strip_cells(rotated=False):
    cells = ([(0, 8, y, y+8) for y in range(0, 24, 8)]
             + [(8, 12, y, y+4) for y in range(0, 24, 4)])
    return [(c, d, a, b) for a, b, c, d in cells] if rotated else cells


def evolve(path, cells, bed_at, depth_at, seconds=.125):
    xy, bed, topo, side_xy = square_mesh(path, cells, bed_at)
    depth = np.r_[0, [depth_at(x, y, bed[i], i) for i, (x, y) in enumerate(xy[1:], 1)]].astype('float32')
    zeros = np.zeros_like(depth)
    words = np.zeros(len(depth), dtype='uint32')
    seed = path/'seed.fdb'
    write_checkpoint(seed, depth=depth, bed=bed, q=np.zeros(len(topo), dtype='float32'),
        rain_time=zeros, infiltration=zeros, max_depth=depth, max_speed=zeros,
        first_time_hi=words, first_time_lo=words, clock_hi=0, clock_lo=0, gate_open=False,
        metadata_json=json.dumps(dict(run_state=False, ne_sha256=sha256(path/'ne.txt'),
                                     ns_sha256=sha256(path/'ns.txt'), gate_config='null')))
    x1, y1 = max(c[1] for c in cells), max(c[3] for c in cells)
    with rasterio.open(path/'rain.tif', 'w', driver='GTiff', width=1, height=1, count=1,
                       dtype='float32', crs='EPSG:2326', transform=from_origin(0, y1, x1, y1)) as out:
        out.write(np.zeros((1, 1), dtype='float32'), 1)
    (path/'intervals.csv').write_text('interval_start,interval_end\n'
                                    '2023-09-07T08:00:00+00:00,2023-09-07T09:00:00+00:00\n')
    dc = DomainConfig(ne=str(path/'ne.txt'), ns=str(path/'ns.txt'), epsg_code=2326,
        domain_dir=str(path/'out'), start=START, duration=max(1, int(seconds)), warmup_duration=0,
        land_parameters={10: dict(manning_n=.03, infiltration='none')}, restart_state=str(seed))
    fc = ForceConfig(boundaries=[], rainfall=dict(raster=str(path/'rain.tif'), intervals=str(path/'intervals.csv')))
    with RunRecord(dc, fc) as record:
        record.hash_inputs(dc, fc)
        preprocess(dc, fc)
        _solve(dc, fc, record, end_time=seconds)
    return read_checkpoint(path/'out/output/state_final.fdb'), depth, xy, topo, side_xy


@pytest.mark.parametrize('rotated', [False, True])
def test_linear_tangential_surface_has_only_roundoff_crossflow(tmp_path, rotated):
    axis = 0 if rotated else 1
    result, _, xy, topo, side_xy = evolve(tmp_path, strip_cells(rotated),
        lambda x, y, i: .001*(x if rotated else y), lambda x, y, z, i: 1.)
    interface = (topo[:, 0] == (1 if rotated else 2)) & (topo[:, 1] != 0) & (topo[:, 2] != 0)
    assert np.max(np.abs(result.q[interface])) < 3e-7
    assert .001/3 * 9.81*.125 > 1000*np.max(np.abs(result.q[interface]))
    layout = hanging_geometry(xy[:, 0], xy[:, 1], side_xy[:, 0], side_xy[:, 1], topo)
    np.testing.assert_array_equal(layout.cells, [1, 2, 3])
    assert len(layout.offsets) == 6
    np.testing.assert_array_equal(layout.axes[:, axis], 1)
    np.testing.assert_array_equal(layout.axes[:, 1-axis], 0)


@pytest.mark.parametrize('rotated', [False, True])
def test_variable_bed_still_water_is_bitwise_unchanged(tmp_path, rotated):
    result, initial, *_ = evolve(tmp_path, strip_cells(rotated),
        lambda x, y, i: ((i % 7)-3)*.125, lambda x, y, z, i: 2.-z, seconds=60.)
    np.testing.assert_array_equal(result.depth.view('uint32'), initial.view('uint32'))
    assert np.all(result.q == 0.)


@pytest.mark.parametrize('dry_cells', [(1,), (3,), (3, .02), (1, 3)])
def test_dry_tangent_neighbours_use_one_side_or_zero(tmp_path, dry_cells):
    threshold = dry_cells == (3, .02)
    unavailable = (3,) if threshold else dry_cells
    result, _, _, topo, _ = evolve(tmp_path, strip_cells(), lambda x, y, i: .001*y,
        lambda x, y, z, i: (.02 if threshold else 0.) if i in unavailable else 1.)
    middle = (topo[:, 0] == 2) & (topo[:, 1] == 2)
    if len(unavailable) == 1:
        assert np.max(np.abs(result.q[middle])) < 3e-7
    else:
        np.testing.assert_allclose(np.sort(result.q[middle]), [-.00040875, .00040875], atol=3e-7, rtol=0)


def test_one_cell_caches_both_axes_and_reconstructs_linear_plane(tmp_path):
    cells = [(0, 8, 0, 8), (8, 12, 0, 4), (8, 12, 4, 8),
             (0, 4, 8, 12), (4, 8, 8, 12), (8, 12, 8, 12)]
    a, b = .0013, -.0007
    result, _, xy, topo, side_xy = evolve(tmp_path, cells, lambda x, y, i: a*x+b*y,
                                         lambda x, y, z, i: 1.)
    layout = hanging_geometry(xy[:, 0], xy[:, 1], side_xy[:, 0], side_xy[:, 1], topo)
    np.testing.assert_array_equal(layout.cells, [1])
    np.testing.assert_array_equal(layout.axes, [[1, 1]])
    edges = layout.edge_index >= 0
    expected = -9.81*.125*np.where(topo[edges, 0] == 2, a, b)
    np.testing.assert_allclose(result.q[edges], expected, atol=3e-7, rtol=0)


def test_aligned_mesh_allocates_no_reconstruction(tmp_path):
    cells = [(x, x+4, y, y+4) for x in (0, 4) for y in (0, 4, 8)]
    xy, _, topo, side_xy = square_mesh(tmp_path, cells, lambda x, y, i: 0.)
    assert hanging_geometry(xy[:, 0], xy[:, 1], side_xy[:, 0], side_xy[:, 1], topo) is None


def split_face_reconstruction():
    """A coarse cell has two upper and two right subedges of unequal neighbours."""
    # 1: coarse centre; 2/3: right neighbours; 4: bottom; 5/6: upper neighbours.
    x = np.array([0, 4, 10, 10, 4, 2, 6], dtype=np.float32)
    y = np.array([0, 4, 2, 6, -4, 10, 10], dtype=np.float32)
    side_x = np.array([0, 8, 8, 4, 2, 6], dtype=np.float32)
    side_y = np.array([0, 2, 6, 0, 8, 8], dtype=np.float32)
    topology = np.array([[0, 0, 0], [2, 1, 2], [2, 1, 3],
                         [1, 4, 1], [1, 1, 5], [1, 1, 6]], dtype=np.int32)
    layout = hanging_geometry(x, y, side_x, side_y, topology)
    np.testing.assert_array_equal(layout.cells, [1])
    np.testing.assert_array_equal(layout.axes, [[1, 1]])
    face_ptr = np.zeros((len(x), 4), dtype=np.int32)
    face_count = np.zeros_like(face_ptr)
    # Direction order is left, right, bottom, top, as used by the solver.
    face_ptr[1] = [0, 0, 2, 3]
    face_count[1] = [0, 2, 1, 2]
    depth = ti.field(ti.f32, shape=len(x))
    pressure = PressureReconstruction(layout,
        bed=copy_to_taichi(np.zeros(len(x), dtype=np.float32), ti.f32, None), depth=depth,
        x=copy_to_taichi(x, ti.f32, None), y=copy_to_taichi(y, ti.f32, None),
        lengths=copy_to_taichi(np.array([0, 4, 4, 8, 4, 4], dtype=np.float32), ti.f32, None),
        topology=copy_to_taichi(topology, ti.i32, None),
        face_ptr=copy_to_taichi(face_ptr, ti.i32, None),
        face_count=copy_to_taichi(face_count, ti.i32, None),
        sides=copy_to_taichi(np.array([1, 2, 3, 4, 5], dtype=np.int32), ti.i32, None),
        min_h=.25)
    differences = ti.field(ti.f32, shape=2)

    @ti.kernel
    def update():
        for slot in pressure.cells:
            pressure.update_cell(slot)

    @ti.kernel
    def reconstruct():
        for side in range(1, 3):
            lower, higher = pressure.topology[side, 1], pressure.topology[side, 2]
            raw = pressure.depth[higher] - pressure.depth[lower]
            differences[side - 1] = pressure.difference(side, lower, higher, raw)

    def step(values):
        depth.from_numpy(np.array([0, *values], dtype=np.float32))
        update()
        reconstruct()
        return pressure.gradients.to_numpy()[0, 1], differences.to_numpy()

    return step


def test_partly_dry_split_neighbour_face_uses_other_face_fallback():
    step = split_face_reconstruction()
    # The upper face has a wet half and a half exactly at min_h. Its wet half
    # must not be mistaken for a complete usable face.
    for upper_wet in (1.5, 2.0):
        gradient, difference = step([1., 1., 1., .5, upper_wet, .25])
        assert gradient == pytest.approx(.5 / 8, abs=1e-7)
        np.testing.assert_allclose(difference, [.125, -.125], atol=2e-7, rtol=0)


def test_consecutive_threshold_crossings_recompute_gradients_and_differences():
    step = split_face_reconstruction()
    # Each row is a new time step on the same Taichi fields and reconstruction.
    # Check both the cached gradient and the corrected hanging-edge differences.
    states = [
        ([1., 1., 1., .5, 1.5, .25], .5 / 8, [.125, -.125]),
        ([1., 1., 1., .5, 1.5, 1.75], 1.125 / 14, [2 * 1.125 / 14, -2 * 1.125 / 14]),
        ([1., 1., 1., .25, 1.5, 1.75], .625 / 6, [2 * .625 / 6, -2 * .625 / 6]),
        ([.25, 1., 1., .5, 1.5, 1.75], 0., [.75, .75]),
        ([1.25, 1., 1., .5, 1.5, .25], .75 / 8, [-.25 + 2 * .75 / 8, -.25 - 2 * .75 / 8]),
        ([1.25, .25, 1., .5, 1.5, .25], .75 / 8, [-1., -.25 - 2 * .75 / 8]),
        ([1.25, 1., 1., .25, .25, .25], 0., [-.25, -.25]),
    ]
    for depths, expected_gradient, expected_difference in states:
        gradient, difference = step(depths)
        assert gradient == pytest.approx(expected_gradient, abs=2e-7)
        np.testing.assert_allclose(difference, expected_difference, atol=3e-7, rtol=0)
