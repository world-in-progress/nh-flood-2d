"""Host-only contracts for sparse output and restart files."""

import csv
from pathlib import Path

import fastdb4py as fdb
import numpy as np
import pytest

from src.nh_flood_2d.core.output import (
    append_station_samples, list_depth_frames, read_bed_at, read_checkpoint,
    read_depth_frame, read_extrema, resolve_station_cells, write_bed,
    write_checkpoint, write_depth_frame, write_extrema, write_manifest,
)
from src.nh_flood_2d.schema.feature import F32Value, IndexLike, Ne, Ns


def test_depth_frames_are_single_f32_arrays_and_bed_events_rebuild_wse(tmp_path):
    root = tmp_path / 'output'
    write_bed(root / 'bed.fdb', [0, 10, 20])
    write_depth_frame(root / 'frames/depth_900.fdb', [0, .25, .5])
    write_depth_frame(root / 'frames/depth_0.fdb', [0, 1, 2])
    write_manifest(root.parent / 'manifest.json', {'bed_events': [
        {'elapsed_seconds': 450, 'cell_ids': [1], 'new_bed': [0]},
    ]})

    assert [seconds for seconds, _ in list_depth_frames(root)] == [0, 900]
    np.testing.assert_array_equal(read_bed_at(root, 0), [0, 10, 20])
    np.testing.assert_array_equal(read_bed_at(root, 900), [0, 0, 20])
    depth = read_depth_frame(root / 'frames/depth_900.fdb', expected_count=3)
    assert depth.dtype == np.float32
    np.testing.assert_array_equal(depth, [0, .25, .5])
    db = fdb.ORM.load(str(root / 'frames/depth_900.fdb'), from_file=True)
    try:
        np.testing.assert_array_equal(db[F32Value]['depth'].column.value, depth)
    finally:
        db.close()
    with pytest.raises(ValueError, match='expected 4'):
        read_depth_frame(root / 'frames/depth_900.fdb', expected_count=4)


def test_checkpoint_preserves_exact_f32_and_metadata_and_extrema(tmp_path):
    state_path = tmp_path / 'checkpoint.fdb'
    depth = np.array([0, np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)
    metadata_json = '{ "budget_units": ["12345678901234567890"], "phase": "simulation" }'
    write_checkpoint(state_path, depth=depth, bed=[0, 10000], q=[0, -0.25, 0.5],
                     rain_time=[0, 12], infiltration=[0, .0001],
                     max_depth=[0, .5], first_time_hi=[0, 2], first_time_lo=[0, 3],
                     max_speed=[0, 1.25], clock_hi=2, clock_lo=4294967295, gate_open=True,
                     metadata_json=metadata_json)
    state = read_checkpoint(state_path, expected_cells=2, expected_sides=3)
    np.testing.assert_array_equal(state.depth.view(np.uint32), depth.view(np.uint32))
    np.testing.assert_array_equal(state.bed, [0, 10000])
    np.testing.assert_array_equal(state.q, [0, -.25, .5])
    np.testing.assert_array_equal(state.max_depth, [0, .5])
    np.testing.assert_array_equal(state.first_time_hi, [0, 2])
    np.testing.assert_array_equal(state.first_time_lo, [0, 3])
    np.testing.assert_array_equal(state.max_speed, [0, 1.25])
    assert not hasattr(state, 'next_dt')
    assert (state.clock_hi, state.clock_lo, state.gate_open) == (2, 4294967295, True)
    assert state.metadata_json == metadata_json
    with pytest.raises(ValueError, match='side count'):
        read_checkpoint(state_path, expected_sides=4)
    with pytest.raises(ValueError, match='depth'):
        write_checkpoint(state_path, depth=[0, np.nan], bed=[0, 10000], q=[0, 0, 0],
                         rain_time=[0, 0], infiltration=[0, 0], max_depth=depth,
                         first_time_hi=[0, 0], first_time_lo=[0, 0], max_speed=[0, 0],
                         clock_hi=0, clock_lo=0, gate_open=False, metadata_json='{}')

    maxima_path = tmp_path / 'extrema.fdb'
    write_extrema(maxima_path, [0, .5], [0, 2], [0, 4294967295], [0, 1.25])
    maxima = read_extrema(maxima_path, expected_count=2)
    np.testing.assert_array_equal(maxima.max_depth, [0, .5])
    np.testing.assert_array_equal(maxima.first_time_hi, [0, 2])
    np.testing.assert_array_equal(maxima.first_time_lo, [0, 4294967295])
    np.testing.assert_array_equal(maxima.max_speed, [0, 1.25])


def _two_cell_mesh(ne_path: Path, ns_path: Path):
    ne_db = fdb.ORM.truncate([fdb.TableDefn(Ne, 3),
        fdb.TableDefn(IndexLike, 9, 'isl_data'),
        *[fdb.TableDefn(IndexLike, 3, f'isl_ptr_{side}') for side in 'lrbt']])
    ne_db[Ne][Ne].column.x[:] = [0, 5, 15]
    ne_db[Ne][Ne].column.y[:] = [0, 5, 5]
    ne_db[IndexLike]['isl_data'].column.index[:] = [0, 1, 2, 2, 3, 4, 5, 6, 7]
    for side, pointers in {'l': [0, 1, 2], 'r': [0, 3, 4],
                           'b': [0, 5, 6], 't': [0, 7, 8]}.items():
        ne_db[IndexLike][f'isl_ptr_{side}'].column.index[:] = pointers
    ne_db.save(str(ne_path))
    ne_db.unlink()

    ns_db = fdb.ORM.truncate([fdb.TableDefn(Ns, 8)])
    ns_db[Ns][Ns].column.x[:] = [0, 0, 10, 20, 5, 15, 5, 15]
    ns_db[Ns][Ns].column.y[:] = [0, 5, 5, 5, 0, 0, 10, 10]
    ns_db.save(str(ns_path))
    ns_db.unlink()


def test_station_boundary_chooses_smallest_cell_and_csv_has_one_row_per_station(tmp_path):
    ne, ns = tmp_path / 'ne.fdb', tmp_path / 'ns.fdb'
    _two_cell_mesh(ne, ns)
    cells = resolve_station_cells({'joint': (10, 5), 'right': (15, 5)}, ne, ns)
    assert cells == {'joint': 1, 'right': 2}
    with pytest.raises(ValueError, match='outside the mesh'):
        resolve_station_cells({'missing': (25, 5)}, ne, ns)

    csv_path = tmp_path / 'stations.csv'
    append_station_samples(csv_path, elapsed_seconds=300, time='2024-01-01T00:05:00+00:00',
                           samples={'right': {'cell_id': 2, 'depth': .5, 'level': 20.5,
                                              'u': 1, 'v': 2},
                                    'joint': {'cell_id': 1, 'depth': .25, 'level': 10.25,
                                              'u': 3, 'v': 4}})
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['station'] for row in rows] == ['joint', 'right']
    assert rows[0]['cell_id'] == '1'
    assert rows[0]['level'] == '10.25'
