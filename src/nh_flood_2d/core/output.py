"""FDB and CSV storage for sparse solver outputs and resumable state.

This module performs host-side I/O only. The solver owns sampling, event timing,
device reductions, and the meaning of checkpoint metadata.
"""

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

import fastdb4py as fdb
import numpy as np

from ..schema.feature import F32Value, IndexLike, Ne, Ns, U8Value


STATION_FIELDS = ('elapsed_seconds', 'time', 'station', 'cell_id',
                  'depth', 'level', 'u', 'v')
_CHECKPOINT_FLOATS = ('depth', 'bed', 'q', 'rain_time', 'infiltration',
                      'max_depth', 'max_speed')
_CHECKPOINT_INTS = ('clock_hi', 'clock_lo', 'gate_open')
_CHECKPOINT_TIME = ('first_time_hi', 'first_time_lo')


@dataclass
class Extrema:
    max_depth: np.ndarray
    first_time_hi: np.ndarray
    first_time_lo: np.ndarray
    max_speed: np.ndarray


@dataclass
class Checkpoint:
    depth: np.ndarray
    bed: np.ndarray
    q: np.ndarray
    rain_time: np.ndarray
    infiltration: np.ndarray
    max_depth: np.ndarray
    first_time_hi: np.ndarray
    first_time_lo: np.ndarray
    max_speed: np.ndarray
    clock_hi: int
    clock_lo: int
    gate_open: bool
    metadata_json: str


def _float_array(value, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite one-dimensional F32 array')
    return array


def _uint_array(value, name):
    array = np.asarray(value)
    if (array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or
            not np.all(np.isfinite(array)) or np.any(array != np.floor(array)) or
            np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max)):
        raise ValueError(f'{name} must be a one-dimensional U32 array')
    return array.astype(np.uint32)


def _save(path, definitions, fill):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    db = fdb.ORM.truncate(definitions)
    try:
        fill(db)
        db.save(str(temporary))
    finally:
        db.unlink()
    os.replace(temporary, path)


def _read_named(path, feature, name, column, expected_count=None):
    db = fdb.ORM.load(str(path), from_file=True)
    try:
        values = getattr(db[feature][name].column, column).copy()
    finally:
        db.close()
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(f'{path}: {name} has {len(values)} rows, expected {expected_count}')
    return values


def write_bed(path, bed):
    values = _float_array(bed, 'bed')
    _save(path, [fdb.TableDefn(F32Value, len(values), 'bed')],
          lambda db: db[F32Value]['bed'].column.value.__setitem__(slice(None), values))


def read_bed(path, expected_count=None):
    return _read_named(path, F32Value, 'bed', 'value', expected_count)


def write_depth_frame(path, depth):
    values = _float_array(depth, 'depth')
    if np.any(values < 0):
        raise ValueError('depth must be nonnegative')
    _save(path, [fdb.TableDefn(F32Value, len(values), 'depth')],
          lambda db: db[F32Value]['depth'].column.value.__setitem__(slice(None), values))


def read_depth_frame(path, expected_count=None):
    values = _read_named(path, F32Value, 'depth', 'value', expected_count)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f'{path}: invalid depth frame')
    return values


def list_depth_frames(output_root):
    """Return (elapsed seconds, path) pairs in numeric order."""
    frames = []
    for path in (Path(output_root) / 'frames').glob('depth_*.fdb'):
        try:
            seconds = int(path.stem.removeprefix('depth_'))
        except ValueError as exc:
            raise ValueError(f'Invalid depth frame name: {path}') from exc
        frames.append((seconds, path))
    return sorted(frames)


def write_extrema(path, max_depth, first_time_hi, first_time_lo, max_speed):
    arrays = {
        'max_depth': _float_array(max_depth, 'max_depth'),
        'max_speed': _float_array(max_speed, 'max_speed'),
        'first_time_hi': _uint_array(first_time_hi, 'first_time_hi'),
        'first_time_lo': _uint_array(first_time_lo, 'first_time_lo'),
    }
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError('extrema arrays must have the same number of cells')
    definitions = [fdb.TableDefn(F32Value, len(arrays[name]), name)
                   for name in ('max_depth', 'max_speed')]
    definitions += [fdb.TableDefn(IndexLike, len(arrays[name]), name)
                    for name in ('first_time_hi', 'first_time_lo')]

    def fill(db):
        for name in ('max_depth', 'max_speed'):
            db[F32Value][name].column.value[:] = arrays[name]
        for name in ('first_time_hi', 'first_time_lo'):
            db[IndexLike][name].column.index[:] = arrays[name]

    _save(path, definitions, fill)


def read_extrema(path, expected_count=None):
    db = fdb.ORM.load(str(path), from_file=True)
    try:
        result = Extrema(
            db[F32Value]['max_depth'].column.value.copy(),
            db[IndexLike]['first_time_hi'].column.index.copy(),
            db[IndexLike]['first_time_lo'].column.index.copy(),
            db[F32Value]['max_speed'].column.value.copy(),
        )
    finally:
        db.close()
    if len({len(result.max_depth), len(result.first_time_hi), len(result.first_time_lo),
            len(result.max_speed)}) != 1 or (expected_count is not None and
                                              len(result.max_depth) != expected_count):
        raise ValueError(f'{path}: extrema cell count differs from mesh')
    return result


def write_checkpoint(path, *, depth, bed, q, rain_time, infiltration,
                     max_depth, first_time_hi, first_time_lo, max_speed,
                     clock_hi, clock_lo, gate_open, metadata_json):
    """Persist an already completed solver step; metadata_json is kept verbatim."""
    arrays = {name: _float_array(value, name) for name, value in
              [('depth', depth), ('bed', bed), ('q', q), ('rain_time', rain_time),
               ('infiltration', infiltration), ('max_depth', max_depth),
               ('max_speed', max_speed)]}
    if len({len(arrays[name]) for name in ('depth', 'bed', 'rain_time', 'infiltration',
                                         'max_depth', 'max_speed')}) != 1:
        raise ValueError('checkpoint cell arrays must have the same length')
    times = {name: _uint_array(value, name) for name, value in
             [('first_time_hi', first_time_hi), ('first_time_lo', first_time_lo)]}
    if any(len(value) != len(arrays['depth']) for value in times.values()):
        raise ValueError('checkpoint first-time arrays must match cell count')
    if np.any(arrays['depth'] < 0):
        raise ValueError('checkpoint depth must be nonnegative')
    if not isinstance(metadata_json, str):
        raise TypeError('metadata_json must be a JSON string')
    json.loads(metadata_json)
    if not isinstance(gate_open, (bool, np.bool_)):
        raise TypeError('gate_open must be boolean')
    integers = [_uint_array([value], name) for name, value in
                [('clock_hi', clock_hi), ('clock_lo', clock_lo), ('gate_open', int(gate_open))]]
    encoded = metadata_json.encode('utf-8')
    definitions = [fdb.TableDefn(F32Value, len(array), name) for name, array in arrays.items()]
    definitions += [fdb.TableDefn(IndexLike, 1, name) for name in _CHECKPOINT_INTS]
    definitions += [fdb.TableDefn(IndexLike, len(value), name)
                    for name, value in times.items()]
    definitions.append(fdb.TableDefn(U8Value, len(encoded), 'state_meta'))

    def fill(db):
        for name, array in arrays.items():
            db[F32Value][name].column.value[:] = array
        for name, array in zip(_CHECKPOINT_INTS, integers):
            db[IndexLike][name].column.index[:] = array
        for name, array in times.items():
            db[IndexLike][name].column.index[:] = array
        db[U8Value]['state_meta'].column.value[:] = np.frombuffer(encoded, dtype=np.uint8)

    _save(path, definitions, fill)


def read_checkpoint(path, expected_cells=None, expected_sides=None):
    db = fdb.ORM.load(str(path), from_file=True)
    try:
        floats = {name: db[F32Value][name].column.value.copy() for name in _CHECKPOINT_FLOATS}
        ints = {name: db[IndexLike][name].column.index.copy() for name in _CHECKPOINT_INTS}
        times = {name: db[IndexLike][name].column.index.copy() for name in _CHECKPOINT_TIME}
        metadata_json = db[U8Value]['state_meta'].column.value.tobytes().decode('utf-8')
    finally:
        db.close()
    if len({len(floats[name]) for name in ('depth', 'bed', 'rain_time', 'infiltration',
                                         'max_depth', 'max_speed')}) != 1:
        raise ValueError(f'{path}: inconsistent checkpoint cell counts')
    if any(len(value) != len(floats['depth']) for value in times.values()):
        raise ValueError(f'{path}: inconsistent checkpoint first-time counts')
    if expected_cells is not None and len(floats['depth']) != expected_cells:
        raise ValueError(f'{path}: checkpoint cell count differs from mesh')
    if expected_sides is not None and len(floats['q']) != expected_sides:
        raise ValueError(f'{path}: checkpoint side count differs from mesh')
    if any(len(value) != 1 for value in ints.values()):
        raise ValueError(f'{path}: checkpoint scalar count must be one')
    if (any(not np.all(np.isfinite(value)) for value in floats.values())
            or np.any(floats['depth'] < 0)):
        raise ValueError(f'{path}: invalid checkpoint F32 state')
    if ints['gate_open'][0] not in (0, 1):
        raise ValueError(f'{path}: invalid checkpoint gate state')
    json.loads(metadata_json)
    return Checkpoint(floats['depth'], floats['bed'], floats['q'], floats['rain_time'],
                      floats['infiltration'], floats['max_depth'], times['first_time_hi'],
                      times['first_time_lo'], floats['max_speed'], int(ints['clock_hi'][0]),
                      int(ints['clock_lo'][0]), bool(ints['gate_open'][0]), metadata_json)


def resolve_station_cells(points, ne_fdb_path, ns_fdb_path):
    """Resolve closed rectangle overlaps by the smallest real element ID."""
    ne_db = fdb.ORM.load(str(ne_fdb_path), from_file=True)
    ns_db = fdb.ORM.load(str(ns_fdb_path), from_file=True)
    try:
        ne = ne_db[Ne][Ne]
        ns = ns_db[Ns][Ns]
        sides = ne_db[IndexLike]['isl_data'].column.index
        bounds = []
        for direction, coordinates in [('l', ns.column.x), ('b', ns.column.y),
                                       ('r', ns.column.x), ('t', ns.column.y)]:
            pointers = ne_db[IndexLike][f'isl_ptr_{direction}'].column.index
            bounds.append(coordinates[sides[pointers]].astype(np.float64))
        left, bottom, right, top = bounds
        resolved = {}
        for name, (x, y) in points.items():
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError(f'Station {name}: coordinates must be finite')
            candidates = np.flatnonzero((left[1:] <= x) & (x <= right[1:]) &
                                        (bottom[1:] <= y) & (y <= top[1:]))
            if not len(candidates):
                raise ValueError(f'Station {name}: ({x}, {y}) lies outside the mesh')
            resolved[name] = int(candidates[0] + 1)
        return resolved
    finally:
        ne_db.close()
        ns_db.close()


def append_station_samples(path, *, elapsed_seconds, time, samples):
    """Append one 300-second sample per station in stable name order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=STATION_FIELDS)
        if needs_header:
            writer.writeheader()
        for name in sorted(samples):
            sample = samples[name]
            writer.writerow({'elapsed_seconds': elapsed_seconds, 'time': time,
                             'station': name, **{key: sample[key] for key in
                             ('cell_id', 'depth', 'level', 'u', 'v')}})


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def read_manifest(path):
    return json.loads(Path(path).read_text())


class BedTimeline:
    """Replay bed changes in time order without rereading the base grid."""

    def __init__(self, output_root, manifest=None, expected_count=None):
        output_root = Path(output_root)
        manifest = read_manifest(output_root.parent / 'manifest.json') if manifest is None else manifest
        self.bed = read_bed(output_root / 'bed.fdb', expected_count)
        self.events = sorted(manifest.get('bed_events', []),
                             key=lambda event: event['elapsed_seconds'])
        self.cursor = 0
        self.seconds = float('-inf')

    def at(self, elapsed_seconds):
        if elapsed_seconds < self.seconds:
            raise ValueError('BedTimeline requires nondecreasing frame times')
        while self.cursor < len(self.events):
            event = self.events[self.cursor]
            if event['elapsed_seconds'] > elapsed_seconds:
                break
            cell_ids = _uint_array(event['cell_ids'], 'bed event cell_ids')
            new_bed = _float_array(event['new_bed'], 'bed event new_bed')
            if len(cell_ids) != len(new_bed) or np.any(cell_ids >= len(self.bed)):
                raise ValueError('bed event has invalid cell ids or values')
            self.bed[cell_ids] = new_bed
            self.cursor += 1
        self.seconds = elapsed_seconds
        return self.bed


def read_bed_at(output_root, elapsed_seconds, manifest=None, expected_count=None):
    """Replay documented bed events through a frame's elapsed time."""
    return BedTimeline(output_root, manifest, expected_count).at(elapsed_seconds)
