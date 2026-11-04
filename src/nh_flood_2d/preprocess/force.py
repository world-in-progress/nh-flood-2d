"""Validated absolute-time forcing and exterior-side boundary selection."""
from dataclasses import dataclass
import csv
import json

import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely import STRtree, intersection, length, linestrings
from shapely.geometry import shape
from shapely.ops import unary_union

from ..input import DomainConfig, ForceConfig
from ..input.force import absolute_time
from ..schema.feature import Ne, Ns, IndexLike, SideTopoInfo


@dataclass
class ExteriorEdges:
    side_ids: np.ndarray
    cell_ids: np.ndarray
    face_ids: np.ndarray  # left, right, bottom, top = 0, 1, 2, 3
    lengths: np.ndarray
    segments: np.ndarray  # edge x endpoint x xy


@dataclass
class MeshGeometry:
    x: np.ndarray
    y: np.ndarray
    bounds: np.ndarray
    area: np.ndarray
    exterior_edges: ExteriorEdges

    @classmethod
    def from_fdb(cls, ne_db, ns_db):
        ne, ns = ne_db.table(Ne), ns_db.table(Ns)
        x, y = ne.column.x.astype(float), ne.column.y.astype(float)
        sides = ne_db.table(IndexLike, name='isl_data').column.index
        coords = []
        for key, values in [('l', ns.column.x), ('b', ns.column.y),
                            ('r', ns.column.x), ('t', ns.column.y)]:
            ptr = ne_db.table(IndexLike, name=f'isl_ptr_{key}').column.index
            coords.append(values[sides[ptr]].astype(float))
        bounds = np.column_stack(coords)
        widths = bounds[:, 2] - bounds[:, 0]
        heights = bounds[:, 3] - bounds[:, 1]
        area = widths * heights
        area[0] = 0
        if np.any(widths[1:] <= 0) or np.any(heights[1:] <= 0):
            raise ValueError('Mesh cells must have positive rectangular areas')
        if np.any(widths[1:] != heights[1:]):
            raise ValueError('Mesh cells must be square for the edge-length CFL scale')
        topo = ns_db.table(SideTopoInfo).column.info.reshape(-1, 3)
        outer = (topo[:, 1] == 0) ^ (topo[:, 2] == 0)
        side_ids = np.flatnonzero(outer)
        cells = topo[side_ids, 1] + topo[side_ids, 2]
        orient = topo[side_ids, 0]
        vertical, horizontal = orient == 2, orient == 1
        if np.any(~(vertical | horizontal)):
            raise ValueError('Exterior sides must be horizontal or vertical')
        sx, sy = ns.column.x[side_ids].astype(float), ns.column.y[side_ids].astype(float)
        lengths = ns.column.length[side_ids].astype(float)
        on_faces = np.column_stack((
            vertical & (sx == bounds[cells, 0]),
            vertical & (sx == bounds[cells, 2]),
            horizontal & (sy == bounds[cells, 1]),
            horizontal & (sy == bounds[cells, 3]),
        ))
        if np.any(on_faces.sum(axis=1) != 1):
            raise ValueError('Exterior side does not lie on one cell face')
        segments = np.empty((len(side_ids), 2, 2), dtype=float)
        segments[:, :, 0] = sx[:, None]
        segments[:, :, 1] = sy[:, None]
        segments[vertical, 0, 1] -= lengths[vertical] / 2
        segments[vertical, 1, 1] += lengths[vertical] / 2
        segments[horizontal, 0, 0] -= lengths[horizontal] / 2
        segments[horizontal, 1, 0] += lengths[horizontal] / 2
        edges = ExteriorEdges(side_ids, cells, on_faces.argmax(axis=1), lengths, segments)
        return cls(x, y, bounds, area, edges)


@dataclass
class TimeSeries:
    times: np.ndarray   # seconds relative to simulation start, float64
    values: np.ndarray

    @classmethod
    def load(cls, path, origin, begin, end):
        with open(path, newline='') as file:
            rows = list(csv.DictReader(file))
        times = np.array([(absolute_time(r['time']) - origin).total_seconds() for r in rows])
        values = np.array([float(r['value']) for r in rows])
        if len(times) < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(times) <= 0):
            raise ValueError(f'{path}: need finite values and strictly increasing times (at least two)')
        if times[0] > begin or times[-1] < end:
            raise ValueError(f'{path}: series does not cover [{begin}, {end}] seconds from start')
        return cls(times, values)

    def at(self, seconds):
        return float(np.interp(seconds, self.times, self.values))


def load_line(path, epsg):
    with open(path) as file:
        data = json.load(file)
    if 'crs' not in data:
        raise ValueError(f'{path}: GeoJSON must declare its CRS')
    if CRS.from_string(data['crs']['properties']['name']) != CRS.from_epsg(epsg):
        raise ValueError(f'{path}: boundary CRS differs from mesh EPSG:{epsg}')
    if data['type'] == 'FeatureCollection':
        geometries = [shape(f['geometry']) for f in data['features']]
    elif data['type'] == 'Feature':
        geometries = [shape(data['geometry'])]
    else:
        geometries = [shape(data)]
    if not geometries or any(g.geom_type not in ('LineString', 'MultiLineString')
                             or g.is_empty or not g.is_valid for g in geometries):
        raise ValueError(f'{path}: expected nonempty LineString or MultiLineString')
    return unary_union(geometries)


@dataclass
class Boundary:
    name: str
    kind: str
    cells: np.ndarray
    weights: np.ndarray
    side_ids: np.ndarray
    series: TimeSeries

    def distribute(self, q):
        return self.weights * q


def load_boundaries(config, mesh, epsg, origin, begin, end):
    if not config:
        return []
    edges = mesh.exterior_edges
    geometries = linestrings(edges.segments)
    tree = STRtree(geometries)
    claimed = np.zeros(len(mesh.x), dtype=bool)
    boundaries = []
    for b in config:
        line = load_line(b.line, epsg)
        candidates = tree.query(line)
        selected = np.sort(candidates[length(intersection(geometries[candidates], line)) > 0])
        side_ids = edges.side_ids[selected]
        if len(side_ids) == 0:
            raise ValueError(f'{b.name}: boundary selects no exterior edges; '
                             'snap the line to mesh exterior edges in GIS')
        cells, inverse = np.unique(edges.cell_ids[selected], return_inverse=True)
        if np.any(claimed[cells]):
            raise ValueError(f'{b.name}: boundary cells overlap another boundary')
        claimed[cells] = True
        lengths = np.bincount(inverse, weights=edges.lengths[selected])
        weights = lengths / lengths.sum(dtype=np.float64)
        boundaries.append(Boundary(b.name, b.kind, cells, weights, side_ids,
                                   TimeSeries.load(b.series, origin, begin, end)))
    return boundaries


@dataclass
class RainField:
    starts: np.ndarray
    ends: np.ndarray
    millimetres: np.ndarray  # bands x unique sampled pixels, never bands x cells
    pixel_index: np.ndarray  # cell -> sampled pixel; 0th cell unused

    @classmethod
    def load(cls, config, mesh, epsg, origin, begin, end):
        with open(config.intervals, newline='') as file:
            rows = list(csv.DictReader(file))
        starts = np.array([(absolute_time(r['interval_start']) - origin).total_seconds() for r in rows])
        ends = np.array([(absolute_time(r['interval_end']) - origin).total_seconds() for r in rows])
        if not len(rows) or np.any(ends <= starts) or np.any(starts[1:] != ends[:-1]):
            raise ValueError('Rainfall intervals must be continuous, positive and nonoverlapping')
        if starts[0] > begin or ends[-1] < end:
            raise ValueError('Rainfall intervals do not cover the simulation window')
        with rasterio.open(config.raster) as src:
            if src.crs != CRS.from_epsg(epsg):
                raise ValueError('Rainfall CRS differs from mesh CRS')
            if src.count != len(rows):
                raise ValueError('Rainfall band count differs from interval count')
            rr, cc = rasterio.transform.rowcol(src.transform, mesh.x[1:], mesh.y[1:])
            rr, cc = np.asarray(rr), np.asarray(cc)
            if np.any((rr < 0) | (rr >= src.height) | (cc < 0) | (cc >= src.width)):
                raise ValueError('Cell centre is outside the rainfall raster')
            pixels, inverse = np.unique(rr * src.width + cc, return_inverse=True)
            coords = [src.xy(int(p // src.width), int(p % src.width)) for p in pixels]
            rain = np.ma.column_stack(list(src.sample(coords, masked=True)))
            if np.any(np.ma.getmaskarray(rain)) or not np.all(np.isfinite(rain)) or np.any(rain < 0):
                raise ValueError('Rainfall at a cell centre is NoData, nonfinite or negative')
        mapping = np.zeros(len(mesh.x), dtype=np.int32)
        mapping[1:] = inverse
        return cls(starts, ends, np.asarray(rain, dtype=float), mapping)

    def index_at(self, seconds):
        return int(np.searchsorted(self.ends, seconds, side='right'))

    def rates(self, index):
        return self.millimetres[index] * (0.001 / (self.ends[index] - self.starts[index]))


@dataclass
class Forcing:
    boundaries: list[Boundary]
    rain: RainField
    events: np.ndarray
    gate_opening_time: float | None

    @classmethod
    def load(cls, domain: DomainConfig, config: ForceConfig, mesh):
        begin, end = 0., domain.duration
        boundaries = load_boundaries(config.boundaries, mesh, domain.epsg_code,
                                     domain.start, begin, end)
        rain = RainField.load(config.rainfall, mesh, domain.epsg_code, domain.start, begin, end)
        crossings = []
        opening = None
        if config.gate:
            series = next(b.series for b in boundaries if b.name == config.gate.level_boundary)
            offset = series.values - config.gate.threshold
            rising = (offset[:-1] <= 0) & (offset[1:] > 0)
            candidates = []
            for i in np.flatnonzero((offset[:-1] * offset[1:] < 0) | rising):
                crossing = series.times[i] - offset[i] * (series.times[i+1] - series.times[i]) / (offset[i+1] - offset[i])
                crossings.append(crossing)
                if rising[i] and 0. <= crossing < end:
                    candidates.append(crossing)
            if series.at(0.) > config.gate.threshold:
                opening = -float(domain.warmup_duration)
            elif candidates:
                opening = float(min(candidates))
        events = np.unique(np.concatenate([rain.ends, [0., end], crossings, [] if opening is None else [opening]] + [b.series.times for b in boundaries]))
        return cls(boundaries, rain, events, opening)
