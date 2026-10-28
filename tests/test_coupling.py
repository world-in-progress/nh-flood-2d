"""Surface/pipe exchange conservation and real SWMM clock checks."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import fastdb4py as fdb
import numpy as np
import pytest

from src.nh_flood_2d.core.coupled.exchange import compute_exchange
from src.nh_flood_2d.core.coupled.runtime import PipeCoupling
from src.nh_flood_2d.input.pipe import PipeConfig
from src.nh_flood_2d.schema.feature import F32Value, IndexLike, Node, PipeTopo


SMALL_INP = """[TITLE]
Small coupled solver validation
[OPTIONS]
FLOW_UNITS CMS
INFILTRATION HORTON
FLOW_ROUTING DYNWAVE
LINK_OFFSETS DEPTH
ALLOW_PONDING NO
START_DATE 09/07/2023
START_TIME 08:00:00
REPORT_START_DATE 09/07/2023
REPORT_START_TIME 08:00:00
END_DATE 09/07/2023
END_TIME 08:01:00
REPORT_STEP 00:00:01
WET_STEP 00:00:01
DRY_STEP 00:00:01
ROUTING_STEP 00:00:01
VARIABLE_STEP 0
MINIMUM_STEP 0.5
[JUNCTIONS]
J1 0 2 0 0 0
[OUTFALLS]
O1 -1 FIXED 0 NO
[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0
[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1
[COORDINATES]
J1 5 5
O1 15 5
[REPORT]
INPUT NO
CONTROLS NO
SUBCATCHMENTS NONE
NODES ALL
LINKS ALL
"""



pytestmark = pytest.mark.usefixtures('taichi_cpu')

def test_shared_cells_and_outfall_conserve_exchange():
    source, boundaries, diag, relationships = compute_exchange(
        5., np.array([0., .25, .25]), np.array([0., 5., 5.]),
        np.array([0., 100., 100.]), np.array([0, 10, 10]),
        ['J1', 'J2', 'O1'], np.array([False, False, True]),
        np.array([1, 1, 2]), np.array([2, 2, 2]),
        np.array([0, 2, 3, 3]), np.array([6., 6., 0.]),
        {'J1': 8., 'J2': 0., 'O1': 0.},
    )
    assert source.dtype == np.float32
    assert relationships[1] == (0, 1)
    assert relationships[2] == (0, 1)
    assert boundaries['O1'] == {'level': 5.25, 'flow': 0.0}
    assert diag['overflow_m3_s'] > 0
    assert diag['drainage_m3_s'] > 0
    assert diag['surface_pipe_balance_m3_s'] == pytest.approx(0., abs=1e-12)
    assert float(source.sum()) == pytest.approx(
        -sum(boundaries[n]['flow'] for n in ('J1', 'J2')), abs=1e-6,
    )
    # A depth of 0.25 m over a bed at 5 m must still be able to drain.
    assert source[2] < 0


def _pipe_fdb(path: Path, primary_cells=(1, 2)):
    db = fdb.ORM.create()
    for i, (name, outfall) in enumerate([('J1', False), ('O1', True)]):
        node = Node()
        node.index, node.name, node.x, node.y, node.is_outfall = i, name, 5 + i * 10, 5, outfall
        db.push(node)
        primary = IndexLike()
        primary.index = primary_cells[i]
        db.push(primary, 'node_primary_ei')
        rim = F32Value()
        rim.value = 2. if not outfall else 0.
        db.push(rim, 'node_rim')
    for value in (0, 0, 0):
        row = IndexLike()
        row.index = value
        db.push(row, 'topo_ptr')
    count = IndexLike()
    count.index = 2
    db.push(count, 'node_count')
    # FDB requires a nonempty table even when CSR has no secondary entries.
    dummy = PipeTopo()
    dummy.ei = 0
    db.push(dummy)
    db.save(str(path))
    db.unlink()


def test_continuous_swmm_actor_keeps_warmup_clock(tmp_path):
    source_inp = tmp_path / 'source.inp'
    source_inp.write_text(SMALL_INP)
    cfg = PipeConfig(inp=str(source_inp), pipe_dir=str(tmp_path / 'runtime'),
                     coupling_interval=3, exchange_timeout=20)
    _pipe_fdb(Path(cfg.pipe_fdb))
    original = source_inp.read_bytes()
    start = datetime(2023, 9, 7, 8, tzinfo=timezone(timedelta(hours=8)))
    domain = SimpleNamespace(start=start, duration=6, warmup_duration=4)
    depth = np.array([0., .1, 0.], dtype=np.float32)
    bed = np.zeros(3, dtype=np.float32)
    area = np.array([0., 100., 100.], dtype=np.float32)
    types = np.array([0, 10, 10], dtype=np.uint8)
    with PipeCoupling(domain, cfg) as coupling:
        for elapsed, dt in [(-4, 4), (0, 3), (3, 3)]:
            rate = coupling.begin_window(elapsed, dt, depth, bed, area, types)
            assert rate[1] < 0
            coupling.end_window(rate.astype(np.float64) * dt)
        metadata = coupling.metadata()
    assert source_inp.read_bytes() == original
    assert metadata['swmm_cumulative_seconds'] == pytest.approx(10.)
    assert metadata['warmup_completed_seconds'] == 4
    assert metadata['official_completed_seconds'] == 6
    assert metadata['continuous_clock_at_zero']
    assert [w['swmm_after_s'] for w in metadata['windows']] == [4., 7., 10.]
    runtime_inp = Path(cfg.inp_runtime).read_text()
    assert 'START_TIME           07:59:56' in runtime_inp
    assert 'REPORT_START_TIME    07:59:56' in runtime_inp
    assert 'END_TIME             08:00:06' in runtime_inp


def test_swmm_actor_startup_error_reaches_caller(tmp_path):
    source_inp = tmp_path / 'invalid.inp'
    source_inp.write_text('[OPTIONS]\nFLOW_UNITS CMS\nSTART_DATE 09/07/2023\n')
    cfg = PipeConfig(inp=str(source_inp), pipe_dir=str(tmp_path / 'runtime'),
                     coupling_interval=5, exchange_timeout=5)
    _pipe_fdb(Path(cfg.pipe_fdb))
    start = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)
    domain = SimpleNamespace(start=start, duration=5, warmup_duration=0)
    with pytest.raises(RuntimeError, match=r"SWMM actor failed:.*") as caught:
        with PipeCoupling(domain, cfg):
            pass
    assert 'SWMM [OPTIONS] missing' in str(caught.value)


def test_subsecond_start_is_rejected_before_swmm_input_is_written(tmp_path):
    source_inp = tmp_path / 'source.inp'
    source_inp.write_text(SMALL_INP)
    cfg = PipeConfig(inp=str(source_inp), pipe_dir=str(tmp_path / 'runtime'))
    _pipe_fdb(Path(cfg.pipe_fdb))
    domain = SimpleNamespace(start=datetime(2023, 9, 7, 8, 0, 0, 123456,
                                           tzinfo=timezone.utc),
                             duration=5, warmup_duration=0)
    with pytest.raises(ValueError, match='whole second'):
        PipeCoupling(domain, cfg)
    assert not Path(cfg.inp_runtime).exists()


def test_pipe_input_requires_cubic_metres_per_second(tmp_path):
    from src.nh_flood_2d.core.coupled.pipe_1d import _prepare_inp
    source=tmp_path/'source.inp'
    source.write_text(SMALL_INP.replace('FLOW_UNITS CMS','FLOW_UNITS CFS'))
    with pytest.raises(ValueError,match='FLOW_UNITS must be CMS'):
        _prepare_inp(str(source),str(tmp_path/'runtime.inp'),
                     datetime(2023,9,7,8),datetime(2023,9,7,8,1))


class _FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def test_donor_limited_surface_volume_reaches_real_swmm(tmp_path, monkeypatch):
    import json
    import rasterio
    from rasterio.transform import from_origin
    from test_positivity import _grid, _cell, _vertical, _horizontal
    from src.nh_flood_2d.core.output import write_checkpoint
    from src.nh_flood_2d.core.run_record import sha256
    from src.nh_flood_2d.core.solver_compact import solver
    from src.nh_flood_2d.input import DomainConfig, ForceConfig
    from src.nh_flood_2d.preprocess import preprocess

    _grid(tmp_path)
    start = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)
    with rasterio.open(tmp_path / 'rain.tif', 'w', driver='GTiff', width=1, height=1,
                       count=1, dtype='float32', crs='EPSG:2326',
                       transform=from_origin(0, 30, 30, 30)) as raster:
        raster.write(np.zeros((1, 1), dtype='float32'), 1)
    (tmp_path / 'intervals.csv').write_text('interval_start,interval_end\n' +
        f'{start.isoformat()},{(start + timedelta(seconds=5)).isoformat()}\n')
    depth = np.zeros(10, dtype='float32')
    depth[_cell(1, 1)] = .1
    q = np.zeros(25, dtype='float32')
    for side, value in {_vertical(1, 1): -1, _vertical(1, 2): 1,
                        _horizontal(1, 1): -1, _horizontal(2, 1): 1}.items():
        q[side] = value
    zeros = np.zeros_like(depth)
    words = np.zeros(10, dtype='uint32')
    write_checkpoint(tmp_path / 'seed.fdb', depth=depth, bed=zeros, q=q,
        rain_time=zeros, infiltration=zeros, max_depth=depth,
        max_speed=zeros, first_time_hi=words, first_time_lo=words,
        clock_hi=0, clock_lo=0, gate_open=False,
        metadata_json=json.dumps(dict(run_state=False, gate_config='null',
            ne_sha256=sha256(tmp_path / 'ne.txt'), ns_sha256=sha256(tmp_path / 'ns.txt'))))
    cfg = DomainConfig(ne=str(tmp_path / 'ne.txt'), ns=str(tmp_path / 'ns.txt'),
        epsg_code=2326, domain_dir=str(tmp_path / 'out'), start=start,
        duration=5, warmup_duration=0, yield_step=5,
        land_parameters={10: dict(manning_n=1e-6, infiltration='none')},
        restart_state=str(tmp_path / 'seed.fdb'))
    force = ForceConfig(boundaries=[], rainfall=dict(raster=str(tmp_path / 'rain.tif'),
                                                    intervals=str(tmp_path / 'intervals.csv')))
    (tmp_path / 'source.inp').write_text(SMALL_INP)
    pipe = PipeConfig(inp=str(tmp_path / 'source.inp'), pipe_dir=str(tmp_path / 'pipe'),
                      coupling_interval=5, exchange_timeout=20)
    _pipe_fdb(Path(pipe.pipe_fdb), primary_cells=(_cell(1, 1), _cell(1, 2)))
    sent = []
    original_enter = PipeCoupling.__enter__

    class RecordingConnection:
        def __init__(self, connection):
            self.connection = connection

        def send(self, message):
            sent.append(message)
            self.connection.send(message)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    def enter(coupling):
        original_enter(coupling)
        coupling._conn = RecordingConnection(coupling._conn)
        return coupling

    monkeypatch.setattr(PipeCoupling, '__enter__', enter)
    preprocess(cfg, force)
    budget = solver(cfg, force, pipe)
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    window = manifest['coupling']['windows'][0]
    assert window['limited_drainage_return_m3'] > 0
    assert window['requested_surface_exchange_m3'] < window['applied_surface_exchange_m3'] < 0
    assert budget['pipe_exchange_m3'] == pytest.approx(window['applied_surface_exchange_m3'], abs=1e-6)
    payload = next(payload for command, payload in sent if command == 'advance')
    actual_node_volume = sum(v['flow'] for v in payload['boundaries'].values()) * payload['seconds']
    assert actual_node_volume == pytest.approx(-budget['pipe_exchange_m3'], abs=1e-6)
    assert manifest['coupling']['swmm_cumulative_seconds'] == 5
    assert abs(budget['residual_m3']) < 1e-5


def test_limited_drainage_is_returned_equally_to_unique_junctions():
    start = datetime(2023, 9, 7, 8, tzinfo=timezone.utc)
    coupling = object.__new__(PipeCoupling)
    coupling.domain_cfg = SimpleNamespace(start=start, duration=5)
    coupling.absolute_start = start
    coupling.absolute_end = start + timedelta(seconds=5)
    coupling.pipe_cfg = SimpleNamespace()
    coupling._elapsed_next = 0.
    coupling._conn = _FakeConn()
    coupling._process = object()
    coupling._mesh_validated = True
    coupling._pending = None
    coupling._pending_plan = None
    coupling._heads = {'J1': 0., 'J2': 0.}
    coupling._windows = []
    coupling._swmm_seconds = 0.
    coupling.node_names = ['J1', 'J2']
    coupling.is_outfall = np.array([False, False])
    coupling.primary_ei = np.array([1, 1])
    coupling.topo_ei = np.array([1, 1])  # duplicate CSR links stay unique
    coupling.topo_ptr = np.array([0, 1, 2])
    coupling.rims = np.array([2., 2.])
    coupling._receive = lambda expected: {'swmm_seconds': 5., 'heads': coupling._heads}
    depth = np.array([0., 0.25])
    bed = np.zeros(2)
    area = np.array([0., 100.])
    land_types = np.array([0, 10])
    rate = coupling.begin_window(0., 5., depth, bed, area, land_types)
    assert not coupling._conn.sent  # SWMM has not yet advanced
    requested_volume = float(rate[1]) * 5.
    actual = np.array([0., requested_volume * 0.4], dtype=np.float64)
    coupling.end_window(actual)
    command, payload = coupling._conn.sent[0]
    assert command == 'advance'
    assert payload['seconds'] == 5
    flows = [payload['boundaries'][name]['flow'] for name in ('J1', 'J2')]
    assert flows[0] == pytest.approx(flows[1], abs=1e-12)
    assert sum(flows) * 5 == pytest.approx(-actual.sum(), abs=1e-9)
    window = coupling.metadata()['windows'][0]
    assert window['limited_drainage_return_m3'] == pytest.approx(actual[1] - requested_volume)
    assert window['requested_surface_exchange_m3'] == pytest.approx(requested_volume)
    assert window['applied_surface_exchange_m3'] == pytest.approx(actual[1])
    assert abs(window['surface_swmm_balance_error_m3']) <= 1e-9
