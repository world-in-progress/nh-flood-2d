"""FDB readers and solver state must outlive released native backing safely."""
import fastdb4py as fdb
import numpy as np
import pytest

from src.nh_flood_2d.core import output


def _poison_on_close(monkeypatch):
    load = fdb.RecordEngine.load
    records = []

    def tracked_load(*args, **kwargs):
        db = load(*args, **kwargs)
        table, close = db.table, db.close
        record = {'path': str(args[0]), 'closed': False, 'views': []}
        records.append(record)

        def tracked_table(feature, *args, **kwargs):
            assert not record['closed'], 'Table accessed after engine close'
            result = table(feature, *args, **kwargs)
            for name in feature.__annotations__:
                column = getattr(result.column, name)
                if isinstance(column, np.ndarray):
                    record['views'].append(column)
            return result

        def tracked_close():
            # Corrupt backing before releasing it so a retained view cannot
            # masquerade as a detached copy through its ownership reference.
            for view in record['views']:
                view.fill(0)
            record['closed'] = True
            close()

        db.table, db.close = tracked_table, tracked_close
        return db

    monkeypatch.setattr(fdb.RecordEngine, 'load', tracked_load)
    return records


def test_output_readers_detach_before_close(tmp_path, monkeypatch):
    values = np.array([0, .25, .5], dtype=np.float32)
    words = np.array([0, 1, 4294967295], dtype=np.uint32)
    output.write_bed(tmp_path / 'bed.fdb', values)
    output.write_depth_frame(tmp_path / 'depth.fdb', values)
    output.write_extrema(tmp_path / 'extrema.fdb', values, words, words, values)
    output.write_checkpoint(tmp_path / 'state.fdb', depth=values, bed=values, q=values,
        rain_time=values, infiltration=values, max_depth=values, max_speed=values,
        first_time_hi=words, first_time_lo=words, clock_hi=1, clock_lo=2,
        gate_open=True, metadata_json='{"owned": true}')
    records = _poison_on_close(monkeypatch)

    np.testing.assert_array_equal(output.read_bed(tmp_path / 'bed.fdb'), values)
    np.testing.assert_array_equal(output.read_depth_frame(tmp_path / 'depth.fdb'), values)
    extrema = output.read_extrema(tmp_path / 'extrema.fdb')
    np.testing.assert_array_equal(extrema.max_depth, values)
    np.testing.assert_array_equal(extrema.first_time_lo, words)
    state = output.read_checkpoint(tmp_path / 'state.fdb')
    for name in output._CHECKPOINT_FLOATS:
        np.testing.assert_array_equal(getattr(state, name), values)
    np.testing.assert_array_equal(state.first_time_hi, words)
    assert (state.clock_hi, state.clock_lo, state.gate_open) == (1, 2, True)
    assert state.metadata_json == '{"owned": true}'
    assert len(records) == 4 and all(record['closed'] for record in records)


def test_station_lookup_closes_first_engine_if_second_load_fails(tmp_path, monkeypatch):
    from test_output_io import _two_cell_mesh
    ne, ns = tmp_path / 'ne.fdb', tmp_path / 'ns.fdb'
    _two_cell_mesh(ne, ns)
    records = _poison_on_close(monkeypatch)
    with pytest.raises(FileNotFoundError):
        output.resolve_station_cells({}, ne, tmp_path / 'missing.fdb')
    assert len(records) == 1 and records[0]['closed']


def test_writer_closes_engine_when_save_fails(tmp_path, monkeypatch):
    close = fdb.RecordEngine.close
    closed = []

    def fail_save(*args):
        raise OSError('injected save failure')

    def tracked_close(db):
        closed.append(db)
        close(db)

    monkeypatch.setattr(fdb.RecordEngine, 'save', fail_save)
    monkeypatch.setattr(fdb.RecordEngine, 'close', tracked_close)
    with pytest.raises(OSError, match='injected save failure'):
        output.write_depth_frame(tmp_path / 'depth.fdb', [0, .5])
    assert len(closed) == 1


@pytest.mark.usefixtures('taichi_cpu')
def test_solver_uses_owned_mesh_after_engine_close(tmp_path, monkeypatch):
    from test_restart import _inputs, _run
    inputs = tmp_path / 'inputs'
    _inputs(inputs)
    _, expected, expected_manifest = _run(inputs, tmp_path / 'expected', 10)
    records = _poison_on_close(monkeypatch)
    _, actual, actual_manifest = _run(inputs, tmp_path / 'actual', 10)
    for name, value in vars(expected).items():
        actual_value = getattr(actual, name)
        if isinstance(value, np.ndarray):
            assert actual_value.dtype == value.dtype
            assert actual_value.tobytes() == value.tobytes(), name
        else:
            assert actual_value == value, name
    assert actual_manifest['dt_sequence_sha256'] == expected_manifest['dt_sequence_sha256']
    assert records and all(record['closed'] for record in records)
