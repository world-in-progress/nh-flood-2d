"""StepClock arithmetic and batch protocol checks on the CPU backend."""

from fractions import Fraction

import numpy as np
import pytest
import taichi as ti

from src.nh_flood_2d.core.clock import StepClock


pytestmark = pytest.mark.usefixtures('taichi_cpu')
SCALE = 1 << 40


def end_ticks(row):
    return (int(row[1]) << 32) | int(row[2])


def dt_from_row(row):
    return np.array([row[0]], dtype=np.uint32).view(np.float32)[0]


def candidate_field(value):
    field = ti.field(ti.f32, shape=())
    field[None] = value
    return field


def test_regular_f32_dt_is_encoded_exactly_and_traced():
    clock = StepClock(0., batch_size=2)
    ndt = candidate_field(0.2)
    clock.begin(1., 0.)
    clock.prepare(ndt, 0)
    assert clock.active[None] == 1
    assert float(clock.local[None]) == 0.
    assert abs(float(clock.end_local[None]) - float(np.float32(0.2))) < 1e-8
    clock.finish(0)
    rows = clock.read_trace()
    assert rows[0, 3] == clock.COMPLETED
    assert rows[1, 3] == clock.DONE
    expected = Fraction.from_float(float(np.float32(0.2))) * SCALE
    assert expected.denominator == 1
    assert end_ticks(rows[0]) == expected.numerator
    assert dt_from_row(rows[0]) == np.float32(0.2)
    assert clock.host_now_seconds == float(np.float32(0.2))


def test_short_event_below_min_dt_uses_a_positive_physical_step():
    initial = 169199.9999999
    clock = StepClock(initial, batch_size=2)
    ndt = candidate_field(0.25)
    clock.begin(169200., initial)
    clock.prepare(ndt, 0)
    assert clock.active[None] == 1
    assert 0. < float(clock.dt[None]) < 1e-4
    clock.finish(0)
    clock.prepare(ndt, 1)
    assert clock.active[None] == 0
    clock.finish(1)
    rows = clock.read_trace()
    assert rows[0, 3] == clock.COMPLETED
    assert rows[1, 3] == clock.DONE
    assert clock.host_now_seconds == 169200.


def test_two_word_carry_and_exact_batch_origin():
    initial = 1 / 256 - 0.0002
    clock = StepClock(initial, batch_size=1)
    ndt = candidate_field(0.0005)
    clock.begin(initial + 0.0005, initial)
    clock.prepare(ndt, 0)
    clock.finish(0)
    first = clock.read_trace()[0]
    assert first[3] == clock.COMPLETED
    assert first[1] == 1  # 2**32 ticks = 1/256 second
    assert first[2] > 0
    # A rounded host F64 second can differ by several ticks at long times;
    # begin uses the exact end ticks from the previous trace as its origin.
    clock.begin(clock.host_now_seconds + 0.001, clock.host_now_seconds)
    assert np.array_equal(clock.origin.to_numpy(), np.array(first[1:3], dtype=np.uint32))
    clock.read_trace()


@pytest.mark.parametrize('initial, expected_snap', [
    (169199.7999991, True),
    (169199.7999989, False),
])
def test_snap_threshold_is_checked_without_a_zero_length_update(initial, expected_snap):
    clock = StepClock(initial, batch_size=1)
    ndt = candidate_field(0.2)
    clock.begin(169200., initial)
    clock.prepare(ndt, 0)
    clock.finish(0)
    row = clock.read_trace()[0]
    assert row[3] == clock.COMPLETED
    assert (end_ticks(row) == 169200 * SCALE) is expected_snap


def test_bad_cfl_stops_batch_before_physics_or_clock_advance():
    clock = StepClock(0., batch_size=2)
    ndt = candidate_field(5e-5)
    clock.begin(1., 0.)
    clock.prepare(ndt, 0)
    assert clock.active[None] == 0
    clock.finish(0)
    row = clock.read_trace()[0]
    assert row[3] == clock.BAD_CFL
    assert clock.host_now_seconds == 0.


def test_target_already_reached_produces_only_unused_trace_slots():
    clock = StepClock(47 * 3600., batch_size=2)
    ndt = candidate_field(0.25)
    clock.begin(47 * 3600., 47 * 3600.)
    for slot in range(2):
        clock.prepare(ndt, slot)
        assert clock.active[None] == 0
        clock.finish(slot)
    assert not clock.read_trace().any()
    assert clock.host_now_seconds == 47 * 3600.


def test_47h_end_step_preserves_actual_dt_and_exact_event_tick():
    initial = 47 * 3600. - 0.25
    clock = StepClock(initial, batch_size=1)
    ndt = candidate_field(0.25)
    clock.begin(47 * 3600., initial)
    clock.prepare(ndt, 0)
    clock.finish(0)
    row = clock.read_trace()[0]
    assert dt_from_row(row) == np.float32(0.25)
    assert end_ticks(row) == 47 * 3600 * SCALE
    assert clock.host_now_seconds == 47 * 3600.


def test_host_rejects_out_of_range_clock_and_stale_batch_origin():
    with pytest.raises(ValueError):
        StepClock(-1.)
    with pytest.raises(ValueError):
        StepClock(194.181 * 86400.)
    clock = StepClock(1.)
    with pytest.raises(ValueError):
        clock.begin(2., 0.)
    clock.begin(2., 1.)
    with pytest.raises(RuntimeError):
        clock.begin(2., 1.)
    clock.read_trace()


def test_sub_microsecond_candidate_rounds_to_nearest_even_tick():
    # A sub-2**-17 s F32 step has fractional 2**-40 s ticks.
    small = np.nextafter(np.float32(2**-20), np.float32(1.0))
    exact = Fraction.from_float(float(small)) * SCALE
    assert exact.denominator > 1
    clock = StepClock(0., batch_size=1, min_dt=float(small))
    ndt = candidate_field(float(small))
    clock.begin(1., 0.)
    clock.prepare(ndt, 0)
    clock.finish(0)
    row = clock.read_trace()[0]
    assert row[3] == clock.COMPLETED
    assert end_ticks(row) == round(exact)
    assert abs(end_ticks(row) / SCALE - float(small)) <= 0.5 / SCALE
