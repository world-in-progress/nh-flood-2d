"""CPU checks for exact F32 diagnostic accumulation on the device."""

from fractions import Fraction
import math

import numpy as np
import pytest
import taichi as ti

from src.nh_flood_2d.core.device_budget import DeviceBudget



pytestmark = pytest.mark.usefixtures('taichi_cpu')

UNIT = 1 << 149


def exact_units(value):
    return int(Fraction.from_float(float(value)) * UNIT)


def exact_units_from_bits(bits):
    """Decode IEEE F32 as a rational, avoiding host F32 denormal flushing."""
    bits = int(bits)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7fffff
    sign = -1 if bits >> 31 else 1
    significand = Fraction(fraction + ((1 << 23) if exponent else 0), 1 << 23)
    power = exponent - 127 if exponent else -126
    return int(sign * significand * Fraction(2) ** power * UNIT)


def inject(budget, values):
    budget.step.from_numpy(np.asarray(values, dtype=np.float32))
    budget.collect()
    budget.step_finished()


def test_cancellation_across_blocks_and_outputs_keeps_small_values():
    budget = DeviceBudget(2, 2, 2)
    large = np.float32(1e30)
    inject(budget, [[2**25, large], [1, 1]])
    first = budget.flush()
    assert budget.total_units == [exact_units(np.float32(2**25)) + UNIT,
                                  exact_units(large) + UNIT]
    assert first[0] == 2**25 + 1
    assert first[1] == float(large)

    inject(budget, [[-2**25, -large], [0, 0]])
    np.testing.assert_array_equal(budget.flush(), [1., 1.])
    assert budget.total_units == [UNIT, UNIT]
    assert budget.transfer_count == 2


def test_signed_subnormals_zero_and_normal_boundary():
    budget = DeviceBudget(2, 5, 1)
    bits = np.array([[0x00000001, 0x80000001, 0x007fffff, 0x80000000, 0x00800000],
                     [0x80000001, 0x00000001, 0x80800000, 0x00000000, 0x007fffff]],
                    dtype=np.uint32)
    inject(budget, bits.view(np.float32))
    result = budget.flush()
    assert budget.total_units == [0, 0, -1, 0, (1 << 24) - 1]
    np.testing.assert_array_equal(result, [0., 0., -math.ldexp(1., -149), 0.,
                                           math.ldexp(float((1 << 24) - 1), -149)])
    assert not np.signbit(result[3])


def test_rounds_the_complete_integer_once():
    budget = DeviceBudget(3, 1, 1)
    inject(budget, [[2**27], [2**-26], [2**-126]])
    result = budget.flush()
    assert result[0].hex() == '0x1.0000000000001p+27'
    assert budget.total_units[0] == sum(map(exact_units,
                                             np.array([2**27, 2**-26, 2**-126],
                                                      dtype=np.float32)))


def test_random_finite_patterns_match_fraction_reference_across_flushes():
    rng = np.random.default_rng(48193)
    budget = DeviceBudget(7, 4, 40)
    reference = [0] * 4
    for step in range(40):
        bits = rng.integers(0, 1 << 32, size=(7, 4), dtype=np.uint32)
        nonfinite = ((bits >> 23) & 255) == 255
        bits[nonfinite] ^= np.uint32(1 << 23)
        values = bits.view(np.float32)
        inject(budget, values)
        for term in range(4):
            reference[term] += sum(exact_units_from_bits(value) for value in bits[:, term])
        if step in (16, 39):
            result = budget.flush()
            assert budget.total_units == reference
            np.testing.assert_array_equal(result,
                [math.ldexp(float(units), -149) for units in reference])
    assert budget.transfer_count == 2


@pytest.mark.parametrize('power', [0, 1, 2, 4, 5])
def test_power_of_two_step_bounds_and_interval_reset(power):
    steps = 1 << power
    budget = DeviceBudget(1, 1, steps)
    assert budget.word_count == (277 + steps.bit_length() + 15) // 16
    assert budget.max_count == steps
    assert budget.step.dtype == ti.f32
    assert budget.digits.dtype == ti.i32
    assert budget.invalid.dtype == ti.i32
    assert budget.digits.shape == (2, 1, 1, budget.word_count)
    for _ in range(steps):
        inject(budget, [[1.]])
    assert budget.count == steps
    np.testing.assert_array_equal(budget.flush(), [float(steps)])
    assert budget.count == 0
    assert budget.transfer_count == 1
    assert not np.any(budget.digits.to_numpy())

    # The latest step stays available for the caller until the next tick clears it.
    np.testing.assert_array_equal(budget.step.to_numpy(), [[1.]])
    budget.step.fill(0)
    budget.collect()
    budget.step_finished()
    np.testing.assert_array_equal(budget.flush(), [float(steps)])
    assert budget.transfer_count == 2


def test_initial_collect_is_included_before_the_first_output():
    budget = DeviceBudget(1, 1, 1)
    inject(budget, [[3.5]])
    np.testing.assert_array_equal(budget.flush(), [3.5])
    assert budget.total_units == [exact_units(np.float32(3.5))]


def test_accumulate_can_run_inside_the_producing_kernel():
    budget = DeviceBudget(2, 2, 1)

    @ti.kernel
    def produce_and_accumulate():
        for block, term in budget.step:
            budget.step[block, term] = ti.cast(block + term + 1, ti.f32)
            budget.accumulate(block, term)

    produce_and_accumulate()
    budget.step_finished()
    np.testing.assert_array_equal(budget.flush(), [3., 5.])


def test_maximum_finite_f32_carries_into_high_words():
    budget = DeviceBudget(1, 1, 32)
    bits = np.array([[0x7f7fffff]], dtype=np.uint32)
    for _ in range(32):
        inject(budget, bits.view(np.float32))
    budget.flush()
    assert budget.total_units == [32 * exact_units_from_bits(bits[0, 0])]
    assert budget.invalid[None] == 0


@pytest.mark.parametrize('bits', [0x7f800000, 0xff800000, 0x7fc00001])
def test_nonfinite_value_fails_without_clearing_accumulator(bits):
    budget = DeviceBudget(1, 1, 1)
    value = np.array([[bits]], dtype=np.uint32).view(np.float32)
    inject(budget, value)
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        budget.flush()
    assert budget.invalid[None] == 1
    assert not np.any(budget.digits.to_numpy())
    assert budget.transfer_count == 0
    assert budget.count == 1
