"""Exact accumulation of finite F32 block diagnostics between outputs."""

import math

import numpy as np
import taichi as ti


@ti.data_oriented
class DeviceBudget:
    """Store F32 values as signed integer multiples of 2**-149.

    Each (block, term) has one writer per step. The two sign planes avoid
    device-side subtraction and carry/borrow races; host reduction happens
    only when an output is requested.
    """

    def __init__(self, block_count: int, term_count: int, max_steps: int):
        self.word_count = (277 + max_steps.bit_length() + 15) // 16
        self.max_count = max_steps
        self.count = 0
        self.transfer_count = 0
        self.total_units = [0] * term_count
        self.step = ti.field(ti.f32, shape=(block_count, term_count))
        self.digits = ti.field(ti.i32, shape=(2, block_count, term_count, self.word_count))
        self.invalid = ti.field(ti.i32, shape=())

    @ti.func
    def accumulate(self, block: ti.i32, term: ti.i32):
        bits = ti.bit_cast(self.step[block, term], ti.u32)
        if (bits & 0x7fffffff) != 0:
            exponent = ti.cast((bits >> 23) & 255, ti.i32)
            if exponent == 255:
                ti.atomic_max(self.invalid[None], 1)
            else:
                sign = ti.cast(bits >> 31, ti.i32)
                mantissa = ti.cast(bits & 0x7fffff, ti.i32)
                if exponent != 0:
                    mantissa |= 1 << 23
                offset = ti.max(exponent - 1, 0)
                word, shift = offset // 16, offset % 16
                low = (mantissa & 65535) << shift
                high = (mantissa >> 16) << shift
                carry = 0
                for j in range(word, self.word_count):
                    source_digit = 0
                    if j == word:
                        source_digit = low & 65535
                    elif j == word + 1:
                        source_digit = (low >> 16) + (high & 65535)
                    elif j == word + 2:
                        source_digit = high >> 16
                    total = self.digits[sign, block, term, j] + source_digit + carry
                    self.digits[sign, block, term, j] = total & 65535
                    carry = total >> 16
                    if j >= word + 2 and carry == 0:
                        break

    @ti.kernel
    def collect(self):
        for block, term in self.step:
            self.accumulate(block, term)

    def step_finished(self):
        """Record one completed injection without reading the device."""
        self.count += 1

    def flush(self) -> np.ndarray:
        """Return cumulative F64 totals, then clear the interval digits."""
        invalid = self.invalid[None]
        if invalid == 1:
            raise FloatingPointError('Nonfinite F32 device budget diagnostic')
        word_sums = self.digits.to_numpy().sum(axis=1, dtype=np.int64)
        for term in range(len(self.total_units)):
            self.total_units[term] += sum(
                (int(word_sums[0, term, j]) - int(word_sums[1, term, j])) << (16 * j)
                for j in range(self.word_count)
            )
        totals = np.array([math.ldexp(float(units), -149) for units in self.total_units],
                          dtype=np.float64)
        self.digits.fill(0)
        self.count = 0
        self.transfer_count += 1
        return totals
