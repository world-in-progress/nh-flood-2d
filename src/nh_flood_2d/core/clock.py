"""Metal-compatible two-U32 fixed-point step clock (2**-40 seconds/tick)."""

import math

import numpy as np
import taichi as ti


_TICKS_PER_SECOND = 1 << 40
_MAX_TICKS = (1 << 64) - 1
_SNAP_TICKS_EXCLUSIVE = math.ceil(1e-6 * _TICKS_PER_SECOND)


def _seconds_to_ticks(seconds: float) -> int:
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Clock seconds must be finite and nonnegative')
    numerator, denominator = seconds.as_integer_ratio()
    scaled, remainder = divmod(numerator * _TICKS_PER_SECOND, denominator)
    twice = remainder * 2
    ticks = scaled + int(twice > denominator or (twice == denominator and scaled & 1))
    if ticks > _MAX_TICKS:
        raise ValueError('Clock exceeds the two-U32 range')
    return ticks


def _words(ticks: int) -> np.ndarray:
    return np.array([ticks >> 32, ticks & 0xffffffff], dtype=np.uint32)


@ti.data_oriented
class StepClock:
    """Prepare one F32 physics step and commit it only after all update stages.

    Trace columns: F32 dt bit pattern, ending high/low tick words, status,
    CFL bits, event-step flag, limited cell/face counts, minimum donor-scale bits.
    Status 0 means no step, 1 completed, 2 invalid/too-small CFL,
    3 unrepresentable F32 dt, 4 clock overshoot beyond the snap tolerance.
    All scalar fields are device state; read_trace() is the one batch readback.
    """

    DONE = 0
    COMPLETED = 1
    BAD_CFL = 2
    BAD_DT = 3
    OVERSHOOT = 4

    def __init__(self, initial_seconds: float, batch_size: int = 32,
                 min_dt: float = 1e-4):
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError('batch_size must be between 1 and 64')
        if not math.isfinite(min_dt) or not 0 < min_dt <= 1:
            raise ValueError('min_dt must be finite and in (0, 1]')
        self.batch_size = batch_size
        self.min_dt = float(np.float32(min_dt))
        if self.min_dt <= 0:
            raise ValueError('min_dt rounds to zero in F32')
        self._host_now_ticks = _seconds_to_ticks(initial_seconds)
        self._batch_pending = False
        self.now = ti.field(ti.u32, shape=2)
        self.next = ti.field(ti.u32, shape=2)
        self.origin = ti.field(ti.u32, shape=2)
        self.step_end = ti.field(ti.u32, shape=2)
        self.dt = ti.field(ti.f32, shape=())
        self.active = ti.field(ti.i32, shape=())
        self.local = ti.field(ti.f32, shape=())
        self.end_local = ti.field(ti.f32, shape=())
        self.trace = ti.field(ti.u32, shape=(batch_size, 9))
        self.now.from_numpy(_words(self._host_now_ticks))

    @property
    def host_now_seconds(self) -> float:
        return self._host_now_ticks / _TICKS_PER_SECOND

    def begin(self, target_seconds: float, batch_origin_seconds: float) -> None:
        """Start a batch at the last read-back clock; upload its next event.

        batch_origin_seconds checks the caller's timeline. The device origin is
        copied from the exact last end tick, avoiding F64 round-trip loss.
        """
        if self._batch_pending:
            raise RuntimeError('Read the previous batch trace before begin')
        if not math.isfinite(batch_origin_seconds) or abs(
                batch_origin_seconds - self.host_now_seconds) >= 1e-6:
            raise ValueError('batch_origin_seconds differs from the last end tick')
        target_ticks = _seconds_to_ticks(target_seconds)
        if target_ticks < self._host_now_ticks:
            raise ValueError('Target precedes the current clock')
        self.next.from_numpy(_words(target_ticks))
        self.origin.from_numpy(_words(self._host_now_ticks))
        self.trace.fill(0)
        self._batch_pending = True

    @ti.func
    def choose(self, candidate: ti.f32, slot: ti.i32):
        """Choose dt from the previous CFL; do not run physics if active=0."""
        self.active[None] = 0
        self.dt[None] = 0.0
        self.local[None] = 0.0
        self.end_local[None] = 0.0
        now_hi, now_lo = self.now[0], self.now[1]
        target_hi, target_lo = self.next[0], self.next[1]
        if now_hi < target_hi or (now_hi == target_hi and now_lo < target_lo):
            if not (candidate >= self.min_dt):
                self.trace[slot, 3] = self.BAD_CFL
            else:
                borrow = ti.cast(now_lo > target_lo, ti.u32)
                rem_hi = target_hi - now_hi - borrow
                rem_lo = target_lo - now_lo
                remaining = ti.cast(rem_hi, ti.f32) * (1.0 / 256.0) + \
                    ti.cast(rem_lo, ti.f32) * (1.0 / _TICKS_PER_SECOND)
                chosen = ti.min(ti.min(candidate, 1.0), remaining)
                self.trace[slot, 4] = ti.bit_cast(candidate, ti.u32)
                self.trace[slot, 5] = ti.cast(chosen < ti.min(candidate, 1.), ti.u32)
                bits = ti.bit_cast(chosen, ti.u32)
                exponent = ti.cast((bits >> 23) & 255, ti.i32) - 127
                mantissa = (bits & 0x7fffff) | 0x800000
                dt_hi, dt_lo = ti.u32(0), ti.u32(0)
                if not (chosen > 0.0) or exponent < -40 or exponent > 0:
                    self.trace[slot, 3] = self.BAD_DT
                else:
                    shift = exponent + 17
                    if shift >= 0:
                        dt_lo = mantissa << shift
                        if shift >= 9:
                            dt_hi = mantissa >> (32 - shift)
                    else:
                        right = -shift
                        dt_lo = mantissa >> right
                        dropped = mantissa & ((ti.u32(1) << right) - 1)
                        half = ti.u32(1) << (right - 1)
                        if dropped > half or (dropped == half and (dt_lo & 1) != 0):
                            dt_lo += 1
                    if dt_hi == 0 and dt_lo == 0:
                        self.trace[slot, 3] = self.BAD_DT
                    else:
                        dt_exceeds = dt_hi > rem_hi or (dt_hi == rem_hi and dt_lo > rem_lo)
                        diff_hi, diff_lo = ti.u32(0), ti.u32(0)
                        if dt_exceeds:
                            diff_hi = dt_hi - rem_hi - ti.cast(dt_lo < rem_lo, ti.u32)
                            diff_lo = dt_lo - rem_lo
                        else:
                            diff_hi = rem_hi - dt_hi - ti.cast(rem_lo < dt_lo, ti.u32)
                            diff_lo = rem_lo - dt_lo
                        end_hi, end_lo = now_hi, now_lo
                        valid_end = 1
                        if diff_hi == 0 and diff_lo < _SNAP_TICKS_EXCLUSIVE:
                            end_hi, end_lo = target_hi, target_lo
                        elif dt_exceeds:
                            self.trace[slot, 3] = self.OVERSHOOT
                            valid_end = 0
                        else:
                            end_lo = now_lo + dt_lo
                            end_hi = now_hi + dt_hi + ti.cast(end_lo < now_lo, ti.u32)
                        if valid_end == 1:
                            origin_hi, origin_lo = self.origin[0], self.origin[1]
                            local_hi = now_hi - origin_hi - ti.cast(now_lo < origin_lo, ti.u32)
                            local_lo = now_lo - origin_lo
                            end_local_hi = end_hi - origin_hi - ti.cast(end_lo < origin_lo, ti.u32)
                            end_local_lo = end_lo - origin_lo
                            self.local[None] = ti.cast(local_hi, ti.f32) * (1.0 / 256.0) + \
                                ti.cast(local_lo, ti.f32) * (1.0 / _TICKS_PER_SECOND)
                            self.end_local[None] = ti.cast(end_local_hi, ti.f32) * (1.0 / 256.0) + \
                                ti.cast(end_local_lo, ti.f32) * (1.0 / _TICKS_PER_SECOND)
                            self.dt[None] = chosen
                            self.step_end[0], self.step_end[1] = end_hi, end_lo
                            self.active[None] = 1

    @ti.kernel
    def prepare(self, ndt: ti.template(), slot: ti.i32):
        self.choose(ndt[None], slot)

    @ti.func
    def commit(self, slot: ti.i32):
        """Call after physical update and budget collection for this slot."""
        if self.active[None] == 1:
            self.now[0], self.now[1] = self.step_end[0], self.step_end[1]
            self.trace[slot, 0] = ti.bit_cast(self.dt[None], ti.u32)
            self.trace[slot, 1] = self.step_end[0]
            self.trace[slot, 2] = self.step_end[1]
            self.trace[slot, 3] = self.COMPLETED

    @ti.kernel
    def finish(self, slot: ti.i32):
        self.commit(slot)

    def read_trace(self) -> np.ndarray:
        """Download one trace and update the exact host mirror for next begin."""
        if not self._batch_pending:
            raise RuntimeError('Call begin before reading a trace')
        rows = self.trace.to_numpy()
        for row in rows:
            status = int(row[3])
            if status == self.COMPLETED:
                self._host_now_ticks = (int(row[1]) << 32) | int(row[2])
            elif status in (self.BAD_CFL, self.BAD_DT, self.OVERSHOOT):
                break
            elif status == self.DONE:
                break
            else:
                raise RuntimeError(f'Unknown clock trace status {status}')
        self._batch_pending = False
        return rows
