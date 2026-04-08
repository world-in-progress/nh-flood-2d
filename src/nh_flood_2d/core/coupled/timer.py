"""Wall-clock timing statistics for the coupled solver."""

import time


class CouplingTimer:
    """Accumulates wall-clock durations for 2D steps, 1D steps, and exchange overhead."""

    def __init__(self):
        self.wall_start = time.perf_counter()
        self.exchange_count = 0
        self.total_2d_step = 0.0
        self.total_exchange_overhead = 0.0  # pure 2D-side exchange work (excl. 1D wait)
        self.total_1d_wait = 0.0            # wall time spent waiting for 1D response
        self.total_1d_step = 0.0            # 1D process self-reported step time

    def report(self, prefix: str = '[timer]') -> None:
        elapsed = time.perf_counter() - self.wall_start
        n = max(self.exchange_count, 1)
        print(f'{prefix} ────────────── Coupling Statistics ──────────────')
        print(f'{prefix}   Exchange windows completed : {self.exchange_count}')
        print(f'{prefix}   Total wall time            : {elapsed:.1f} s')
        print(f'{prefix}   2D step (avg)              : {self.total_2d_step / n:.3f} s')
        print(f'{prefix}   1D step (avg)              : {self.total_1d_step / n:.3f} s')
        print(f'{prefix}   1D wait (avg)              : {self.total_1d_wait / n:.3f} s')
        print(f'{prefix}   Exchange overhead (avg)     : {self.total_exchange_overhead / n:.3f} s')
        print(f'{prefix}   2D step total              : {self.total_2d_step:.1f} s')
        print(f'{prefix}   1D step total              : {self.total_1d_step:.1f} s')
        print(f'{prefix}   1D wait total              : {self.total_1d_wait:.1f} s')
        print(f'{prefix}   Exchange overhead total     : {self.total_exchange_overhead:.1f} s')
        print(f'{prefix} ────────────────────────────────────────────────')
