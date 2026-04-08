"""Wall-clock timing statistics for the coupled solver."""

import time


class CouplingTimer:
    """Accumulates wall-clock durations for 2D steps, 1D steps, and exchange overhead."""

    def __init__(self):
        self.wall_start = time.perf_counter()
        self.exchange_count = 0
        self.total_2d_step = 0.0
        self.total_exchange = 0.0
        self.total_1d_step = 0.0

    def report(self, prefix: str = '[timer]') -> None:
        elapsed = time.perf_counter() - self.wall_start
        n = max(self.exchange_count, 1)
        print(f'{prefix} ────────────── Coupling Statistics ──────────────')
        print(f'{prefix}   Exchange windows completed : {self.exchange_count}')
        print(f'{prefix}   Total wall time            : {elapsed:.1f} s')
        print(f'{prefix}   2D per exchange-step (avg)  : {self.total_2d_step / n:.3f} s')
        print(f'{prefix}   1D per exchange-step (avg)  : {self.total_1d_step / n:.3f} s')
        print(f'{prefix}   Exchange overhead (avg)     : {self.total_exchange / n:.3f} s')
        print(f'{prefix}   2D total                    : {self.total_2d_step:.1f} s')
        print(f'{prefix}   1D total                    : {self.total_1d_step:.1f} s')
        print(f'{prefix}   Exchange total              : {self.total_exchange:.1f} s')
        print(f'{prefix} ────────────────────────────────────────────────')
