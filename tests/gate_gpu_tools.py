"""Opt-in Metal validation helpers; importing this module never opens a device."""
import ctypes
from datetime import datetime, time
import platform
import resource
import subprocess
from time import perf_counter
from zoneinfo import ZoneInfo


def gpu_window():
    now = datetime.now(ZoneInfo('Asia/Singapore'))
    result = subprocess.run(['pgrep', '-f', '[p]ython.*main[.]py'],
                            capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError('pgrep could not establish the solver-process gate')
    pids = [int(value) for value in result.stdout.split()]
    return dict(time=now.isoformat(), solver_pids=pids,
                allowed=not pids or now.timetz().replace(tzinfo=None) >= time(20, 30),
                policy='No python main.py solver PID, or local time at/after 20:30')


def require_gpu_window():
    state = gpu_window()
    if not state['allowed']:
        raise RuntimeError(f'Metal validation deferred: {state}')
    return state


class MetalMemory:
    """MTLDevice.currentAllocatedSize plus the process unified-memory high water.

    Apple SDK MTLDevice.h defines currentAllocatedSize as bytes of resources
    allocated by this device. This is a sampled allocation count, not GPU
    residency or system-wide VRAM. The M1 uses unified memory.
    """
    def __init__(self):
        require_gpu_window()
        if platform.system() != 'Darwin':
            raise RuntimeError('Metal measurements require macOS')
        self.metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        self.device = self.metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            raise RuntimeError('Metal returned no default device')
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.selector = self.objc.sel_registerName(b'currentAllocatedSize')
        self.objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.objc.objc_msgSend.restype = ctypes.c_size_t
        self.peak_allocated_bytes = 0
        self.samples = 0

    def sample(self):
        allocated = int(self.objc.objc_msgSend(self.device, self.selector))
        self.peak_allocated_bytes = max(allocated, self.peak_allocated_bytes)
        self.samples += 1
        return allocated

    def report(self):
        self.sample()
        return dict(metal_sampled_peak_allocated_bytes=self.peak_allocated_bytes,
            metal_allocation_samples=self.samples,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            memory_definition='MTLDevice.currentAllocatedSize: sampled device-resource allocation bytes; process RSS is unified-memory high water, not separate VRAM residency')


class GraphProfile:
    def __init__(self, ti, memory):
        self.ti, self.memory = ti, memory
        self.batches = []
        self.pending_seconds = None

    def install(self):
        from taichi.graph._graph import Graph
        from src.nh_flood_2d.core.clock import StepClock
        self.original_run, self.original_trace = Graph.run, StepClock.read_trace
        profiler = self
        def timed(graph, values):
            profiler.ti.sync()
            begin = perf_counter()
            result = profiler.original_run(graph, values)
            profiler.ti.sync()
            profiler.pending_seconds = perf_counter()-begin
            profiler.memory.sample()
            return result
        def capture(clock):
            rows = profiler.original_trace(clock)
            if profiler.pending_seconds is not None:
                profiler.batches.append(dict(seconds=profiler.pending_seconds,
                    steps=int((rows[:, 3] == clock.COMPLETED).sum())))
                profiler.pending_seconds = None
            return rows
        Graph.run, StepClock.read_trace = timed, capture

    def uninstall(self):
        from taichi.graph._graph import Graph
        from src.nh_flood_2d.core.clock import StepClock
        Graph.run, StepClock.read_trace = self.original_run, self.original_trace

    def report(self):
        steady = self.batches[2:] if len(self.batches) > 2 else self.batches
        steps = sum(batch['steps'] for batch in steady)
        seconds = sum(batch['seconds'] for batch in steady)
        return dict(graph_batches=self.batches, steady_steps=steps,
            steady_graph_seconds=seconds,
            steady_graph_seconds_per_step=seconds/steps if steps else None,
            timing_definition='Synchronized device graph wall time, excluding first two batches; includes graph guards for unused batch slots; excludes preprocessing, compilation and host output work')
