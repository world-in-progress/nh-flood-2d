"""Device-side, irreversible per-gate control and post-limiter accounting."""
import csv
from pathlib import Path
import numpy as np
import taichi as ti

from ..util.ti import copy_to_taichi
from .device_budget import DeviceBudget
from .clock import _seconds_to_ticks


def append_gate_balance(path, date, seconds, report):
    """Keep cumulative exchange processes at existing solver output times."""
    path = Path(path)
    fields = ['time', 'elapsed_seconds', 'region', 'pond_in_m3', 'pond_out_m3',
        'pond_net_m3', 'river_in_m3', 'river_out_m3', 'channel_storage_m3',
        'initial_storage_m3', 'storage_m3', 'overtopping_in_m3', 'overtopping_out_m3',
        'rain_m3', 'infiltration_m3', 'positivity_m3', 'pipe_m3', 'boundary_m3',
        'rounding_m3', 'residual_m3', 'river_net_m3']
    header = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if header:
            writer.writeheader()
        for gate in report['gates']:
            writer.writerow(dict(time=date.isoformat(), elapsed_seconds=seconds,
                region=gate['name'], **{key: gate[key] for key in fields[3:9]},
                **{key: value for key, value in gate['channel_budget'].items()
                   if key not in ('pond_net_m3',)}))
        writer.writerow(dict(time=date.isoformat(), elapsed_seconds=seconds,
            region='pond_control_volume', **report['pond_budget']))


@ti.data_oriented
class GateController:
    def __init__(self, config, layout, bed, land_types, clock, origin_seconds,
                 min_h, step_bound, side_count, working_bed, depth, area):
        self.config, self.layout, self.clock = config, layout, clock
        self.count = len(config.gates)
        self.bed, self.depth = working_bed, depth
        self.area = area
        self.min_h = min_h
        self.formal_tick = _seconds_to_ticks(origin_seconds)
        self.cell_gate = copy_to_taichi(layout.cell_gate, ti.i32, None)
        self.targets = copy_to_taichi(np.concatenate(layout.cells), ti.i32, None)
        self.control = copy_to_taichi(layout.controls, ti.i32, None)
        self.closed = copy_to_taichi(bed, ti.f32, None)
        self.pond = copy_to_taichi((land_types == config.pond_land_type).astype(np.int32), ti.i32, None)
        self.pond[0] = 0
        self.commanded = ti.field(ti.i32, self.count)
        self.command = ti.field(ti.u32, (self.count, 2))
        self.flow_seen = ti.field(ti.i32, self.count)
        self.first_flow = ti.field(ti.u32, (self.count, 2))
        self.opened = ti.field(ti.i32, self.count)
        self.opened_at = ti.field(ti.u32, (self.count, 2))
        # Four directed gate exchanges, then overtopping in/out and pond sources.
        self.offset = 4 * self.count
        self.channel_source_offset = self.offset + 8
        self.budget = DeviceBudget((max(len(bed), side_count) + 255) // 256,
                                   self.channel_source_offset + 6 * self.count, step_bound)
        self.initial_storage = None
        self.initial_channel_storage = None
        self.previous = ti.field(ti.f32, len(bed))
        self.inventory_delta = ti.field(ti.f32, len(bed))
        self.sources = ti.field(ti.f32, (len(bed), 4))

    @ti.func
    def since_command(self, gate, hi, lo):
        borrow = ti.cast(lo < self.command[gate, 1], ti.u32)
        high = hi - self.command[gate, 0] - borrow
        low = lo - self.command[gate, 1]
        return ti.cast(high, ti.f32) / 256. + ti.cast(low, ti.f32) * (2.**-40)

    @ti.func
    def move(self, bed, hi, lo):
        for slot in range(self.targets.shape[0]):
            cell = self.targets[slot]
            gate = self.cell_gate[cell]
            if self.clock.active[None] == 1 and self.commanded[gate] == 1:
                fraction = ti.min(self.since_command(gate, hi, lo) / self.config.opening_seconds, 1.)
                if fraction >= 1.:
                    bed[cell] = self.config.open_elevation
                    if ti.atomic_or(self.opened[gate], 1) == 0:
                        self.opened_at[gate, 0], self.opened_at[gate, 1] = hi, lo
                else:
                    bed[cell] = self.closed[cell] + (self.config.open_elevation - self.closed[cell]) * fraction

    @ti.func
    def check(self):
        for gate in range(self.count):
            if self.clock.active[None] == 1:
                hi, lo = self.clock.now[0], self.clock.now[1]
                if hi > self.formal_tick >> 32 or (hi == self.formal_tick >> 32 and lo >= (self.formal_tick & 0xffffffff)):
                    cell = self.control[gate]
                    if self.commanded[gate] == 0 and self.depth[cell] >= self.min_h:
                        if self.bed[cell] + self.depth[cell] > self.config.trigger_level:
                            self.commanded[gate] = 1
                            self.command[gate, 0], self.command[gate, 1] = hi, lo
        self.move(self.bed, self.clock.now[0], self.clock.now[1])

    @ti.func
    def move_to_end(self):
        self.move(self.bed, self.clock.step_end[0], self.clock.step_end[1])

    @ti.func
    def clear_step(self):
        for block, term in self.budget.step:
            if self.clock.active[None] == 1:
                self.budget.step[block, term] = 0.

    @ti.func
    def add(self, block, term, value):
        if value != 0.:
            ti.atomic_add(self.budget.step[block, term], value)

    @ti.func
    def directed(self, block, term, into):
        self.add(block, term, ti.max(into, 0.))
        self.add(block, term + 1, ti.max(-into, 0.))

    @ti.func
    def record_flux(self, side, left, right, volume):
        gl, gr = self.cell_gate[left], self.cell_gate[right]
        pl, pr = self.pond[left], self.pond[right]
        block = side // 256
        if gl >= 0 and gr < 0:
            if pr == 1:
                self.directed(block, 4 * gl, volume)
                if volume != 0.:
                    if ti.atomic_or(self.flow_seen[gl], 1) == 0:
                        self.first_flow[gl, 0], self.first_flow[gl, 1] = self.clock.step_end[0], self.clock.step_end[1]
            else:
                self.directed(block, 4 * gl + 2, -volume)
        elif gr >= 0 and gl < 0:
            if pl == 1:
                self.directed(block, 4 * gr, -volume)
                if volume != 0.:
                    if ti.atomic_or(self.flow_seen[gr], 1) == 0:
                        self.first_flow[gr, 0], self.first_flow[gr, 1] = self.clock.step_end[0], self.clock.step_end[1]
            else:
                self.directed(block, 4 * gr + 2, volume)
        elif pl != pr:
            self.directed(block, self.offset, ti.select(pr == 1, volume, -volume))

    @ti.func
    def source_offset(self, cell):
        offset = -1
        if self.pond[cell] == 1:
            offset = self.offset + 2
        elif self.cell_gate[cell] >= 0:
            offset = self.channel_source_offset + 6 * self.cell_gate[cell]
        return offset

    @ti.func
    def record_cell(self, cell, previous, current, area, rain_depth, removed,
                    floor, pipe_volume, discharge_volume, level_exchange, increment):
        offset = self.source_offset(cell)
        if offset >= 0:
            block = cell // 256
            self.add(block, offset, rain_depth * area)
            self.add(block, offset + 1, removed * area)
            self.add(block, offset + 2, floor * area)
            self.add(block, offset + 3, pipe_volume)
            self.add(block, offset + 4, discharge_volume + level_exchange)
            self.previous[cell] = previous
            self.sources[cell, 0], self.sources[cell, 1] = increment, removed
            self.sources[cell, 2], self.sources[cell, 3] = floor, level_exchange

    @ti.func
    def measure_inventory(self):
        # Separate offload: do not let fast-math replace the actually rounded
        # depth with its algebraic source expression and cancel the error.
        for cell in range(1, self.pond.shape[0]):
            if self.clock.active[None] == 1:
                if self.source_offset(cell) >= 0:
                    self.inventory_delta[cell] = self.depth[cell] - self.previous[cell]

    @ti.func
    def collect(self):
        for cell in range(1, self.pond.shape[0]):
            if self.clock.active[None] == 1:
                offset = self.source_offset(cell)
                if offset >= 0:
                    rounding = self.inventory_delta[cell] - self.sources[cell, 0] + self.sources[cell, 1] - self.sources[cell, 2]
                    self.add(cell // 256, offset + 5, rounding * self.area[cell] - self.sources[cell, 3])
        for block, term in self.budget.step:
            if self.clock.active[None] == 1:
                self.budget.accumulate(block, term)

    def snapshot(self):
        return dict(commanded=self.commanded.to_numpy().tolist(),
                    command_ticks=self.command.to_numpy().tolist(),
                    flow_seen=self.flow_seen.to_numpy().tolist(),
                    first_flow_ticks=self.first_flow.to_numpy().tolist(),
                    opened=self.opened.to_numpy().tolist(),
                    opened_ticks=self.opened_at.to_numpy().tolist(),
                    initial_storage_m3=self.initial_storage,
                    initial_channel_storage_m3=self.initial_channel_storage.tolist(),
                    budget_units=self.budget.total_units)

    def restore(self, saved):
        self.commanded.from_numpy(np.array(saved['commanded'], dtype=np.int32))
        self.command.from_numpy(np.array(saved['command_ticks'], dtype=np.uint32))
        self.flow_seen.from_numpy(np.array(saved['flow_seen'], dtype=np.int32))
        self.first_flow.from_numpy(np.array(saved['first_flow_ticks'], dtype=np.uint32))
        self.opened.from_numpy(np.array(saved['opened'], dtype=np.int32))
        self.opened_at.from_numpy(np.array(saved['opened_ticks'], dtype=np.uint32))
        self.initial_storage = saved['initial_storage_m3']
        self.initial_channel_storage = np.array(saved['initial_channel_storage_m3'], dtype=float)
        self.budget.total_units = list(saved['budget_units'])

    def initialize_inventory(self, depth, area):
        pond = self.pond.to_numpy().astype(bool)
        self.initial_storage = float(np.sum(depth[pond].astype(float)*area[pond]))
        self.initial_channel_storage = np.array([
            np.sum(depth[cells].astype(float)*area[cells]) for cells in self.layout.cells])

    def report(self, depth, area, clock_seconds):
        totals = self.budget.flush()
        commanded, ticks = self.commanded.to_numpy(), self.command.to_numpy()
        seen, flow = self.flow_seen.to_numpy(), self.first_flow.to_numpy()
        opened, opened_at = self.opened.to_numpy(), self.opened_at.to_numpy()
        def seconds(words):
            return ((int(words[0]) << 32) | int(words[1])) / 2**40 - self.formal_tick / 2**40
        gates = []
        closed_bed = self.closed.to_numpy()
        for i, gate in enumerate(self.config.gates):
            command = seconds(ticks[i]) if commanded[i] else None
            end = command + self.config.opening_seconds if command is not None else None
            cells = self.layout.cells[i]
            channel_storage = float(np.sum(depth[cells].astype(float)*area[cells]))
            sources = totals[self.channel_source_offset + 6*i:self.channel_source_offset + 6*i + 6]
            river_net = float(totals[4*i+2]-totals[4*i+3])
            pond_net = float(totals[4*i]-totals[4*i+1])
            channel_net = river_net - pond_net + sources[0]-sources[1]+sources[2]+sources[3]+sources[4]+sources[5]
            channel_budget = dict(initial_storage_m3=float(self.initial_channel_storage[i]),
                storage_m3=channel_storage, river_net_m3=river_net, pond_net_m3=pond_net,
                rain_m3=float(sources[0]), infiltration_m3=float(sources[1]),
                positivity_m3=float(sources[2]), pipe_m3=float(sources[3]),
                boundary_m3=float(sources[4]), rounding_m3=float(sources[5]),
                residual_m3=float(channel_storage-self.initial_channel_storage[i]-channel_net))
            gates.append(dict(name=gate.name, cell_ids=cells.tolist(),
                closed_elevation=closed_bed[cells].tolist(),
                command_clock_ticks=ticks[i].tolist() if commanded[i] else None,
                control_cell=int(self.layout.controls[i]), command_seconds=command,
                first_effective_flow_seconds=seconds(flow[i]) if seen[i] else None,
                fully_open_seconds=end if end is not None and clock_seconds >= end else None,
                first_fully_open_sample_seconds=seconds(opened_at[i]) if opened[i] else None,
                scheduled_fully_open_seconds=end,
                pond_in_m3=float(totals[4*i]), pond_out_m3=float(totals[4*i+1]),
                pond_net_m3=float(totals[4*i]-totals[4*i+1]),
                river_in_m3=float(totals[4*i+2]), river_out_m3=float(totals[4*i+3]),
                channel_storage_m3=channel_storage, channel_budget=channel_budget))
        t = totals[self.offset:self.offset+8]
        pond = self.pond.to_numpy().astype(bool)
        storage = float(np.sum(depth[pond].astype(float)*area[pond]))
        net = sum(g['pond_net_m3'] for g in gates) + t[0]-t[1]+t[2]-t[3]+t[4]+t[5]+t[6]+t[7]
        return dict(enabled=True, trigger_level=self.config.trigger_level,
            pond_land_type=self.config.pond_land_type,
            opening_seconds=self.config.opening_seconds, open_elevation=self.config.open_elevation,
            clock_origin_before_start_s=self.formal_tick / 2**40,
            clock_quantum_s=2.**-40, first_effective_flow_definition=
            'End of first completed step with nonzero post-limiter gate-to-pond subedge volume in either direction',
            gates=gates, pond_budget=dict(initial_storage_m3=self.initial_storage,
                storage_m3=storage, overtopping_in_m3=float(t[0]), overtopping_out_m3=float(t[1]),
                rain_m3=float(t[2]), infiltration_m3=float(t[3]), positivity_m3=float(t[4]),
                pipe_m3=float(t[5]), boundary_m3=float(t[6]), rounding_m3=float(t[7]),
                residual_m3=float(storage-self.initial_storage-net)))
