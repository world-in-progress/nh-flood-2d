"""Explicitly opted-in Metal checks; default test collection opens no GPU."""
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

from src.nh_flood_2d.core.clock import StepClock, _seconds_to_ticks, _words
from src.nh_flood_2d.core.gate import GateController
from src.nh_flood_2d.core.run_record import source_tree_sha256
from src.nh_flood_2d.input.force import GateConfig
from src.nh_flood_2d.preprocess.gate import GateLayout
from src.nh_flood_2d.util import ti as runtime


@pytest.fixture
def metal_runtime():
    if os.environ.get('NH_GATE_METAL_TESTS') != '1':
        pytest.skip('Set NH_GATE_METAL_TESTS=1 after the solver-process/time gate')
    from gate_gpu_tools import require_gpu_window
    window = require_gpu_window()
    ti.reset()
    runtime._TI_INITIALIZED = False
    runtime.init_taichi(arch=ti.metal)
    yield window
    ti.reset()
    runtime._TI_INITIALIZED = False


@pytest.mark.parametrize('duration', [1800,3600])
def test_metal_late_command_entire_opening_and_fractional_ages(tmp_path, metal_runtime, duration):
    command = 30000.
    origin = 0.
    closed, opened = np.float32(10.), np.float32(1.3)
    clock = StepClock(command-1)
    bed_np = np.array([0,closed,0,0],dtype=np.float32)
    bed,depth,area = (ti.field(ti.f32,4) for _ in range(3))
    bed.from_numpy(bed_np)
    depth.from_numpy(np.zeros(4,dtype=np.float32))
    area.from_numpy(np.array([0,1,1,1],dtype=np.float32))
    config = GateConfig(land_type=11,pond_land_type=10,enabled=True,
        trigger_level=1.,opening_seconds=duration,open_elevation=float(opened),
        gates=[dict(name='A',inside_point=(1,0),control_point=(2,0))])
    layout = GateLayout(np.array([-1,0,-1,-1],dtype=np.int32),
        np.array([2],dtype=np.int32),[np.array([1],dtype=np.int32)])
    controller = GateController(config,layout,bed_np,np.array([0,11,1,10]),
        clock,origin,.02,4,4,bed,depth,area)
    controller.initialize_inventory(np.zeros(4,dtype=np.float32),np.array([0,1,1,1]))
    # Every integral second of the full travel, plus a noninteger age.
    times = np.unique(np.r_[command-1,command+np.arange(duration+1),command+1.234567])
    sample_count = len(times)
    words_np = np.array([_words(_seconds_to_ticks(t)) for t in times],dtype=np.uint32)
    ticks = ti.field(ti.u32,words_np.shape)
    ticks.from_numpy(words_np)
    observations = ti.field(ti.f32,sample_count)
    commands = ti.field(ti.i32,sample_count)
    @ti.kernel
    def traverse():
        ti.loop_config(serialize=True)
        for index in range(sample_count):
            clock.now[0],clock.now[1] = ticks[index,0],ticks[index,1]
            clock.active[None] = 1
            depth[2] = ti.select(index == 0,0.,2.)
            controller.check()
            observations[index] = bed[1]
            commands[index] = controller.commanded[0]
    traverse()
    actual = observations.to_numpy()
    assert commands.to_numpy()[0] == 0
    assert np.all(commands.to_numpy()[1:] == 1)
    reference = []
    command_tick = _seconds_to_ticks(command)
    for t in times:
        difference = max(_seconds_to_ticks(t)-command_tick,0)
        age = np.float32(difference>>32)/np.float32(256.) + np.float32(
            difference & 0xffffffff)*np.float32(2.**-40)
        fraction = min(np.float32(age/np.float32(duration)),np.float32(1.))
        reference.append(opened if fraction == 1 else closed+(opened-closed)*fraction)
    reference = np.array(reference,dtype=np.float32)
    error = np.abs(actual.astype(float)-reference.astype(float))
    bound = float(4*np.finfo(np.float32).eps*max(1.,abs(float(closed)),abs(float(opened))))
    assert float(error.max()) <= bound
    assert actual[-1] == opened
    assert np.all(np.diff(actual[1:]) <= 0)
    report = controller.report(np.zeros(4,dtype=np.float32),np.array([0,1,1,1]),command+duration)
    gate = report['gates'][0]
    assert gate['command_seconds'] == command
    assert gate['fully_open_seconds'] == command+duration
    assert gate['first_fully_open_sample_seconds'] == command+duration
    evidence = dict(backend=ti.lang.impl.current_cfg().arch.name,
        command_seconds=command,duration_seconds=duration,sample_count=len(times),
        max_replay_error_m=float(error.max()),absolute_bound_m=bound,
        bitwise_equal=actual.tobytes()==reference.tobytes(),gate=gate,
        gpu_window=metal_runtime,source_sha256=source_tree_sha256(Path(__file__).parents[1]/'src'),
        dependencies={name:importlib.metadata.version(name) for name in ('taichi','fastdb4py','numpy')})
    (tmp_path/'late-clock.json').write_text(json.dumps(evidence,indent=2)+'\n')


@pytest.mark.parametrize('regime',['near_balance','dry_wet'])
def test_metal_paired_section_uses_half_open_bed(tmp_path,metal_runtime,regime):
    from test_gate_extended import test_uniform_and_hanging_section_agree_on_width_and_cumulative_flow
    test_uniform_and_hanging_section_agree_on_width_and_cumulative_flow(tmp_path,regime)


@pytest.mark.parametrize('pond_depth',[0,2])
def test_metal_pressure_reconstruction_reads_changed_bed(tmp_path,metal_runtime,monkeypatch,pond_depth):
    from test_gate_solver import test_hanging_subedges_have_one_shared_flux_and_new_bed_pressure
    test_hanging_subedges_have_one_shared_flux_and_new_bed_pressure(tmp_path,pond_depth,monkeypatch)
