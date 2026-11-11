from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.nh_flood_2d.input.force import GateConfig
from src.nh_flood_2d.preprocess.gate import GateLayout


def config(**changes):
    return GateConfig(**(dict(land_type=11, pond_land_type=10, enabled=True, trigger_level=3.07,
        opening_seconds=1800, open_elevation=1.3, gates=[dict(name='A',
            inside_point=(1.5, .5), control_point=(.5, .5))]) | changes))


@pytest.mark.parametrize('changes', [dict(threshold=3), dict(level_boundary='sea'),
    dict(opening_seconds=0), dict(opening_seconds=float('inf')),
    dict(pond_land_type=-1), dict(pond_land_type=256),
    dict(pond_land_type=11),
    dict(gates=[dict(name='A', inside_point=(float('nan'), 0), control_point=(0, 0))])])
def test_invalid_config(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def test_pond_land_type_is_required():
    raw = config().model_dump()
    del raw['pond_land_type']
    with pytest.raises(ValidationError, match='pond_land_type'):
        GateConfig(**raw)


def layout(cfg, types=(0, 1, 11, 11, 10, 11, 1), boundaries=(),
           bed=None, topology=None):
    mesh = SimpleNamespace(bounds=np.array([[0, 0, 0, 0]] +
        [[i, 0, i+1, 1] for i in range(len(types)-1)]))
    if topology is None:
        topology = np.array([[0, 0, 0]] +
            [[2, i, i+1] for i in range(1, len(types)-1)])
    if bed is None:
        bed = np.full(len(types), 2.0)
    return GateLayout.load(cfg, mesh, np.array(types), topology, boundaries,
                           np.array(bed))


def two_gates(**changes):
    return config(gates=[dict(name='A', inside_point=(1.5, .5), control_point=(.5, .5)),
        dict(name='B', inside_point=(4.5, .5), control_point=(5.5, .5))], **changes)


def test_component_partition():
    resolved = layout(two_gates())
    assert resolved.cell_gate.tolist() == [-1, -1, 0, 0, -1, 1, -1]
    assert resolved.controls.tolist() == [1, 6]
    assert [a.tolist() for a in resolved.cells] == [[2, 3], [5]]


def test_configurable_pond_land_type():
    resolved = layout(two_gates(pond_land_type=7),
                      types=(0, 1, 11, 11, 7, 11, 1))
    assert [a.tolist() for a in resolved.cells] == [[2, 3], [5]]


def test_mesh_loader_rejects_overlapping_regions_after_model_mutation():
    cfg = two_gates()
    cfg.pond_land_type = cfg.land_type
    with pytest.raises(ValueError, match='must not overlap'):
        layout(cfg)


def test_equal_f32_closed_target_is_allowed_but_next_higher_target_is_not():
    beds=np.full(7,np.float32(1.3),dtype=np.float32)
    layout(two_gates(open_elevation=1.3),bed=beds)
    with pytest.raises(ValueError,match='open_elevation'):
        layout(two_gates(open_elevation=float(np.nextafter(np.float32(1.3),np.float32(2.)))),bed=beds)


@pytest.mark.parametrize('enabled', [True, False])
def test_unclaimed_component_rejected_even_when_disabled(enabled):
    with pytest.raises(ValueError, match='exactly one'):
        layout(config(enabled=enabled))


@pytest.mark.parametrize('field,point,error', [('inside_point', (3.5, .5), 'not on'),
    ('inside_point', (10, 10), 'outside'), ('control_point', (1.5, .5), 'on a gate'),
    ('control_point', (10, 10), 'outside')])
def test_invalid_point(field, point, error):
    cfg = two_gates()
    setattr(cfg.gates[0], field, point)
    with pytest.raises(ValueError, match=error):
        layout(cfg)


def test_duplicate_component_and_name_rejected():
    cfg = two_gates()
    cfg.gates[1].inside_point = (2.5, .5)
    with pytest.raises(ValueError, match='exactly one'):
        layout(cfg)
    with pytest.raises(ValidationError, match='unique'):
        config(gates=[cfg.gates[0], cfg.gates[0]])


@pytest.mark.parametrize('enabled', [True, False])
def test_level_overlap_rejected(enabled):
    with pytest.raises(ValueError, match='overlap'):
        layout(two_gates(enabled=enabled),
               boundaries=[SimpleNamespace(kind='level', cells=np.array([3]))])


@pytest.mark.parametrize('enabled', [True, False])
def test_empty_pond_rejected_even_when_disabled(enabled):
    with pytest.raises(ValueError, match='selects no pond cells'):
        layout(two_gates(enabled=enabled), types=(0, 1, 11, 11, 1, 11, 1))
    with pytest.raises(ValueError, match='selects no pond cells'):
        layout(two_gates(enabled=enabled, pond_land_type=0))


@pytest.mark.parametrize('enabled', [True, False])
def test_missing_gate_interface_rejected_even_when_disabled(enabled):
    cfg = two_gates(enabled=enabled)
    no_pond_edge = np.array([[0, 0, 0], [2, 1, 2], [2, 2, 3],
                             [2, 5, 6]])
    with pytest.raises(ValueError, match='edges to pond'):
        layout(cfg, topology=no_pond_edge)
    no_river_edge = np.array([[0, 0, 0], [2, 2, 3], [2, 3, 4],
                              [2, 4, 5], [2, 5, 6]])
    with pytest.raises(ValueError, match='non-pond'):
        layout(cfg, topology=no_river_edge)


@pytest.mark.parametrize('enabled', [True, False])
def test_open_elevation_above_any_closed_cell_rejected_even_when_disabled(enabled):
    with pytest.raises(ValueError, match='open_elevation'):
        layout(two_gates(enabled=enabled), bed=(0, 2, 2, 1.2, 2, 2, 2))
    assert layout(two_gates(enabled=enabled),
                  bed=(0, 2, 2, 1.3, 2, 2, 2)).cell_gate[3] == 0
