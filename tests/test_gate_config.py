from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.nh_flood_2d.input.force import GateConfig
from src.nh_flood_2d.preprocess.gate import GateLayout


def config(**changes):
    return GateConfig(**(dict(land_type=11, enabled=True, trigger_level=3.07,
        opening_seconds=1800, open_elevation=1.3, gates=[dict(name='A',
            inside_point=(.5, .5), control_point=(2.5, .5))]) | changes))


@pytest.mark.parametrize('changes', [dict(threshold=3), dict(level_boundary='sea'),
    dict(opening_seconds=0), dict(opening_seconds=float('inf')),
    dict(gates=[dict(name='A', inside_point=(float('nan'), 0), control_point=(0, 0))])])
def test_invalid_config(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def layout(cfg, types=(0, 11, 11, 10, 11), boundaries=()):
    mesh = SimpleNamespace(bounds=np.array([[0, 0, 0, 0]] +
        [[i, 0, i+1, 1] for i in range(4)]))
    topo = np.array([[0, 0, 0], [2, 1, 2], [2, 2, 3], [2, 3, 4]])
    return GateLayout.load(cfg, mesh, np.array(types), topo, boundaries)


def two_gates(**changes):
    return config(gates=[dict(name='A', inside_point=(.5, .5), control_point=(2.5, .5)),
        dict(name='B', inside_point=(3.5, .5), control_point=(2.5, .5))], **changes)


def test_component_partition():
    resolved = layout(two_gates())
    assert resolved.cell_gate.tolist() == [-1, 0, 0, -1, 1]
    assert resolved.controls.tolist() == [3, 3]
    assert [a.tolist() for a in resolved.cells] == [[1, 2], [4]]


@pytest.mark.parametrize('enabled', [True, False])
def test_unclaimed_component_rejected_even_when_disabled(enabled):
    with pytest.raises(ValueError, match='exactly one'):
        layout(config(enabled=enabled))


@pytest.mark.parametrize('field,point,error', [('inside_point', (2.5, .5), 'not on'),
    ('inside_point', (10, 10), 'outside'), ('control_point', (.5, .5), 'on a gate'),
    ('control_point', (10, 10), 'outside')])
def test_invalid_point(field, point, error):
    cfg = two_gates()
    setattr(cfg.gates[0], field, point)
    with pytest.raises(ValueError, match=error):
        layout(cfg)


def test_duplicate_component_and_name_rejected():
    cfg = two_gates()
    cfg.gates[1].inside_point = (1.5, .5)
    with pytest.raises(ValueError, match='exactly one'):
        layout(cfg)
    with pytest.raises(ValidationError, match='unique'):
        config(gates=[cfg.gates[0], cfg.gates[0]])


def test_level_overlap_rejected():
    with pytest.raises(ValueError, match='overlap'):
        layout(two_gates(), boundaries=[SimpleNamespace(kind='level', cells=np.array([2]))])
