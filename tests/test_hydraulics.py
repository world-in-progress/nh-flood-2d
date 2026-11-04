from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from src.nh_flood_2d.input import DomainConfig
from src.nh_flood_2d.input.domain import LandParameters
from src.nh_flood_2d.preprocess.hydraulics import (map_land_parameters,
                                                  open_boundary_face_fractions)


def test_open_boundary_fraction_counts_only_selected_exterior_subedges():
    selected = np.array([False, True, False, False, False])
    lengths = np.array([0., 2., 3., 4., 1.])
    topology = np.array([[0, 0, 0], [2, 0, 1], [2, 0, 1],
                         [2, 1, 2], [1, 1, 0]])
    pointers = np.array([[0, 0, 0, 0], [0, 2, 3, 4], [0, 0, 0, 0]])
    counts = np.array([[0, 0, 0, 0], [2, 1, 1, 1], [0, 0, 0, 0]])
    side_ids = np.array([1, 2, 3, 4, 4])
    fractions = open_boundary_face_fractions(selected, lengths, topology,
                                             pointers, counts, side_ids)
    np.testing.assert_array_equal(fractions[0], np.zeros(4))
    np.testing.assert_array_equal(fractions[2], np.zeros(4))
    np.testing.assert_allclose(fractions[1], [2 / 5, 0, 0, 0])


def land(manning_n, infiltration='none'):
    return LandParameters(manning_n=manning_n, infiltration=infiltration)


def test_land_mapping_averages_real_neighbours_and_mirrors_exterior():
    land_types = np.array([0, 1, 8, 10], dtype=np.uint8)
    topology = np.array([[0, 0, 0], [2, 1, 2], [1, 2, 3],
                         [2, 0, 1], [1, 3, 0]], dtype=np.uint32)
    table = {
        1: land(.02, dict(f0_mm_h=76.2, fc_mm_h=2.54, k_per_h=2)),
        8: land(.04),
        10: land(.06, dict(f0_mm_h=20.32, fc_mm_h=.508, k_per_h=10)),
    }
    side_n, f0, fc, k = map_land_parameters(table, land_types, topology)
    assert all(values.dtype == np.float32 for values in (side_n, f0, fc, k))
    np.testing.assert_allclose(side_n, [0., .03, .05, .02, .06], rtol=1e-7, atol=0.)
    np.testing.assert_allclose(f0 * 3_600_000, [0., 76.2, 0., 20.32], rtol=1e-6)
    np.testing.assert_allclose(fc * 3_600_000, [0., 2.54, 0., .508], rtol=1e-6)
    np.testing.assert_array_equal(k, [0., 2., 0., 10.])


def test_land_mapping_requires_every_real_land_class():
    with pytest.raises(ValueError, match=r'land_parameters missing land types: \[8\]'):
        map_land_parameters({10: land(.03)}, np.array([0, 8, 10], dtype=np.uint8),
                            np.array([[0, 0, 0]], dtype=np.uint32))


def test_uniform_manning_preserves_each_side_value():
    land_types = np.array([0, 1, 8, 10], dtype=np.uint8)
    topology = np.array([[0, 0, 0], [2, 1, 2], [1, 2, 3],
                         [2, 0, 1], [1, 3, 0]], dtype=np.uint32)
    side_n, *_ = map_land_parameters({key: land(.033) for key in (1, 8, 10)},
                                      land_types, topology)
    np.testing.assert_array_equal(side_n[1:], np.full(4, .033, dtype=np.float32))


def domain_config(tmp_path, **overrides):
    (tmp_path / 'ne').write_text('mesh')
    (tmp_path / 'ns').write_text('mesh')
    values = dict(ne=str(tmp_path / 'ne'), ns=str(tmp_path / 'ns'),
                  domain_dir=str(tmp_path / 'out'), epsg_code=2326,
                  start=datetime(2023, 9, 7, tzinfo=timezone.utc), duration=1,
                  warmup_duration=0, land_parameters={10: dict(manning_n=.03, infiltration='none')})
    values.update(overrides)
    return DomainConfig(**values)


@pytest.mark.parametrize('parameters', [
    .03,
    {'manning_n': -.01, 'infiltration': 'none'},
    {'manning_n': float('nan'), 'infiltration': 'none'},
    {'manning_n': .03},
    {'manning_n': .03, 'infiltration': 'invalid'},
    {'manning_n': .03, 'infiltration': {'f0_mm_h': 1., 'fc_mm_h': 2., 'k_per_h': 1.}},
    {'manning_n': .03, 'infiltration': {'f0_mm_h': -1., 'fc_mm_h': 0., 'k_per_h': 1.}},
    {'manning_n': .03, 'infiltration': {'f0_mm_h': 1., 'fc_mm_h': 0., 'k_per_h': -1.}},
    {'manning_n': .03, 'infiltration': {'f0_mm_h': 1., 'fc_mm_h': float('inf'), 'k_per_h': 1.}},
    {'manning_n': .03, 'infiltration': {'f0_mm_h': 1., 'fc_mm_h': 0.}},
    {'manning_n': .03, 'infiltration': 'none', 'unexpected': 1},
])
def test_land_config_rejects_invalid_parameter_sets(tmp_path, parameters):
    with pytest.raises(ValidationError):
        domain_config(tmp_path, land_parameters={10: parameters})


def test_old_manning_table_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        domain_config(tmp_path, land_parameters={10: dict(manning_n=.03, infiltration='none')},
                      manning_n={10: .03})


def test_json_land_parameters_accept_horton_and_none(tmp_path):
    import json
    config = domain_config(tmp_path, land_parameters={
        1: dict(manning_n=.12, infiltration=dict(f0_mm_h=76.2, fc_mm_h=2.54, k_per_h=2)),
        4: dict(manning_n=.03, infiltration='none'),
    })
    restored = DomainConfig.model_validate_json(config.model_dump_json())
    assert restored.land_parameters[1].infiltration.f0_mm_h == pytest.approx(76.2)
    assert json.loads(restored.model_dump_json())['land_parameters']['4']['infiltration'] == 'none'


def test_zero_friction_is_valid_for_analytic_experiments(tmp_path):
    cfg = domain_config(tmp_path, land_parameters={10: dict(manning_n=0, infiltration='none')})
    n, *_ = map_land_parameters(cfg.land_parameters, np.array([0,10]), np.array([[0,0,0],[2,0,1]]))
    assert np.all(n == 0)
