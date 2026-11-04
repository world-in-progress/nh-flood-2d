"""Observed levels and depths must compare with the same simulated quantity."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.nh_flood_2d.output import hydrograph


@pytest.mark.parametrize('quantity,expected', [('depth', 1.), ('level', 11.)])
def test_observation_quantity_selects_matching_samples_and_labels(tmp_path, monkeypatch, quantity, expected):
    output = tmp_path / 'output'
    output.mkdir()
    pd.DataFrame({'station': ['S1', 'S1'], 'elapsed_seconds': [0, 1],
        'time': ['2023-09-07T08:00:00+00:00', '2023-09-07T08:00:01+00:00'],
        'depth': [1., 1.], 'level': [11., 11.]}).to_csv(output / 'stations.csv', index=False)
    observations = tmp_path / 'observations'
    observations.mkdir()
    pd.DataFrame({'Date': ['07/09/2023', '07/09/2023'], 'Time': ['08:00:00', '08:00:01'],
        'Water Depth': [1., 1.], 'Water Level': [11., 11.]}).to_csv(observations / 'S1.csv', index=False)
    cfg = SimpleNamespace(domain_dir=str(tmp_path), observation_dir=str(observations))
    monkeypatch.setattr(plt, 'show', lambda: None)
    simulated = hydrograph._extract_data(cfg, 'S1', quantity)
    assert simulated[quantity].tolist() == [expected, expected]
    rmse, _ = hydrograph.compare_hydrograph([cfg], 'S1', quantity, show=False)
    assert rmse == [0.]
    hydrograph.draw_hydrograph(cfg, 'S1', quantity)
    assert plt.gca().get_ylabel() == f'Water {quantity.title()} (m)'
    plt.close('all')


@pytest.mark.parametrize('quantity,other', [('depth', 'Level'), ('level', 'Depth')])
def test_observation_rejects_the_other_quantity(tmp_path, quantity, other):
    path = tmp_path / 'obs.csv'
    pd.DataFrame({'Date': ['07/09/2023'], 'Time': ['08:00:00'],
                  f'Water {other}': [11.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=f'water {quantity} column'):
        hydrograph._load_observation_series(path, quantity)
