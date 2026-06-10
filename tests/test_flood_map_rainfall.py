from datetime import datetime
from pathlib import Path
import sys

import fastdb4py as fdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nh_flood_2d.output import flood_map
from src.nh_flood_2d.schema.feature import Rainfall


def test_load_rainfall_plot_series_round_trips_local_timestamps(tmp_path):
    rain_fdb = tmp_path / "rain.fdb"

    db = fdb.ORM.truncate([fdb.TableDefn(Rainfall, 2)])
    db[Rainfall][Rainfall].column.time[:] = np.array(
        [
            datetime(2023, 9, 7, 17, 0).timestamp(),
            datetime(2023, 9, 7, 17, 5).timestamp(),
        ],
        dtype=np.float64,
    )
    db[Rainfall][Rainfall].column.quantity[:] = np.array([1.5, 2.0], dtype=np.float64)
    db.save(str(rain_fdb))

    loader = getattr(flood_map, "_load_rainfall_plot_series", None)
    assert loader is not None, "expected flood_map to expose _load_rainfall_plot_series()"

    rain_times, rain_qty = loader(rain_fdb)

    assert rain_times == [
        datetime(2023, 9, 7, 17, 0),
        datetime(2023, 9, 7, 17, 5),
    ]
    np.testing.assert_allclose(rain_qty, [1.5, 2.0])
