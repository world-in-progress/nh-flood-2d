"""Depth-frame mesh mismatches are rejected at the reader boundary."""
import numpy as np
import pytest

from src.nh_flood_2d.core.output import read_depth_frame, write_depth_frame


def test_depth_frame_rejects_different_mesh_size(tmp_path):
    path = tmp_path / 'depth_900.fdb'
    write_depth_frame(path, np.array([0, .25, .5], dtype=np.float32))
    with pytest.raises(ValueError, match='expected 4'):
        read_depth_frame(path, expected_count=4)
