from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nh_flood_2d.output import flood_map


def test_validate_uvh_snapshot_size_raises_clear_error_for_mesh_mismatch():
    validator = getattr(flood_map, "_validate_uvh_snapshot_size", None)
    assert validator is not None, "expected flood_map to expose _validate_uvh_snapshot_size()"

    with pytest.raises(ValueError) as exc_info:
        validator(
            label="cfg_ref",
            uvh_path=Path("resource/4/uvh/uvh_20230907-173000.fdb"),
            timestamp="20230907-173000",
            actual_size=16551401,
            expected_size=16519361,
        )

    msg = str(exc_info.value)
    assert "cfg_ref" in msg
    assert "16551401" in msg
    assert "16519361" in msg
    assert "different mesh" in msg.lower()
