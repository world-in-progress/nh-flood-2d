"""
Clean a UVH snapshot for warm-start: keep water only on specified
element types (default: fish pond + water body), reset all others to dry bed.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import fastdb4py as fdb

from ..schema.feature import Ne, UVH


def clean_uvh_for_warmstart(
    uvh_path: str,
    ne_fdb_path: str,
    output_path: str,
    keep_types: Sequence[int] = (7, 8),
) -> str:
    """Clean a UVH .fdb so only *keep_types* elements retain water.

    Parameters
    ----------
    uvh_path     : Source UVH .fdb snapshot.
    ne_fdb_path  : ne.fdb with element type info.
    output_path  : Destination for the cleaned .fdb.
    keep_types   : Element types that keep their water (default 7=fish pond, 8=water body).

    Returns
    -------
    str — the *output_path* written.
    """
    ne_db = fdb.ORM.load(ne_fdb_path, from_file=True)
    eu = ne_db[Ne][Ne].column.type.copy()
    ez = ne_db[Ne][Ne].column.z.copy()
    del ne_db
    e_num = len(eu)

    uvh_db = fdb.ORM.load(uvh_path, from_file=True)
    h = uvh_db[UVH][UVH].column.h
    u = uvh_db[UVH][UVH].column.u
    v = uvh_db[UVH][UVH].column.v
    assert len(h) == e_num, f'UVH size {len(h)} != NE size {e_num}'

    keep_mask = np.isin(eu, list(keep_types))
    wet_before = int(np.sum((h[1:] - ez[1:]) > 0.001))
    wet_keep = int(np.sum(keep_mask[1:] & ((h[1:] - ez[1:]) > 0.001)))

    reset_mask = ~keep_mask
    reset_mask[0] = False  # skip virtual element 0
    h[reset_mask] = ez[reset_mask]
    u[reset_mask] = 0.0
    v[reset_mask] = 0.0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    uvh_db.save(str(out))
    uvh_db.unlink()

    print(f'[warmstart] Input : {uvh_path}')
    print(f'[warmstart] Types kept: {sorted(keep_types)}')
    print(f'[warmstart] Wet cells: {wet_before:,} → {wet_keep:,} (removed {wet_before - wet_keep:,})')
    print(f'[warmstart] Output: {output_path}')
    return str(out)
