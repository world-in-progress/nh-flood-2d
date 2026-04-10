"""
Clean a UVH snapshot for warm-start: keep water only on water-body
and fish-pond elements, reset all other cells to dry bed.

Usage:
    uv run python tools/clean_uvh_for_warmstart.py \
        --uvh  resource/alt-mrcg/uvh/uvh_20240427-000000.fdb \
        --ne   resource/alt-mrcg/preprocessed/ne.fdb \
        --out  resource/alt-mrcg/warmstart.fdb \
        --keep-types 7 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# add project root so fastdb4py and schema imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import fastdb4py as fdb
from nh_flood_2d.schema.feature import Ne, UVH


def main():
    parser = argparse.ArgumentParser(
        description='Clean UVH for warm-start: keep water only on specified element types.',
    )
    parser.add_argument('--uvh', required=True, help='Input UVH .fdb file')
    parser.add_argument('--ne', required=True, help='ne.fdb with element types')
    parser.add_argument('--out', required=True, help='Output cleaned UVH .fdb')
    parser.add_argument(
        '--keep-types', nargs='+', type=int, default=[7, 8],
        help='Element types that retain water (default: 7=fish pond, 8=water body)',
    )
    args = parser.parse_args()

    # load element types
    ne_db = fdb.ORM.load(args.ne, from_file=True)
    eu = ne_db[Ne][Ne].column.type.copy()
    ez = ne_db[Ne][Ne].column.z.copy()
    del ne_db
    e_num = len(eu)

    # load UVH snapshot
    uvh_db = fdb.ORM.load(args.uvh, from_file=True)
    h = uvh_db[UVH][UVH].column.h
    u = uvh_db[UVH][UVH].column.u
    v = uvh_db[UVH][UVH].column.v

    assert len(h) == e_num, f'UVH size {len(h)} != NE size {e_num}'

    keep_set = set(args.keep_types)
    keep_mask = np.isin(eu, list(keep_set))

    wet_before = int(np.sum((h[1:] - ez[1:]) > 0.001))
    wet_keep = int(np.sum(keep_mask[1:] & ((h[1:] - ez[1:]) > 0.001)))
    wet_remove = wet_before - wet_keep

    # reset non-keep cells to ground elevation (dry bed)
    reset_mask = ~keep_mask
    reset_mask[0] = False  # skip virtual element 0
    h[reset_mask] = ez[reset_mask]
    u[reset_mask] = 0.0
    v[reset_mask] = 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uvh_db.save(str(out_path))
    uvh_db.unlink()

    print(f'[warmstart] Input : {args.uvh}')
    print(f'[warmstart] Types kept: {sorted(keep_set)}')
    print(f'[warmstart] Wet cells before : {wet_before:,}')
    print(f'[warmstart] Wet cells kept   : {wet_keep:,}')
    print(f'[warmstart] Wet cells removed: {wet_remove:,}')
    print(f'[warmstart] Output: {args.out}')


if __name__ == '__main__':
    main()
