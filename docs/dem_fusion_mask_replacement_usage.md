# DEM mask replacement

`src/nh_flood_2d/dem/dem_fusion_mask_replacement.py` resamples a DEM, rasterizes
a polygon mask and replaces masked pixels with point-cloud interpolation.
Outside-mask pixels retain the resampled DEM. The active interpolation path calls `pygmt.surface` with tension 0.25.
It imports PyGMT, which also requires an installed GMT shared library.

Run from the repository root:

```bash
uv run python src/nh_flood_2d/dem/dem_fusion_mask_replacement.py \
  --dem local-data/dem.tif \
  --mask local-data/mask.shp \
  --bay-points local-data/bay.txt \
  --shenzhenhe-points local-data/river.csv \
  --output runs/fusion/dem.tif \
  --resolution 4.0 \
  --nodata -9999.0
```

These paths are placeholders for locally authorized data. The provider's case
bathymetry and derived meshes, solver inputs and outputs are not distributed.
Use inputs in the same projected CRS, with compatible horizontal and vertical
units; the point readers do not reproject coordinates.

| CLI field | Python keyword | Default / format |
| --- | --- | --- |
| `--dem` | `dem_path` | Required raster |
| `--mask` | `mask_shp_path` | Required polygon Shapefile and sidecars |
| `--bay-points` | `bay_points_path` | Required whitespace-separated x y z, optionally preceded by an index |
| `--shenzhenhe-points` | `shenzhenhe_points_path` | Required comma-separated x,y,z; header supported |
| `--output` | `output_path` | Required output GeoTIFF |
| `--resolution` | `output_resolution` | 4.0, metres when source CRS uses metres |
| `--nodata` | `nodata_value` | -9999.0 |
| `--tin-output` | `tin_output_path` | None; accepted and logged, but the active path does not write PLY |
| `--sea-output` | `sea_output_path` | None; optional interpolated DEM output |

The Python entry point is
`src.nh_flood_2d.dem.dem_fusion_mask_replacement.fuse_dem_with_mask_replacement`.
It takes the five required paths followed by the optional keywords above.
There is no maximum-triangle-area argument in the current API or CLI.

The examples `examples/run_dem_fusion.sh` and
`examples/run_dem_fusion_mask_replacement.py` use the same current API and CLI.
They resolve the repository root from their own file locations and expect the
DEM, mask and point-cloud files under `resource/dem_rebuilt/input/`. They create
the output directory under `resource/dem_rebuilt/output/`; the output file does
not need to exist before running.

The data-dependent fusion run requires local DEM, mask and point-cloud inputs.
The synthetic-test module `tests/test_dem_fusion_mask_replacement.py` skips
without GMT. It tests resampling, rasterization, point readers and TIN helpers;
the test for the removed `create_tin_from_points` function has been removed.
The active interpolation calls GMT with raster bounds and spacing but no
explicit pixel registration; raster alignment needs validation for the supplied
inputs.
