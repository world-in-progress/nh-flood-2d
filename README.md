# nh-flood-2d

`nh-flood-2d` is a Python 3.10 hydrodynamic modeling workspace for 2D shallow-water flood simulation on multi-resolution Cartesian meshes. The current codebase also includes optional 1D SWMM pipe-network coupling, DEM fusion utilities, and post-processing tools for flood maps and hydrographs.

The solver uses Taichi kernels for the main numerical work and stores meshes, depth frames, extrema, and complete checkpoints in `fastdb4py` FDB files.

## What Is In This Repository

- 2D surface-water preprocessing and simulation (`src/nh_flood_2d/preprocess`, `src/nh_flood_2d/core/solver_compact.py`).
- Optional 2D surface-water and 1D SWMM pipe-network coupling (`src/nh_flood_2d/core/coupled`).
- FDB schema definitions for mesh, forcing, and pipe data (`src/nh_flood_2d/schema/feature.py`), plus frame and checkpoint I/O (`src/nh_flood_2d/core/output.py`).
- Flood-map, maximum-inundation, video, hydrograph, and comparison utilities (`src/nh_flood_2d/output`).
- DEM fusion and TIN helper code (`src/nh_flood_2d/dem`, `examples`, `docs/dem_fusion_mask_replacement_usage.md`).
- Utility scripts for SWMM `.inp` conversion and solver self-checks (`tools`).

Large local inputs and generated outputs live under `resource/` and are ignored by Git. A fresh clone does not include the project-specific DEM, NE/NS mesh, rainfall, tide, gate, observation, SWMM, or checkpoint files needed by your local configurations.

## Requirements

- Python `3.10.17`.
- `uv` for dependency management.
- A Taichi-supported compute backend for solver runs.
- GIS and SWMM dependencies installed through `pyproject.toml`.
- Local model data files matching the configuration JSON files you provide.

Install dependencies:

```bash
uv sync
```

On macOS, if loading `swmm-toolkit` terminates the process because of an invalid bundled dylib signature, run:

```bash
uv run fix-macos-codesign
```

## Verification Commands

```bash
uv lock --check
uv run pytest tests/test_forcing.py tests/test_solver_forcing.py \
  tests/test_clock.py tests/test_device_budget.py tests/test_face_flux.py \
  tests/test_open_boundary.py tests/test_hydraulics.py \
  tests/test_positivity.py tests/test_inventory_rounding.py tests/test_restart.py \
  tests/test_coupling.py tests/test_output_io.py \
  tests/test_output_readers.py tests/test_flood_map_frame_validation.py \
  tests/test_flood_map_rainfall.py
uv run pytest
```

The focused checks cover forcing, the device clock and budget, face fluxes, boundaries, coupling, checkpoint I/O, depth frames, and post-processing. The full command checks the rest of the repository, including tests that may require local DEM data or GMT.

## Data And Configuration

The compact solver uses separate domain and forcing JSON files. Paths are relative to the process working directory. Unknown fields are rejected. `start`, `duration`, `warmup_duration`, and `land_parameters` are required; duration and output intervals use positive whole seconds, and warmup accepts zero. `duration + warmup_duration` must be less than `2**24` seconds.

### Domain Config

```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 2326,
  "domain_dir": "path/to/new-run",
  "start": "2023-09-07T16:00:00+08:00",
  "duration": 7200,
  "warmup_duration": 43200,
  "land_parameters": {
    "1": {"manning_n": 0.12, "infiltration": {"f0_mm_h": 76.2, "fc_mm_h": 2.54, "k_per_h": 2}},
    "2": {"manning_n": 0.08, "infiltration": {"f0_mm_h": 76.2, "fc_mm_h": 2.54, "k_per_h": 2}},
    "3": {"manning_n": 0.08, "infiltration": {"f0_mm_h": 20.32, "fc_mm_h": 0.508, "k_per_h": 10}},
    "4": {"manning_n": 0.03, "infiltration": "none"},
    "5": {"manning_n": 0.025, "infiltration": "none"},
    "6": {"manning_n": 0.035, "infiltration": {"f0_mm_h": 101.6, "fc_mm_h": 7.62, "k_per_h": 2}},
    "7": {"manning_n": 0.03, "infiltration": "none"},
    "8": {"manning_n": 0.03, "infiltration": "none"},
    "9": {"manning_n": 0.06, "infiltration": {"f0_mm_h": 152.4, "fc_mm_h": 25.4, "k_per_h": 3}}
  },
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "yield_step": 300,
  "restart_state": "",
  "stations": {"S4": [825187.3806, 841959.8752]},
  "observation_dir": "path/to/observations"
}
```

`start` must include a timezone. All timestamps are aligned by absolute time, independently of the machine's local timezone. Boundary series and rainfall intervals must cover the formal interval from `start` through `start + duration`; missing coverage is an error. Warmup uses only the value at `start` and does not read earlier data. There is no extrapolation or endpoint padding during the formal simulation. The mesh, boundary lines and rainfall raster must use the same EPSG; coordinates are not reprojected.

`land_parameters` must cover every land class present in the mesh; missing classes are errors. JSON keys are land-class codes. The 1–9 values above come from the paper appendix and are example inputs, not built-in defaults. Add entries for any further mesh classes. Each entry requires nonnegative `manning_n` (zero permits frictionless analytic tests) and either `"infiltration": "none"` or Horton parameters `f0_mm_h`, `fc_mm_h`, and `k_per_h`. Capacities use mm/h, decay uses 1/h, all must be nonnegative, and `f0_mm_h` must be at least `fc_mm_h`. Interior-side n is the arithmetic mean of its two cell values; closed exterior sides use the interior value, since sentinel element 0 is not a land class.

`afa`, `sita`, and `min_h` control CFL, time weighting, and the wet-depth threshold. For θ<1, neighbouring face discharge is the length-weighted mean of all subedges; a single-edge face retains its value exactly. The linear θ-stencil continuation uses only selected exterior edges of prescribed level or discharge cells. Its exterior fractions are weighted by edge length; a fully missing neighbour yields a two-neighbour mean equal to the current edge's old discharge. Physical exterior fluxes stay zero; prescribed levels and cell discharge sources provide boundary exchange. θ=1 bypasses this stencil.

`stations` maps names to mesh coordinates: every closed cell rectangle containing the point is a candidate, the lowest real cell ID wins, and a point outside the mesh is an error. `observation_dir` supplies comparison data. `yield_step` defaults to 300 s and controls station samples, budget rows, and step statistics; depth frames use a fixed 900 s interval.

### Force Config

```json
{
  "boundaries": [
    {
      "name": "sea",
      "kind": "level",
      "line": "path/to/sea.geojson",
      "series": "path/to/sea.csv"
    },
    {
      "name": "shenzhen_river",
      "kind": "discharge",
      "line": "path/to/river.geojson",
      "series": "path/to/river.csv"
    }
  ],
  "rainfall": {
    "raster": "path/to/rainfall.tif",
    "intervals": "path/to/rainfall_intervals.csv"
  },
  "gate": {"threshold": 1.5, "land_type": 11, "level_boundary": "sea"}
}
```

Each boundary has a unique `name`, a `kind`, a line and a series. `level` prescribes water surface elevation in metres using the mesh elevation's vertical datum; it is not depth. `discharge` is total signed flow in m³/s, positive into the domain. An empty boundary list makes all exterior sides closed. The line file must explicitly declare its CRS, for example:

```json
{
  "type": "LineString",
  "crs": {"type": "name", "properties": {"name": "EPSG:2326"}},
  "coordinates": [[808300, 836950], [808400, 836950]]
}
```

The coordinates above only illustrate the format; replace them with a line coincident with the intended mesh exterior. LineString and MultiLineString geometries, including Feature/FeatureCollection wrappers, are accepted. A boundary line selects an entire discrete exterior edge when their collinear overlap has positive length. Corner contact and a single transverse crossing point do not select an edge. A selection with no exterior edges fails with guidance to snap the line to the mesh exterior in GIS. Cells claimed by multiple boundaries are errors. Total Q is distributed by the lengths of selected edges only; each selected cell receives one share even when several of its edges are selected.

A series CSV requires `time` and `value` columns (additional columns are ignored):

```csv
time,value
2023-09-07T16:00:00+08:00,0.6
2023-09-07T17:00:00+08:00,0.8
2023-09-07T18:00:00+08:00,0.9
```

Times must be timezone-aware ISO 8601, strictly increasing, with at least two finite values. The same format applies to river Q, with values in m³/s. Series are interpolated linearly; steps stop at series knots, and midpoint Q integrates each linear segment without duplicating inflow among cells. Level values are applied initially and at each step end.

Rainfall is one multiband GeoTIFF. Band k holds the accumulated millimetres for row k of the interval CSV:

```csv
interval_start,interval_end
2023-09-07T16:00:00+08:00,2023-09-07T17:00:00+08:00
2023-09-07T17:00:00+08:00,2023-09-07T18:00:00+08:00
```

Intervals must have positive duration, be contiguous and nonoverlapping, and match the band count. Record zero-rain intervals explicitly. For this example use two bands, such as 2 and 1 mm. Cell centres are mapped to raster pixels once; centres outside the raster, sampled NoData, nonfinite values and negative rain are errors. Intensity is constant within each interval: band millimetres × 0.001 / interval seconds. A single pixel covering the entire domain is the uniform-rain case. Configured Horton capacity is `fc + (f0 - fc) exp(-k t / 3600)`; at the first rainy instant it is `f0`. Each cell accumulates rainy time `t` only while its own intensity is positive. `none` gives zero infiltration. Actual Horton removal is the minimum of capacity × dt and the water remaining after fluxes and rain.

`gate` is optional; use `null` when no gate is modelled. Its named boundary must be `level`. When that boundary exceeds `threshold`, all cells of `land_type` have bed elevation changed to 0 m once while retaining their water depth and volume. Threshold and level use the same datum. Gate threshold crossings split steps; the manifest records the target-cell count and whether opening occurred.

Cold starts with `warmup_duration > 0` initialize connected still water before warmup. Each level boundary seeds its value at `start` (`series.at(0)`); water reaches a neighbouring cell only over an open interior edge whose effective crest, the maximum of both working beds and the momentum edge crest, is strictly below that level. Initial gate openings occur before this search. Isolated low land, enclosed ponds, and closed gates block filling; pond storage follows the input equivalent terrain without an extra land-class fill. Different starting levels reaching the same water body are an input error. Initial q is zero, and filled water belongs to initial storage. Later gate openings evolve dynamically without another fill. Runs without warmup retain dry initial conditions; checkpoints retain their stored physical state.

Warmup holds each boundary at its value at `start`, with constant starting river discharge and no rainfall. Infiltration remains active. The same engine then continues through the formal simulation. Choose warmup duration from measured station and boundary-band oscillations, storage drift and the complete river/sea/infiltration balance. `warmup_solver(domain_cfg, force_cfg)` runs an already preprocessed domain only through warmup and stops at `start`.

## Running The 2D Solver

```bash
uv run python main.py --domain path/to/domain.json --force path/to/force.json
```

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.core.solver_compact import run_simulation

run_simulation(load_domain_config("path/to/domain.json"),
               load_force_config("path/to/force.json"))
```

Use a new `domain_dir` for each run; existing manifest, budget, or output data is not overwritten. The command writes a manifest before preprocessing, and failures in preprocessing or input loading are recorded as failed runs. `preprocess(domain_cfg, force_cfg)` plus `solver(domain_cfg, force_cfg)` remains available when preprocessing is run separately.

Preprocessing writes `ne.fdb`, `ns.fdb`, and `boundary.fdb` under `domain_dir/preprocessed`. The solver writes `domain_dir/output/bed.fdb` once as the starting working bed. `output/frames/depth_<elapsed_s>.fdb` contains only F32 depth, at the initial time, every 900 s, and the final time. `output/stations.csv` contains `depth`, `level`, `u`, and `v` for configured stations. `output/extrema.fdb` holds the stepwise accumulated `max_depth`, its first occurrence as two U32 clock words, and `max_speed`. Extrema cover the formal simulation; a warmup-only run keeps warmup extrema. `output/state_start.fdb` is the checkpoint at the formal start; `output/state_final.fdb` is the complete final checkpoint. During warmup and formal simulation, once 30 minutes of wall time have passed since the last checkpoint (or simulation began), the next depth-frame output also writes `output/state_latest.fdb`. This fixed interval adds no time-step boundaries. Each checkpoint is written to a temporary file in the same directory and atomically replaces its destination; only the latest periodic checkpoint is kept. Checkpoints store working bed, depth, side discharge q, Horton rain time and capacity, clock, extrema, and exact budget accumulator state. Warmup and formal simulation use the same clock and output format. Frames written during warmup have negative elapsed seconds.

Mesh loading requires positive square cells, matching the edge-length CFL scale. Spatial WSE MAE is NaN when there are no common wet pixels; it does not indicate zero error.

The bed can change when a gate opens. `manifest.json` records each `bed_events` entry with its time, all affected cell IDs, and their new bed values. Replaying these events over `bed.fdb` reconstructs the working bed at a frame; water surface elevation is working bed plus depth.

Each `domain_dir/mass_balance.csv` row records cumulative nominal rainfall, actual infiltration, actual signed volume applied at each discharge boundary, signed volume change at each level boundary, `requested_discharge:<name>_m3`, `bed_clamp_nonlevel_m3`, `bed_clamp_level_m3`, signed `pipe_exchange_m3` (positive from pipe to surface), storage, and residual. The requested discharge column integrates the prescribed Q; the `discharge:<name>_m3` column uses the amount actually applied after donor limiting. Non-level clipping adds water and enters the balance. Level-cell clipping is an intermediate diagnostic overwritten by the prescribed depth, so it is excluded from net sources. A level below bed sets zero depth. Gate bed changes preserve depth and volume, so `gate_bed_m3` is zero. Residual is current storage minus initial storage and accounted signed sources. `relative_residual` uses the maximum of initial storage, cumulative rain plus positive cumulative net inputs, and absolute cumulative non-rain accounting terms; it is blank when that scale is zero.

Positivity uses one donor factor per cell from its depth at step start plus positive explicit sources, divided across total outward edge flux and negative explicit sources. Each shared edge is scaled once by its donor cell, so both adjacent cells see the same flux. Incoming edge flux cannot fund outgoing flux in the same step. Prescribed level cells have an unlimited reservoir and are set to the requested level at step end. CFL uses the original face depth floor `max(h, 0.01)` with the current depths after the cell update and the updated q, while excluding physical exterior edges. A real CFL below `1e-4` s fails explicitly with the limiting edge, its two cell depths, face depth, and face velocity; it is not enlarged to a supported step.

The device clock uses two U32 words at a quantum of `2^-40` s, with the warmup start as its nonnegative origin. Steps advance in device batches, split at forcing events and output times. The clock preserves elapsed time across long runs, while F32 depth, discharge, and other physical state still round. DeviceBudget retains exact integer accumulation of F32 diagnostic terms; the host converts merged integers to F64 for the ledger. Nominal rainfall is integrated separately with CPU F64. `storage_depth_one_ulp_volume_m3` reports the F32 depth-state one-ULP volume scale, separately from the measured residual.

Transport and rain increments are stored in one F32 device field before adding to the inventory. This global pass fixes their rounding order across kernel batching; it does not add a compensating source.

Gate opening uses the first rising threshold crossing from the input series, quantized to the same clock as forcing events. Its zero-time bed change and any prescribed-level correction are processed before output or the next physics step; the CFL bound is refreshed. No zero-dt physics tick is executed. Starting formal simulation from a warmup-only checkpoint resets extrema at elapsed zero.

`domain_dir/step_statistics.csv` has one row at each output, including `standard_dt_min_s`, `standard_dt_max_s`, `standard_dt_mean_s`, and `standard_steps` for standard steps; event-shortened step count, minimum, and total duration; and the tightest CFL edge with its two cell depths, face depth, and face velocity. Budget, station, and statistics rows also occur at frame times between `yield_step` boundaries. `manifest.json` includes git commit/dirty state, full configuration, input hashes, simulation and wall-clock timestamps, step and dt summaries, preprocessing/warmup/simulation timing, budget transfer counts, backend/device, final water budget, and peak device memory (`unavailable` if it cannot be measured).

## Coupled Simulations

The coupled driver uses the same depth engine, forcing loader, clock, budget and manifest. Supply a pipe configuration with `inp`, `pipe_dir`, positive whole-second `coupling_interval` (default 600), `exchange_timeout` and `weak_dist_thresh`:

```json
{
  "inp": "path/to/network.inp",
  "pipe_dir": "path/to/new-run/pipe",
  "coupling_interval": 600,
  "exchange_timeout": 600.0,
  "weak_dist_thresh": 50.0
}
```

```bash
uv run python main.py --domain path/to/domain.json --force path/to/force.json --pipe path/to/pipe.json
```

Pipe exchange occupies its own field and is added to discharge in `ssq_t`. Shared-cell drainage is allocated among unique junction relationships; outfalls receive stage without draining the surface. One SWMM actor strides each complete exchange window, including warmup, and continues across simulation start without restarting. Coupled start timestamps must have whole-second precision. The INP must declare `FLOW_UNITS CMS` so flow and head use m³/s and metres. Runtime INP/report/output files are written under `pipe_dir`; the source INP is read only. The manifest includes pipe configuration/input hash and actual SWMM clock/window metadata. The exchange retains the existing overflow head cap of 3 m and relaxation of 0.5.

## Backend Self-Check

The self-check exercises small rainfall on a 10,000 m bed, dry and wet gates that open without adding water, and checkpoint restarts. Each command needs a new output directory and writes `selfcheck.json`, manifests and budgets. The requested backend must be available; CPU fallback is disabled.

```bash
uv run python tools/solver_selfcheck.py --arch cpu --output path/to/new-check-cpu
uv run python tools/solver_selfcheck.py --arch metal --output path/to/new-check-metal
uv run python tools/solver_selfcheck.py --arch cuda --output path/to/new-check-cuda
```

CUDA device metadata uses the driver's `nvidia-smi` command, which must be on PATH.

## Restarting a Run

To resume an interrupted surface run, set `restart_state` to the previous run's `output/state_latest.fdb`, choose a new `domain_dir`, and keep all other configuration unchanged, including `start`, `duration`, `warmup_duration`, forcing, mesh, gate, and physical parameters. For example, use `"restart_state": "path/to/interrupted-run/output/state_latest.fdb"` and `"domain_dir": "path/to/new-resumed-run"`. The stored clock resumes warmup or formal simulation at the checkpoint; output begins in the new directory. A complete `output/state_start.fdb` or `output/state_final.fdb` can also be used, and `duration` may be extended if boundary and rain inputs cover the extended interval.

A checkpoint later than the requested end is rejected before result files are written; a checkpoint exactly at the end completes with zero steps. Boundary names, kinds, ordering, and selected-edge hashes must match. The next CFL candidate and its limiting-edge diagnostics are derived together from restored depth, bed, and q; no next-dt field is stored. The solver restores working bed, depth, q, Horton state, the two-word clock, extrema, and exact device budget totals. It continues without applying initial boundary levels again. The surface checkpoint does not contain SWMM actor state; attempting to resume a coupled pipe run fails explicitly.

## Post-Processing

Common output functions:

```python
from src.nh_flood_2d.output.flood_map import (
    generate_flood_map,
    generate_max_inundation_extent_map,
    generate_flood_video,
    plot_spatial_mae_curve,
)
from src.nh_flood_2d.output.hydrograph import (
    draw_hydrograph,
    compare_hydrograph,
    compare_hydrograph_panels,
)

generate_flood_map(domain_cfg)
generate_max_inundation_extent_map(domain_cfg, min_depth=0.05)
generate_flood_video(domain_cfg, output_path="path/to/flood_video.mp4")
draw_hydrograph(domain_cfg, "S4", quantity="level")
```

Hydrograph functions require `quantity="level"` or `quantity="depth"`. Observations must contain exactly one matching Water Level or Water Depth column; the other quantity is rejected. Plot labels and simulation samples use the same quantity.

These functions read the preprocessed mesh and depth frames or extrema produced by the solver. Water-surface comparisons replay bed events from the manifest.

## DEM And GIS Utilities

DEM mask-replacement fusion CLI:

```bash
uv run python src/nh_flood_2d/dem/dem_fusion_mask_replacement.py \
  --dem path/to/study_area_dem.tif \
  --mask path/to/study_area_dem_mask.shp \
  --bay-points path/to/bay.txt \
  --shenzhenhe-points path/to/shenzhenhe-fix.csv \
  --output path/to/fused_dem_4m.tif \
  --resolution 4.0 \
  --nodata -9999.0
```

SWMM `.inp` to shapefile conversion:

```bash
uv run python tools/inp2shp.py path/to/network.inp -o path/to/shapefiles --epsg 4547
```

## Data Model Notes

The FDB schema is defined in `src/nh_flood_2d/schema/feature.py`. Core records include:

| Feature | Purpose |
| --- | --- |
| `Ne` | 2D mesh element coordinates, elevation, side counts, and type. |
| `Ns` | 2D side geometry, elevation, length, and attribute. |
| `SideTopoInfo` | Side orientation and adjacent element indices. |
| `Node`, `PipeTopo` | SWMM node and pipe-to-2D topology records for coupling. |
| `IndexLike`, `U8Value`, `F32Value` | Small typed helper tables. |

Index `0` is a virtual/sentinel element or side in the 2D mesh data. Real mesh iteration starts at index `1`.

## Code Map

```text
src/nh_flood_2d/
  input/            Pydantic config models and JSON loaders
  preprocess/       Raw domain, forcing, and pipe preparation
  core/
    solver_compact.py  Surface solver and checkpoint/restart flow
    clock.py, device_budget.py, output.py
    coupled/        2D/1D coupled solver driver and exchange logic
  output/           Flood rasters, videos, hydrographs, and comparison maps
  schema/           fastdb4py Feature definitions
  dem/              DEM fusion and TIN utilities
  util/             Taichi initialization and timing helpers
tools/              Standalone maintenance/conversion scripts
examples/           DEM fusion runner examples
docs/               Design notes and usage documents
```

Source identity is always recorded as `source_sha256`. For rule `nh-flood-2d/src-py-sha256/v1`, SHA-256 starts with the UTF-8 bytes `nh-flood-2d/src-py-sha256/v1` and one NUL byte. Enumerate all `.py` files under `src/`, sorted by their source-relative POSIX paths. For each file append its UTF-8 relative path and a NUL byte, its byte length as an unsigned 8-byte big-endian integer, then its complete bytes. Thus relocation does not change identity; names and contents do. Git checkouts additionally record `git_commit` and `git_dirty`; both are JSON `null` when the source root has no `.git`. Git is not required to run.

Each step-statistics row also records `limited_cell_steps`, `limited_face_steps`, and `minimum_donor_scale` for its output interval. Counts include each donor cell or nonzero face flux reduced by the shared-water limiter in each physical step; the minimum is 1 when no limiting occurs. The manifest `limiter_activity` reports their run totals and minimum.
