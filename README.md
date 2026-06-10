# nh-flood-2d

`nh-flood-2d` is a Python 3.10 hydrodynamic modeling workspace for 2D shallow-water flood simulation on multi-resolution Cartesian meshes. The current codebase also includes optional 1D SWMM pipe-network coupling, DEM fusion utilities, and post-processing tools for flood maps and hydrographs.

The solver uses Taichi kernels for the main numerical work and stores mesh, forcing, and UVH snapshot data in `fastdb4py` FDB files. `fastdb4py` is pinned to `0.1.12` because later FastDB changes are not assumed to be compatible with this model's stored data layout.

## What Is In This Repository

- 2D surface-water preprocessing and simulation (`src/nh_flood_2d/preprocess`, `src/nh_flood_2d/core/solver_compact.py`).
- Optional 2D surface-water and 1D SWMM pipe-network coupling (`src/nh_flood_2d/core/coupled`).
- FDB schema definitions for mesh, forcing, pipe, and UVH data (`src/nh_flood_2d/schema/feature.py`).
- Flood-map, maximum-inundation, video, hydrograph, and comparison utilities (`src/nh_flood_2d/output`).
- DEM fusion and TIN helper code (`src/nh_flood_2d/dem`, `examples`, `docs/dem_fusion_mask_replacement_usage.md`).
- Utility scripts for SWMM `.inp` conversion and warm-start UVH cleanup (`tools`).

Large local inputs and generated outputs live under `resource/` and are ignored by Git. A fresh clone does not include the project-specific DEM, NE/NS mesh, rainfall, tide, gate, observation, SWMM, or UVH files needed to run the local scenarios referenced by `main.py`.

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
uv run pytest tests/test_flood_map_rainfall.py tests/test_flood_map_uvh_validation.py
```

These smoke tests cover output helper behavior that does not require local model data. Full test collection currently includes DEM/GIS and legacy local-data tests; run those only in an environment with the required GMT/GIS libraries and project data. The tests are not a full validation run for a complete flood simulation.

## Data And Configuration

The model is configured with separate JSON files for the 2D domain, external forcing, and optional pipe network. Configuration loaders validate required input paths and create output directories when needed.

### Domain Config

Loaded with `load_domain_config(...)` into `DomainConfig`.

```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 4547,
  "domain_dir": "path/to/domain-output",
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "duration": -1,
  "yield_step": 300,
  "restart_uvh": "",
  "hydrograph_points": {
    "S4": [827040.3, 843912.8]
  },
  "observation_dir": "path/to/observations"
}
```

Important fields:

| Field | Meaning |
| --- | --- |
| `ne`, `ns` | Raw mesh element and side text files. |
| `epsg_code` | CRS code used by raster outputs. |
| `domain_dir` | Output root for preprocessed FDBs, UVH snapshots, flood maps, hydrographs, and maximum-inundation rasters. |
| `afa` | CFL factor used in adaptive time-step calculation. |
| `sita` | Time weighting factor used in the side-flow update. |
| `min_h` | Minimum active water depth in meters. |
| `duration` | Simulation duration in seconds; `-1` runs until forcing data ends. |
| `yield_step` | UVH output interval in seconds. |
| `restart_uvh` | Optional UVH `.fdb` snapshot for warm-start runs. |
| `hydrograph_points` | Station name to `(x, y)` coordinate mapping. |
| `observation_dir` | Optional observation files for hydrograph comparison. |

### Force Config

Loaded with `load_force_config(...)` into `ForceConfig`.

```json
{
  "gate": "path/to/gate.txt",
  "tide": "path/to/tide.csv",
  "rain": "path/to/rain.csv",
  "force_dir": "path/to/force-output"
}
```

`preprocess(...)` converts these files into `gate.fdb`, `tide.fdb`, and `rain.fdb` under `force_dir/preprocessed`.

### Pipe Config

Loaded with `load_pipe_config(...)` into `PipeConfig` when 1D-2D coupling is needed.

```json
{
  "inp": "path/to/network.inp",
  "pipe_dir": "path/to/pipe-output",
  "coupling_interval": 600.0,
  "exchange_timeout": 600.0,
  "weak_dist_thresh": 50.0
}
```

The pipe preprocessor reads SWMM nodes from the `.inp` file, builds a pipe FDB, records primary and weakly related 2D elements for each node, and writes runtime pipe files under `pipe_dir`.

## Running The 2D Solver

`main.py` is a local orchestration script with hard-coded paths under `resource/`. Use it as a template after creating local config files; do not treat it as a generic CLI entry point.

Minimal 2D workflow:

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver

domain_cfg = load_domain_config("path/to/domain.json")
force_cfg = load_force_config("path/to/force.json")

preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

The preprocessing stage creates:

- `domain_dir/preprocessed/ne.fdb`
- `domain_dir/preprocessed/ns.fdb`
- `domain_dir/preprocessed/boundary.fdb`
- `force_dir/preprocessed/gate.fdb`
- `force_dir/preprocessed/tide.fdb`
- `force_dir/preprocessed/rain.fdb`

The solver writes `uvh_*.fdb` snapshots under `domain_dir/uvh`.

## Running Coupled 2D-1D Simulations

For SWMM pipe coupling:

```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
from src.nh_flood_2d.input.pipe import load_pipe_config
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.preprocess.pipe import prepare_pipe
from src.nh_flood_2d.core.coupled import solver_coupled

domain_cfg = load_domain_config("path/to/domain.json")
force_cfg = load_force_config("path/to/force.json")
pipe_cfg = load_pipe_config("path/to/pipe.json")

preprocess(domain_cfg, force_cfg)
prepare_pipe(pipe_cfg, domain_cfg)
solver_coupled(domain_cfg, force_cfg, pipe_cfg)
```

`solver_coupled(...)` starts a 2D Taichi process and, when `pipe_cfg` is provided, a 1D SWMM pipe process. The processes exchange drainage and overflow data through multiprocessing-managed shared state at `coupling_interval` seconds.

Calling `solver_coupled(domain_cfg, force_cfg, None)` runs the coupled 2D code path without the 1D pipe process.

## Warm Starts

`DomainConfig.restart_uvh` can point to a previous UVH snapshot. The helper below creates a cleaned warm-start snapshot that preserves water only on selected element types:

```bash
uv run python tools/clean_uvh_for_warmstart.py \
  --uvh path/to/uvh_20230908-000000.fdb \
  --ne path/to/preprocessed/ne.fdb \
  --out path/to/warmstart.fdb \
  --keep-types 7 8
```

Then set `"restart_uvh": "path/to/warmstart.fdb"` in the domain config.

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
draw_hydrograph(domain_cfg, "S4")
```

These functions expect preprocessed mesh FDBs and UVH snapshots produced by the solver.

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
| `Rainfall`, `Tide`, `Gate` | Preprocessed forcing records. |
| `UVH` | Per-element velocity components and water surface elevation. |
| `Node`, `PipeTopo` | SWMM node and pipe-to-2D topology records for coupling. |
| `IndexLike`, `U8Value`, `F32Value` | Small typed helper tables. |

Index `0` is a virtual/sentinel element or side in the 2D mesh data. Real mesh iteration starts at index `1`.

## Code Map

```text
src/nh_flood_2d/
  input/            Pydantic config models and JSON loaders
  preprocess/       Raw domain, forcing, pipe, and warm-start preparation
  core/
    solver_compact.py
    coupled/        2D/1D coupled solver driver and exchange logic
  output/           Flood rasters, videos, hydrographs, and comparison maps
  schema/           fastdb4py Feature definitions
  dem/              DEM fusion and TIN utilities
  util/             Taichi initialization and timing helpers
tools/              Standalone maintenance/conversion scripts
examples/           DEM fusion runner examples
docs/               Design notes and usage documents
```
