# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nh-flood-2d` is a 2D hydrodynamic model that simulates shallow water flow on multi-resolution Cartesian grids. It uses GPU-accelerated computation via [Taichi](https://taichi-lang.org/) and stores mesh/simulation data in a custom binary format via [`fastdb4py`](https://github.com/world-in-progress/fastdb) (source: https://github.com/world-in-progress/fastdb).

## Commands

This project uses `uv` for dependency management (Python 3.10.17 required).

```bash
# Install dependencies
uv sync

# Run the simulation
uv run python main.py
```
There is no test suite or linter configured.

## Architecture

The pipeline has three main stages: **Preprocess → Solve → Output**.

### Entry Point

`main.py` — loads `./resource/config.json` as an `InputConfig`, then calls `preprocess(config)`, `solver(config)`, and output functions. Some steps are commented out during development.

### Configuration (`src/nh_flood_2d/input/__init__.py`)

`InputConfig` (Pydantic model) reads a JSON config with paths to raw input files (`ne`, `ns`, `gate`, `tide`, `rain`) and an `output_dir`. All derived paths (FDB files, output subdirs) are computed as `@property` methods.

Key config fields:
- `afa` — Courant number (CFL); default 0.5
- `sita` — time weighting factor; default 1.0
- `min_h` — minimum wet depth threshold; default 0.02 m
- `duration` — simulation duration in seconds; `-1` means run until tide data ends
- `yield_step` — output interval in seconds; default 300 (5 min)
- `hydrograph_points` — dict of `name -> (x, y)` for observation stations

### Preprocessing (`src/nh_flood_2d/preprocess/`)

**Pass 1** (`pass_1.py`) — converts raw text/CSV input files into `.fdb` (FastDB) binary files stored in `output_dir/preprocessed/`:
- `ne.fdb` — hydro elements (Ne), with side-index lookup tables (isl1–isl4)
- `ns.fdb` — hydro sides (Ns), with topology info (SideTopoInfo)
- `gate.fdb`, `tide.fdb`, `rain.fdb`

Optional: `_filter_ne_ns()` removes elements with `z == -9999` and renumbers indices.

**Pass 2** (`pass_2.py`) — builds `boundary.fdb` by identifying boundary hydro elements (sides where one neighbor is element 0) using Taichi GPU kernels.

### Core Simulation (`src/nh_flood_2d/core/`)

Two implementations exist:
- **`solver.py`** — functional-style solver (all state as local variables/closures around Taichi kernels). The active solver used by `main.py`.
- **`domain.py`** — object-oriented `Domain` class wrapping the same logic. Currently an alternative/experimental path.

The simulation loop in both:
1. Interpolates tide boundary condition at current time
2. Computes rainfall rate from input CSV
3. Calls the `tick()` Taichi kernel which:
   - Updates gate states (open/close based on upstream vs downstream head)
   - Advances flow on all sides using the semi-implicit Saint-Venant scheme
   - Applies Horton infiltration model per land-use type (7 types)
   - Updates water depth per element; enforces minimum depth
   - Sets boundary element water levels to current tide value
4. Writes UVH (u-velocity, v-velocity, h-water elevation) data every `yield_step` seconds to `output_dir/uvh/uvh_<timestamp>.fdb`

### Data Schema (`src/nh_flood_2d/schema/feature.py`)

All stored data uses `fastdb4py` `Feature` subclasses:
- `Ne` — hydro element: index, x, y, z (ground elevation), 4 side counts, type (land use 1–7)
- `Ns` — hydro side: index, length, x, y, z, attr
- `SideTopoInfo` — packed as `[orient, lower_ei, upper_ei]` per side; orient 1=horizontal, 2=vertical
- `Tide`, `Rainfall` — time-series boundary conditions
- `Gate` — packed array of 100 int32 per gate: [upstream_ei, downstream_ei, height, influenced_ei...]
- `UVH` — simulation output per element: u, v, h
- `IndexLike`, `U8Value` — generic index/flag storage

**Virtual element 0 and virtual side 0** are padding; all real data starts at index 1.

### Output (`src/nh_flood_2d/output/`)

- **`flood_map.py`** — rasterizes UVH snapshots to GeoTIFF flood maps using tiled GPU processing (tile size 4096). Output CRS is set from `config.epsg_code`.
- **`hydrograph.py`** — extracts water level time series at named station points and plots them against observed CSV data using matplotlib.

### Utilities (`src/nh_flood_2d/util/`)

- `ti.py` — `init_taichi()` (with singleton guard) and `copy_to_taichi()` for converting numpy arrays to Taichi fields
- `benchmark.py` — timing decorator

## Key Conventions

- **Index 0 is always virtual** (placeholder). Real elements/sides start at index 1 in all arrays and Taichi kernel loops.
- Taichi kernels use `@no_type_check` because Taichi's type inference conflicts with Python type checkers.
- `fastdb4py` (FDB) is a columnar binary store. Access pattern: `db[FeatureClass]['table_name'].column.field_name` returns a numpy array.
- The `solver.py` and `domain.py` contain near-identical physics logic; `solver.py` is the production path.
- Raw input file formats: NE/NS are CSV-like text files; tide/rainfall are CSV with datetime headers.
