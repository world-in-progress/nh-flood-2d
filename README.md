# nh-flood-2d

2D hydrodynamic model that simulates shallow water flow on multi-resolution Cartesian grids using GPU-accelerated computation.

## Overview

`nh-flood-2d` is a high-performance 2D hydrodynamic simulation framework designed for flood modeling and analysis. It leverages GPU acceleration via [Taichi](https://taichi-lang.org/) and stores mesh/simulation data in a custom binary format using [`fastdb4py`](https://github.com/world-in-progress/fastdb).

### Key Features

- **GPU-accelerated computation** using Taichi for high-performance simulation
- **Multi-resolution Cartesian grids** support for flexible domain representation
- **Comprehensive physics model** including:
  - Semi-implicit Saint-Venant equations for shallow water flow
  - Horton infiltration model for 7 land-use types
  - Gate operation logic (open/close based on water head)
  - Tide boundary conditions with linear interpolation
  - Rainfall forcing with time-varying rates
- **Modular architecture** with separate configuration for domain and forcing
- **Multiple output formats**:
  - GeoTIFF flood maps (rasterized water depth)
  - Hydrograph time series at observation stations
  - Binary FDB files for intermediate data storage
- **Configurable simulation parameters**:
  - Courant number (CFL condition)
  - Time weighting factor
  - Minimum water depth threshold
  - Output intervals and duration

## Installation

This project uses `uv` for dependency management (Python 3.10.17 required).

```bash
# Install dependencies
uv sync

# Run the simulation
uv run python main.py
```

## Project Structure

```
nh-flood-2d/
├── src/nh_flood_2d/
│   ├── input/              # Configuration management
│   │   ├── __init__.py     # Main imports (DomainConfig, ForceConfig)
│   │   ├── domain.py       # Domain configuration (terrain, simulation parameters)
│   │   └── force.py        # Force configuration (boundary conditions)
│   ├── preprocess/         # Data preprocessing
│   │   ├── __init__.py     # Main preprocessing function
│   │   ├── domain.py       # Domain data preparation
│   │   ├── force.py        # Force data preparation
│   │   ├── pass_1.py       # Legacy pass 1 (raw to FDB conversion)
│   │   └── pass_2.py       # Legacy pass 2 (boundary identification)
│   ├── core/               # Core simulation engine
│   │   ├── solver_compact.py  # Main production solver (GPU-accelerated)
│   │   ├── solver.py          # Legacy functional solver
│   │   └── domain.py          # Experimental object-oriented implementation
│   ├── output/             # Output generation
│   │   ├── flood_map.py    # GeoTIFF flood map generation
│   │   └── hydrograph.py   # Hydrograph analysis and plotting
│   ├── schema/             # Data schema definitions
│   │   └── feature.py      # FDB Feature classes for data storage
│   └── util/               # Utility functions
│       ├── ti.py           # Taichi initialization and helpers
│       └── benchmark.py    # Timing decorator for performance measurement
├── main.py                 # Main entry point with example usage
├── resource/               # Configuration and input data files
│   ├── domain_*.json       # Domain configuration files
│   ├── df*.json           # Force configuration files
│   └── elevate/           # Elevation adjustment data
└── CLAUDE.md              # Developer guide for Claude Code
```

## API Reference

The following functions are exposed in `main.py` and available for use:

### Configuration Loading

```python
from src.nh_flood_2d.input import load_domain_config, DomainConfig, load_force_config, ForceConfig

# Load domain configuration (terrain and simulation parameters)
domain_cfg = load_domain_config('./resource/domain_mrcg.json')

# Load force configuration (boundary conditions)
force_cfg = load_force_config('./resource/df7.json')
```

### Main Simulation Pipeline

```python
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver

# Complete simulation workflow
def evolve_domain(domain_cfg: DomainConfig, force_cfg: ForceConfig):
    preprocess(domain_cfg, force_cfg)    # Data preparation
    solver(domain_cfg, force_cfg)        # Core simulation
```

### Elevation Adjustment

```python
from src.nh_flood_2d.core.solver_compact import set_elevation

# Raise ground elevation of elements below specified level
set_elevation(domain_cfg, elevate_meter=3.0)
```

### Output Generation

```python
from src.nh_flood_2d.output.flood_map import generate_flood_map, generate_max_inundation_extent_map
from src.nh_flood_2d.output.hydrograph import draw_hydrograph, compare_hydrograph

# Generate flood maps
generate_flood_map(domain_cfg)                      # Single timestep flood map
generate_max_inundation_extent_map(domain_cfg)      # Maximum inundation extent map

# Analyze hydrographs
draw_hydrograph(domain_cfg, 'D74', clampped=True, translation_second=-3600)

# Compare multiple simulations
mses = compare_hydrograph(
    [domain_cfg1, domain_cfg2],
    'D74',
    clampped=True,
    show=False,
    show_obs=False,
    baseline=domain_cfg1
)
```

## Configuration Files

### Domain Configuration (`domain_*.json`)
```json
{
  "ne": "path/to/ne.txt",
  "ns": "path/to/ns.txt",
  "epsg_code": 4326,
  "domain_dir": "output/domain_mrcg",
  "afa": 0.5,
  "sita": 1.0,
  "min_h": 0.02,
  "duration": -1,
  "yield_step": 300,
  "hydrograph_points": {
    "D74": [827040.3, 843912.8],
    "D75": [827120.5, 843850.2]
  },
  "observation_dir": "path/to/observations"
}
```

### Force Configuration (`df*.json`)
```json
{
  "gate": "path/to/gate.txt",
  "tide": "path/to/tide.csv",
  "rain": "path/to/rain.csv",
  "force_dir": "output/force_df7"
}
```

## Usage Examples

### Basic Simulation
```python
from src.nh_flood_2d.preprocess import preprocess
from src.nh_flood_2d.core.solver_compact import solver
from src.nh_flood_2d.input import load_domain_config, load_force_config

# Load configurations
domain_cfg = load_domain_config('./resource/domain_mrcg.json')
force_cfg = load_force_config('./resource/df7.json')

# Run complete simulation
preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

### Elevation Adjustment + Simulation
```python
from src.nh_flood_2d.core.solver_compact import set_elevation

# Adjust elevations before simulation
set_elevation(domain_cfg, elevate_meter=3.0)

# Run simulation
preprocess(domain_cfg, force_cfg)
solver(domain_cfg, force_cfg)
```

### Post-processing Analysis
```python
# Generate flood maps
generate_flood_map(domain_cfg)
generate_max_inundation_extent_map(domain_cfg, min_depth=0.2)

# Plot hydrographs
draw_hydrograph(domain_cfg, 'D74', clampped=True)

# Compare simulations
mses = compare_hydrograph(
    [domain_mrcg, domain_4],
    'D74',
    clampped=True,
    show=True
)
print(f'RMSE values: {mses}')
```

## Data Schema

The model uses `fastdb4py` Feature subclasses for data storage:

| Feature | Description | Fields |
|---------|-------------|--------|
| `Ne` | Hydro element | `index`, `x`, `y`, `z`, `type` (1-7) |
| `Ns` | Hydro side | `index`, `length`, `x`, `y`, `z`, `attr` |
| `SideTopoInfo` | Side topology | `[orient, lower_ei, upper_ei]` |
| `Tide` | Tide time series | `time`, `level` |
| `Rainfall` | Rainfall time series | `time`, `quantity` |
| `Gate` | Gate information | `info[100]` (per gate) |
| `UVH` | Simulation output | `u`, `v`, `h` per element |
| `IndexLike` | Index storage | `index` |
| `U8Value` | 8-bit value storage | `value` |

**Note:** Index 0 is always virtual (placeholder). Real elements/sides start at index 1.

## Physics Model

### Governing Equations
The model solves the 2D shallow water equations (Saint-Venant equations) using a semi-implicit finite volume scheme:

1. **Continuity equation**: ∂h/∂t + ∇·(hu) = R - I
2. **Momentum equation**: ∂u/∂t + u·∇u = -g∇h - g∇z - τ/ρ

Where:
- `h`: water depth
- `u`: flow velocity vector
- `z`: ground elevation
- `R`: rainfall rate
- `I`: infiltration rate
- `g`: gravitational acceleration
- `τ`: bottom friction (Manning's formula)
- `ρ`: water density

### Infiltration Model
Horton infiltration model applied per land-use type (7 types):
- Building, Road, Agricultural land, Fish pond, Mountainous land, Water body, Catch basin

### Gate Operation
Gates open/close based on upstream vs downstream water head difference.

### Boundary Conditions
- Tide: Time-varying water level at domain boundaries
- Rainfall: Time-varying precipitation rate over domain

## Performance

- **GPU acceleration**: All core computations run on GPU via Taichi
- **Memory efficient**: FDB format minimizes memory footprint
- **Scalable**: Tiled processing for large domains (tile size 4096)
- **Optimized**: CSR-like data layout for efficient neighborhood access

## Contributing

Please refer to `CLAUDE.md` for detailed developer guidelines and codebase conventions.

## License

[License information to be added]

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```

## Acknowledgments

- [Taichi](https://taichi-lang.org/) for GPU computing infrastructure
- [fastdb4py](https://github.com/world-in-progress/fastdb) for efficient binary data storage
- [rasterio](https://rasterio.readthedocs.io/) for GeoTIFF output support