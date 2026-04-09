# GitHub Copilot Instructions

## Commands

```bash
# Install dependencies (Python 3.10.17 required)
uv sync

# Run the simulation
uv run python main.py
```

No test suite or linter is configured.

## Architecture Overview

The pipeline has three stages: **Preprocess → Solve → Output**.

```
main.py
  └── evolve_domain(domain_cfg, force_cfg)
        ├── preprocess(domain_cfg, force_cfg)       # CSV/text → .fdb binary
        │     ├── prepare_force(force_cfg)           # gate, tide, rain → .fdb
        │     └── prepare_domain(domain_cfg)         # ne/ns → ne.fdb, ns.fdb, boundary.fdb
        └── solver(domain_cfg, force_cfg)            # GPU simulation loop → uvh/*.fdb
              (src/nh_flood_2d/core/solver_compact.py — the production solver)
```

Post-processing (independent of solver):
- `generate_flood_map(domain_cfg)` → GeoTIFF per timestep
- `generate_max_inundation_extent_map(domain_cfg)` → max-depth GeoTIFF
- `draw_hydrograph / compare_hydrograph` → time-series plots and RMSE

## Configuration

Two separate Pydantic `BaseModel` configs loaded from JSON:

- **`DomainConfig`** (`src/nh_flood_2d/input/domain.py`) — terrain paths (`ne`, `ns`), EPSG, output dirs, CFL (`afa`), time weighting (`sita`), `min_h`, `duration`, `yield_step`, observation stations
- **`ForceConfig`** (`src/nh_flood_2d/input/force.py`) — `gate`, `tide`, `rain` raw file paths and `force_dir`

Both expose `@property` methods for derived `.fdb` paths (e.g., `domain_cfg.ne_fdb`, `force_cfg.tide_fdb`). **Never construct these paths manually.**

```python
domain_cfg = load_domain_config('./resource/domain_alt.json')
force_cfg  = load_force_config('./resource/df7.json')
```

## Key Conventions

### Index 0 is always virtual
All arrays and Taichi kernel loops start at index 1. Element 0 and side 0 are padding/placeholder. Never access or write index 0 in kernels.

### FDB access pattern
`fastdb4py` is a columnar binary store. Accessing data follows:
```python
db = fdb.ORM.load(path, from_file=True)
table = db[FeatureClass]['table_name']   # or db[FeatureClass][FeatureClass] for default table
arr = table.column.field_name            # returns a numpy array (mutable in-place)
db.save(path)                            # persist changes
```

### Taichi kernels
- Always decorated with both `@ti.kernel` and `@no_type_check` (Taichi's type inference conflicts with Python type checkers)
- `init_taichi()` is a singleton — safe to call multiple times, initializes only once
- Use `copy_to_taichi(np_array, dtype, shape)` to move numpy arrays into Taichi fields
- GPU fields must be declared before kernel definitions that reference them
- Physical constants: `n = 0.033` (Manning), `g = 9.81`, `afa` (Courant), `sita` (time-weighting)

### CSR-like side-index layout
Each element's neighbor sides are stored in flat arrays with per-element pointer arrays:
- `isl_data_t` — flat array of all side indices
- `isl_ptr_l/r/b/top_t` — per-element start positions into `isl_data_t` for each compass direction

### Data schema (`src/nh_flood_2d/schema/feature.py`)
All stored data uses `fdb.Feature` subclasses. Key types:
| Class | Purpose |
|---|---|
| `Ne` | Hydro element (x, y, z, type 1–7) |
| `Ns` | Hydro side (length, x, y, z, attr) |
| `SideTopoInfo` | Packed `[orient, lower_ei, upper_ei]`; orient 1=horizontal, 2=vertical |
| `Gate` | 100 int32 per gate: `[upstream_ei, downstream_ei, height, influenced_ei...]` |
| `UVH` | Simulation output per element: u, v, h |
| `IndexLike` / `U8Value` | Generic index/flag storage |

### Output directory layout (from `domain_cfg`)
- `domain_dir/preprocessed/` — ne.fdb, ns.fdb, boundary.fdb
- `domain_dir/uvh/` — uvh_<timestamp>.fdb per yield step
- `domain_dir/flood_maps/` — GeoTIFF flood maps
- `domain_dir/max_inundation/` — max depth GeoTIFF

### Production solver
`solver_compact.py` is the active solver. `solver.py` and `core/domain.py` are legacy/experimental — do not modify them for production changes.

### Land-use types
`Ne.type` uses integer codes 1–7: Building, Road, Agricultural land, Fish pond, Mountainous land, Water body, Catch basin. The Horton infiltration parameters are indexed by this type.
