import os

# Create the directory
dir_path = r'D:\codespace\wzp\nh-flood-2d\.github'
os.makedirs(dir_path, exist_ok=True)
print(f'Directory created: {dir_path}')

# File content
content = '''# GitHub Copilot Instructions

## Commands

This project uses `uv` for dependency management (Python **3.10.17** required).

```bash
uv sync                    # install dependencies
uv run python main.py      # run the simulation
```

No test suite or linter is configured.

## Architecture

The pipeline has three stages: **Preprocess → Solve → Output**.

```
main.py                          # entry point; loads JSON configs, wires stages
src/nh_flood_2d/
  input/                         # DomainConfig + ForceConfig (pydantic models)
  preprocess/                    # raw text/CSV → .fdb binary files
  core/solver_compact.py         # ← ACTIVE production solver (GPU via Taichi)
  core/solver.py                 # legacy; superseded
  core/domain.py                 # experimental OOP; not used in production
  output/flood_map.py            # rasterise UVH snapshots → GeoTIFF
  output/hydrograph.py           # time-series extraction & RMSE comparison
  schema/feature.py              # all FDB Feature subclasses
  util/ti.py                     # Taichi init (singleton) + copy_to_taichi()
```

### Configuration split

Two separate config objects are always threaded through the pipeline:

| Class | File | Covers |
|---|---|---|
| `DomainConfig` | `input/domain.py` | terrain paths, EPSG, CFL (`afa`), time step, output dir, hydrograph stations |
| `ForceConfig` | `input/force.py` | gate / tide / rain raw-data paths, force output dir |

Both expose `@property` methods for derived `.fdb` paths (e.g. `domain_cfg.ne_fdb`, `force_cfg.tide_fdb`).

Load them with:
```python
from src.nh_flood_2d.input import load_domain_config, load_force_config
domain_cfg = load_domain_config('./resource/domain_alt.json')
force_cfg  = load_force_config('./resource/df7.json')
```

### Simulation loop (solver_compact.py)

Each time-step in `solver()`:
1. Linearly interpolate tide boundary values.
2. Compute current rainfall rate from CSV.
3. Call the `tick()` Taichi GPU kernel — gate logic, semi-implicit Saint-Venant, Horton infiltration (7 land-use types), minimum-depth clamp, boundary water-level assignment.
4. Every `yield_step` seconds write `domain_dir/uvh/uvh_<timestamp>.fdb`.

Physical constants baked in: `n = 0.033` (Manning), `g = 9.81`.

### FDB data access

`fastdb4py` is a columnar binary store. The access pattern is:

```python
import fastdb4py as fdb
db  = fdb.ORM.load('path/to/file.fdb', from_file=True)
arr = db[FeatureClass]['table_name'].column.field_name  # → numpy array
```

All Feature subclasses are defined in `schema/feature.py`.

## Key Conventions

- **Index 0 is always virtual (padding).** All loops over elements/sides start at `range(1, n)`.
- **Taichi kernels must be decorated with `@no_type_check`** — Taichi's type inference conflicts with Python's type checker.
- **`init_taichi()` is a singleton** — calling it multiple times is safe; only the first call takes effect. Call it before any `copy_to_taichi()` usage.
- **`copy_to_taichi(np_array, dtype, shape)`** converts a numpy column from an FDB table to a Taichi field.
- **`solver_compact.py` is the production solver** — do not modify or extend `solver.py` / `core/domain.py`.
- `set_elevation(domain_cfg, elevate_meter)` reads point coordinates from `./resource/elevate/123.txt` (comma-delimited x,y) and raises the ground elevation of any element whose bounding box contains one of those points.
- Raw input formats: NE/NS files are whitespace/comma-delimited text; tide/rain files are CSV with datetime-indexed headers.
- `main.py` typically has several lines commented out — this is intentional during iterative development runs.
- `duration: -1` in domain config means auto-detect simulation length from the forcing data.
'''

# Write the file
file_path = os.path.join(dir_path, 'copilot-instructions.md')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'File created: {file_path}')

# Verify the file exists
if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f'✓ File exists and is {file_size} bytes')
else:
    print('✗ File creation failed')
