# Configuration reference

Models: `src/nh_flood_2d/input/domain.py`, `force.py`, and `pipe.py`.
Run commands from the repository root; paths are relative to the process working
directory, including paths inside JSON files.

## Domain and land parameters

All domain, force and nested parameter models reject unknown fields and NaN/Infinity. Numeric types use Pydantic's normal coercion (for example, an integral `300.0` may be accepted for an integer); the ranges below describe validated values. Required means there is no model default.

| Domain field | Unit / default | Validation and meaning |
| --- | --- | --- |
| `ne`, `ns` | Paths; required | Existing raw mesh text files; paths relative to working directory |
| `epsg_code` | Integer; required | Mesh CRS EPSG code; forcing CRS must match at load time |
| `domain_dir` | Path; required | Created during configuration validation; use a new run directory |
| `start` | Required ISO 8601 datetime | Must include timezone; formal simulation origin |
| `duration` | Seconds; required | Positive integer; formal simulation duration |
| `warmup_duration` | Seconds; required | Nonnegative integer; warmup precedes `start` |
| `land_parameters` | Required mapping | Nonempty; integer class keys 0–255; must cover all real mesh classes |
| `afa` | Dimensionless; `0.5` | `0 < afa <= 1`; scales the CFL time-step bound |
| `sita` | Dimensionless; `1.0` | `0 <= sita <= 1`; θ weights old local discharge against neighbouring discharges |
| `min_h` | Metres; `0.02` | Positive wet-depth threshold; not an initial water depth |
| `yield_step` | Seconds; `300` | Positive integer; station, budget and step-statistics interval |
| `restart_state` | Path; `""` | Empty means cold start; otherwise a complete checkpoint is loaded at runtime |
| `stations` | Coordinates; `{}` | Name to two finite x/y values in mesh CRS; outside-mesh points fail at runtime |
| `observation_dir` | Path; `""` | Used by hydrograph comparison; not needed to run the solver |

`duration + warmup_duration < 2**24` seconds is required. Domain validation checks existence of `ne`/`ns`, not their file format; preprocessing validates mesh content. Configuration loading can create `domain_dir` even if another field later fails validation.

| Land-parameter field | Unit / default | Validation |
| --- | --- | --- |
| `manning_n` | s/m^(1/3); required | Finite, nonnegative |
| `infiltration` | Required | Literal `"none"` or a Horton object |
| `f0_mm_h` | mm/h; required in Horton object | Nonnegative initial capacity |
| `fc_mm_h` | mm/h; required in Horton object | Nonnegative limiting capacity; `fc_mm_h <= f0_mm_h` |
| `k_per_h` | 1/h; required in Horton object | Nonnegative decay coefficient |

## Forcing

| Force field | Default / unit | Validation and meaning |
| --- | --- | --- |
| `boundaries` | Required list | May be empty; names must be unique |
| `boundaries[].name` | Required string | At least one character |
| `boundaries[].kind` | Required | `"level"` or `"discharge"` |
| `boundaries[].line` | Required path | GeoJSON line; geometry, CRS and mesh selection checked at load time |
| `boundaries[].series` | Required path | CSV `time,value`; m for level, m³/s for discharge |
| `rainfall` | Required object | Supply explicit zero bands for a dry run |
| `rainfall.raster` | Required path | Multiband GeoTIFF; band totals in mm |
| `rainfall.intervals` | Required path | CSV `interval_start,interval_end` with timezone-aware timestamps |
| `gate` | `null` | Optional independent, irreversible moving-bed gate controls |
| `gate.land_type` | Required if gate set | Integer 0–255; matching cells form shared-subedge connected components |
| `gate.enabled` | Required boolean | False preserves mesh beds and runs the unchanged numerical path |
| `gate.trigger_level` | Required; m | Strictly exceeded by a wet control cell's bed plus depth |
| `gate.opening_seconds` | Required; s | Finite, strictly positive opening duration |
| `gate.open_elevation` | Required; m | Final bed elevation, in the mesh vertical datum |
| `gate.gates` | Required nonempty list | Exactly one named gate per connected component |
| `gate.gates[].name` | Required string | Nonempty and unique |
| `gate.gates[].inside_point` | Required `[x,y]` | Finite mesh-CRS point in this gate component |
| `gate.gates[].control_point` | Required `[x,y]` | Finite mesh-CRS point inside the mesh, outside all gate cells |

`threshold` and `level_boundary` have been removed and are rejected as unknown
fields. There is no boundary-series gate trigger. See [gate control](gate-control.md)
for a complete configuration, step ordering, replay and accounting definitions.

Force paths are strings in the model; existence, CSV coverage, raster sampling and geometric overlap are checked when preprocessing/solving loads the inputs.

See the [README](../README.md#force-config) for boundary geometry selection,
CSV examples and rainfall interval coverage. Gate configuration is specified
by this reference and [gate control](gate-control.md). For hourly rain,
each band is that hour's accumulated millimetres; intensity stays constant
within the hour. Cells sample the pixel containing their centre, without area
averaging. The loader also accepts other positive contiguous interval lengths;
intensity is always band millimetres × 0.001 / interval seconds.

## Optional SWMM configuration

`PipeConfig` is a separate Pydantic `BaseModel`: unknown fields are ignored,
and it does not inherit the domain/force finite-number restriction. Numeric
coercion follows Pydantic defaults.

| Pipe field | Unit / default | Model validation |
| --- | --- | --- |
| `inp` | Path; required | Must exist |
| `pipe_dir` | Path; required | Created during validation |
| `coupling_interval` | Seconds; `600` | Positive integer |
| `exchange_timeout` | Seconds; `600.0` | Positive float |
| `weak_dist_thresh` | Metres; `50.0` | Float; no range constraint in the model |

These are model constraints; use finite physical values and a meaningful
nonnegative search distance. The runtime additionally requires an INP in CMS
units, whole-second start precision and separate source/runtime INP paths.
See [coupling](coupling-exchange-design.md) and [output](solver-output.md).
