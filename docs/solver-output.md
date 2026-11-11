# Solver output, provenance and continuation

Use `run_simulation(domain_cfg, force_cfg, pipe_cfg=None)` or `main.py` for
preprocessing plus simulation. `solver(...)` runs an already preprocessed domain;
`warmup_solver(...)` runs it only through warmup. The full entry point creates
`manifest.json` before hashing inputs and preprocessing. JSON/model validation
occurs earlier and cannot be captured by that record. Existing manifest,
`mass_balance.csv`, or nonempty `output/` causes refusal; choose a new directory.

## Files

All paths below are relative to `domain_dir`.

| File | Content and schedule |
| --- | --- |
| `preprocessed/ne.fdb`, `ns.fdb`, `boundary.fdb` | Prepared cell, side and topology data |
| `output/bed.fdb` | Starting working bed elevation, metres, F32 |
| `output/frames/depth_<elapsed_s>.fdb` | F32 depth in metres; initial/resumed instant, 900 s frame boundaries and final instant |
| `output/extrema.fdb` | Per-cell maximum depth (m), first occurrence as two U32 clock words, and maximum speed (m/s), accumulated each step |
| `output/stations.csv` | Long-form station records: `elapsed_seconds,time,station,cell_id,depth,level,u,v`; depths/elevations in m, velocities in m/s |
| `mass_balance.csv` | Cumulative surface water budget, volumes in m³ |
| `gate_balance.csv` | Enabled controls only: cumulative per-gate exchange and pond-control-volume budget at existing output times |
| `step_statistics.csv` | Per-output-interval step counts, dt ranges, limiting-edge diagnostics and limiter activity |
| `manifest.json` | Configuration, provenance, phase timings, device, summaries, gate/bed events and final budget |
| `output/state_start.fdb` | Complete state at formal start of a full run, when that instant is reached |
| `output/state_final.fdb` | Complete state at this invocation's endpoint |
| `output/state_latest.fdb` | Latest periodic checkpoint, when eligible |

Station/budget/statistics rows occur initially, on `yield_step` boundaries, at
frame times, and at the final instant. `stations.csv` is absent when no stations
are configured. Times are relative to formal `start`; warmup frames/rows have
negative elapsed seconds. Extrema reset at formal start; warmup-only runs retain
warmup extrema. Decode a first-occurrence clock as
`((hi << 32) | lo) * 2**-40` seconds after the manifest's
`output.extrema_time_origin`, not directly after formal start.

`bed.fdb` is a snapshot of the starting working bed, including a restored bed
on continuation. Continuous gate opening is described by manifest `gate_control`:
closed elevations, final elevation, duration and exact command clock words.
New runs do not append per-step `bed_events`. Use `read_bed_at` or `BedTimeline`
from `core/output.py` to reconstruct elevation. Readers still support historical
instantaneous `bed_events` without enabling the removed control interface.
Water level is working bed plus depth. Depth frames contain neither velocity
nor the complete solver state and cannot be used as restart checkpoints.
For moving gate cells, maximum depth plus a fixed bed is not maximum water
level. These cells must be excluded from fixed-bed water-level extrema analysis
or analysed with their time-varying bed and depth samples.

## Water budget and diagnostics

Budget columns include time, elapsed seconds, phase, `rainfall_m3`,
`infiltration_m3`, one `level:<name>_m3` or `discharge:<name>_m3` per boundary,
`requested_discharge:<name>_m3` for discharge boundaries,
`bed_clamp_nonlevel_m3`, `bed_clamp_level_m3`, `gate_bed_m3`,
`pipe_exchange_m3`, `initial_storage_m3`, `storage_m3`, `residual_m3` and
`relative_residual`. Rainfall and infiltration are positive input/removal totals;
boundary and pipe exchange are positive into the surface domain. Requested
and actually applied discharge can differ when water availability limits
extraction. Level-boundary reservoir corrections are included in boundary
exchange; the intermediate level-cell floor is diagnostic. A gate bed change
retains depth/volume, so `gate_bed_m3` is zero.

Residual is storage minus initial storage minus accounted net input. A zero
normalization scale produces a blank relative residual. Budgets retain exact
integer accumulation of F32 diagnostic terms across time, with host F64
conversion. This does not remove F32 state rounding. The manifest separately
reports `storage_depth_one_ulp_volume_m3`. Step statistics include
`limited_cell_steps`, `limited_face_steps`, and `minimum_donor_scale`; the
manifest's `limiter_activity` holds invocation totals.

## Checkpoints and restart

After 1800 s of wall time since the last checkpoint (or simulation start), the
next depth-frame output writes `state_latest.fdb`. Short runs need not have this
file. Writing a checkpoint uses a sibling temporary file and atomic replacement.
This schedule adds no physics step boundaries. Warmup-only invocations write the formal-start endpoint to `state_final.fdb`.
The formal-start/final checkpoints
also reset the wall-time checkpoint timer.

A checkpoint contains depth, bed, side unit discharge q (m²/s), Horton rainy
time and capacity, fixed-point clock, gate state, extrema, mesh/gate/boundary
identity, physical configuration and exact budget state. Continuation restores
these without reapplying cold-start levels. The next CFL bound is recomputed.
Enabled gates also save each command/first-flow clock, opening parameters via
the validated gate configuration, pond initial inventory and exact cumulative
exchange-budget integers. See [gate control](gate-control.md) for the split
sequence, first-flow definition and pond budget signs.

For a surface run, copy the original domain configuration, change `domain_dir`
to a new directory and set `restart_state` to the previous complete checkpoint.
Keep `start`, mesh, gate, physical parameters and forcing unchanged. Keep warmup
duration unchanged for a direct continuation. Duration can be extended if the
input series/raster cover the longer formal window. Mesh hashes, gate config,
boundary names/kinds/order/selected-edge hashes and stored physics are checked;
the stored provenance does not enforce equality of all forcing-file contents.
The caller must preserve forcing history up to the checkpoint. A state later
than the endpoint fails before writing solver results; one at the endpoint
finishes with zero steps. SWMM actor state is not saved, so coupled restart fails.

## Source and input hashes

`source_sha256` uses rule `nh-flood-2d/src-py-sha256/v1`. Start SHA-256 with the
UTF-8 bytes of that rule and one NUL byte. Enumerate every `.py` under `src/`
sorted by its POSIX path relative to `src/`. Append, for each file, its UTF-8
relative path and one NUL byte, its byte length as an unsigned 8-byte big-endian
integer, and its complete bytes. Relocation does not change the hash; file
names and contents do. This rule does not cover `main.py`, `tools/`, dependency
metadata, documentation or files outside `src/`.

`git_commit` and `git_dirty` supplement this identity when a `.git` entry is
present; otherwise both are JSON null. `input_sha256` hashes raw NE/NS,
boundary lines/series, rainfall raster/intervals, a supplied checkpoint and the
optional source INP. Keys are resolved absolute file paths. Configurations and
error messages can also contain local paths. Review these generated records
before sharing; outputs derived from restricted bathymetry remain restricted.
