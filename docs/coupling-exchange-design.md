# Surface / SWMM coupling

The current implementation uses one surface evolution engine in
`src/nh_flood_2d/core/solver_compact.py` and one continuous SWMM actor in
`core/coupled/pipe_1d.py`. `core/coupled/driver.py::solver_coupled` is a
compatibility entry point for already preprocessed inputs. The full CLI is:

```bash
uv run python main.py --domain path/to/domain.json --force path/to/force.json --pipe path/to/pipe.json
```

Supply your own mesh, rainfall, boundary files and SWMM INP; these placeholders
are not bundled case data. See [configuration](configuration.md) for every pipe
field and the [README](../README.md) for surface forcing and output formats.

## Clock and runtime files

The actor starts at `start - warmup_duration`, advances each exchange window,
and continues across the formal start without restarting. The final window may
be shorter than `coupling_interval`. Start timestamps must have whole-second
precision. The runtime checks the actual SWMM clock against the requested window.

`pipe_dir` contains `pipe.fdb` and a runtime copy of the source INP, with reports
and SWMM output alongside it. The source INP and runtime INP must be different
files. `FLOW_UNITS CMS` is required (m³/s and metres). Runtime preparation updates
start, report-start and end dates/times, sets existing junction surcharge-depth
columns to 1000 m and clears existing inflow baselines. Other INP forcing remains
part of the supplied model; review it when defining an experiment.

## Exchange

`preprocess/pipe.py` maps each node to a primary surface cell and additional
nearby cells. Outfalls receive the primary cell's water-surface elevation and
have zero lateral exchange flow. Junction drainage excludes land class 8.

For each eligible cell, drainage is
`min(0.85 * pi * 0.8 * h**1.5, h * area / window_seconds)` in m³/s, divided
equally among its unique junction relationships. Duplicate topology entries do
not multiply drainage. Junction overflow uses
`delta_h = SWMM_head - (rim + surface_depth)`. Positive head differences use
`dh = min(delta_h, 3)` and
`0.5 * min(0.85 * pi * 0.8 * dh**1.5, dh * area / window_seconds / primary_count)`.
Overflow enters the junction's primary surface cell; outfalls and land class 8
are excluded. The surface source has the opposite sign to SWMM lateral inflow.
The exchange pairs F32 source rounding with a corresponding junction correction.

Surface pipe sources are separate from prescribed discharge sources. When the
surface shared-water limiter reduces drainage, the coupling runtime reconciles
actual applied surface volumes with SWMM input for that same window. It checks
the paired volume balance before advancing SWMM. These exchange checks do not
establish the accuracy of a particular drainage-network calibration.

## Records and restart

The run manifest includes the pipe configuration and INP hash, cumulative SWMM
seconds, warmup/formal completed seconds, window clocks, requested/applied
volumes and surface/SWMM balance errors. The ordinary surface budget includes
`pipe_exchange_m3`. See [output and restart](solver-output.md) and the [pipe topology](pipe-topology.md).

Surface checkpoints do not save SWMM actor state. Coupled restart is rejected;
run the complete coupled warmup and formal interval in one invocation.
