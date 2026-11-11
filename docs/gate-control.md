# Independent gate controls

The optional force configuration models an idealized channel opening by lowering
gate-cell beds. It does not model a gate plate, a weir coefficient or an orifice.
The following coordinates are illustrative and must be replaced with points in
the mesh CRS. The final elevation is an input design parameter, not an elevation
inferred separately on each mesh.

```json
{
  "land_type": 11,
  "pond_land_type": 10,
  "enabled": true,
  "trigger_level": 3.07,
  "opening_seconds": 1800,
  "open_elevation": 1.3,
  "gates": [
    {
      "name": "gate_A",
      "inside_point": [824100.0, 841400.0],
      "control_point": [824080.0, 841400.0]
    }
  ]
}
```

Place this object under `gate` alongside `boundaries` and `rainfall`. All fields
are required. Unknown fields, including the removed `threshold` and
`level_boundary`, fail validation. A missing/null gate object or `enabled: false`
disables all control/bed-motion/accounting kernels and adds no step endpoints,
device synchronization or per-step readback.

## Mapping and validation

Cells of `land_type` are grouped by shared internal subedges, including hanging
subedges. Point contact alone does not join components. Each component must
contain exactly one configured inside point; each gate cell belongs to exactly
one gate. Gate names are unique. Both points must be inside the mesh. Control
points must lie outside every gate cell. Gate cells must not overlap a prescribed
level-boundary cell. `pond_land_type` must differ from `land_type` and select at
least one real mesh cell. Every named gate needs at least one internal shared
subedge to a pond cell and at least one to a real non-pond, non-gate cell.
`open_elevation` must be no higher than the input closed bed of every selected
gate cell. These checks also apply when controls are disabled.

Points on closed rectangle boundaries select the smallest real cell ID, matching
station sampling. Prefer points strictly inside cells and check their physical
location on every mesh. Geometric validation cannot establish that a control cell
is on the intended river: choose each control point in that gate's outside river,
using the same physical coordinates across resolutions. The resolved control ID
and complete gate-cell membership are recorded in `manifest.gate_control`.

## Control, time and numerical order

Each gate independently samples `bed + depth` on the device at the start of every
actual solver step. A cell is valid when `depth >= min_h`. A dry/near-dry control
cell does not trigger from its bed elevation. The first strictly greater level
locks the command clock; equality does not trigger. Subsequent falls in water
level do not close or reset the gate. No interpolation backdates a command, and
there is no fixed polling interval or new event-step schedule. Controls are
inactive during warmup and become eligible at formal time zero. Initial levels
and any restored state are applied before the initial formal check.

For each gate cell, the immutable closed bed is its input-grid F32 elevation:

`z(t) = z_closed + (open_elevation - z_closed) * clip((t-t_cmd)/opening_seconds, 0, 1)`.

The command is stored as two U32 clock words, with quantum `2**-40 s`. Age is
computed from the exact word difference, then converted to F32 for bed evaluation;
there is no accumulated F32 bed decrement. At full opening the bed is assigned
the F32 target elevation directly. Bed motion preserves depth and existing
discharge; it adds no water and does not reset momentum.

The enabled path uses the following explicit first-order split:

1. Initialization applies prescribed starting levels. Warmup initialization uses
   the original closed mesh; the control is inactive throughout warmup.
2. Choose the next step from the previous completed state's CFL and existing
   forcing/output endpoints. Check controls at `t_now`, then evaluate gate beds
   at `t_now` before reconstruction and fluxes. At `t_cmd`, the bed is still closed.
3. Recompute hanging-subedge pressure reconstruction from this bed/depth state;
   calculate inertial/friction fluxes and freeze donor supply scales.
4. Limit each unique shared subedge once. Integrate these actual fluxes times
   the actual step duration for gate/pond accounting. Update depths with the same
   shared fluxes, rain, bounded infiltration, positivity correction and boundary
   reservoir adjustment. Prescribed level cells take the step-end series value.
5. Measure the actually rounded pond inventory change in separate device
   offloads, then accumulate the diagnostics. This avoids algebraic cancellation
   of rounding error by the compiler.
6. Evaluate gate beds at `t_end`, preserving the updated depths and discharges.
   Compute the next CFL from this new completed state, then commit the clock.
   The next step's reconstruction and fluxes therefore use this same bed.
7. Existing output times sample the completed state's depth and bed. Opening
   completion does not create an additional step endpoint. Gate control stays
   entirely inside the graph; batch size does not change the polling rule.

A fixed-bed still lake stays still. Uniform lowering of a closed wet basin lowers
its water surface without flow. Local underwater lowering preserves water volume
but changes local water level, so it generates physical exchange in this model.
For example, after one of two adjacent 100 m² cells is lowered 0.1 m at unchanged
2 m depth, the next 0.1 s frictionless hydraulic step gives approximately
0.1962 m³/s toward that cell. These are different validation expectations.

## Records, replay and continuation

`manifest.gate_control` records the common parameters and, for each gate, its
cell IDs, closed bed elevations, control cell, exact command clock words,
`command_seconds`, `first_effective_flow_seconds`, opening-completion time and
cumulative exchange. `scheduled_fully_open_seconds` is `command + duration`;
`fully_open_seconds` remains null until the run has reached that time.
`first_fully_open_sample_seconds` separately records the first solver clock
at which the bed was actually evaluated as fully open. It can be later when
completion lies inside a step; F32 age/proportion rounding can also make a
near-end sample meet the full-opening condition slightly before that endpoint.

First effective flow means the **end of the first completed step with any nonzero
post-limiter volume across a gate–pond subedge, in either direction**. It does not
mean a trigger, nonzero raw momentum, river entry into the gate channel, or a
prescribed sampling threshold. Before such flow the value is null.

`BedTimeline`/`read_bed_at` reconstruct the opening function using the recorded
closed elevations and exact command clock, including when `bed.fdb` is a bed
restored partway through opening. New runs leave `bed_events` empty. Checkpoints
store the command/first-flow state, configuration identity, fixed-point simulation
clock and exact cumulative budget state. Restarts before, during or after opening
continue this state without relatching events or re-integrating earlier fluxes.
An output row at the restart instant is repeated, as with other solver outputs;
use rows strictly after that instant when concatenating process files.

## Exchange and pond inventory

The pond control volume is the complete set of mesh cells with
`pond_land_type` (10 in this experiment);
it is not a clipped polygon area. Gate-channel storage is reported separately.
Each actual subedge is counted once at its physical width, including coarse/fine
interfaces. Cumulative volumes are in m³:

* `pond_in_m3`, `pond_out_m3`, `pond_net_m3`: gate-to-pond, pond-to-gate and their
  difference, evaluated on gate–pond subedges.
* `river_in_m3`, `river_out_m3`: the other side of each gate channel, positive into
  or out of the channel, on gate–non-gate/non-pond subedges. These may differ from
  pond exchange while the channel fills or drains.
* `overtopping_in_m3`, `overtopping_out_m3`: the remaining boundary of the pond
  control volume, excluding all gate–pond subedges. This name describes the
  experiment's non-gate exchange category, not an imposed one-way flow law.

The initial pond inventory is taken after initial prescribed levels; warmup
exchange is included when warmup is run. `gate_balance.csv` records cumulative
gate and pond rows at every existing output time, so formal-period amounts can
also be computed by subtracting the time-zero row. It adds no physics endpoints.

The manifest's `pond_budget.residual_m3` is

`storage - initial_storage - (sum(gate pond_net) + overtopping_in - overtopping_out + rain - infiltration + positivity + pipe + boundary + rounding)`.

Each `gate_control.gates[].channel_budget` independently checks the gate-channel
control volume:

`storage - initial_storage = river_net - pond_net + rain - infiltration + positivity + pipe + boundary + rounding + residual`.

Here `river_net = river_in - river_out` is into the channel, and positive
`pond_net = pond_in - pond_out` leaves the channel. Gate rain and infiltration
are measured on gate cells, not inferred from the difference of river and pond
exchange. The channel source terms and initial inventory are accumulated and
checkpointed independently of the pond ledger. The gate rows in `gate_balance.csv`
contain these same source terms, `river_net_m3`, initial/final channel storage and
the channel residual. Pond rows contain the pond-control-volume terms. Both
inventories begin after initial prescribed levels and include any warmup.

`boundary` includes actual discharge and prescribed-level reservoir compensation;
`pipe` is net surface input from coupling. `positivity` is the actually applied
depth floor, excluding an intermediate floor overwritten by a level boundary.
`rounding` measures F32 depth-update inventory rounding, separate from the floor.
Bed motion contributes zero volume. Block diagnostics are F32, accumulated
exactly across time as integer multiples of `2**-149`, then reduced on the host.
The residual still exposes subedge/source assembly and block-sum roundoff;
exact time accumulation does not make the underlying F32 arithmetic exact.

## Verification

`tests/test_gate_config.py` checks configuration and component coverage.
`tests/test_gate_solver.py` uses production CPU kernels for control, still water,
local/uniform motion, dry/wet reversal, hanging interfaces, restart, replay,
post-limiter volume and pond closure. `tests/test_gate_compatibility.py` runs an
independent-process three-way comparison with exported source from 7648185,
without checking out or moving that commit. It compares F32 depth/q/bed bytes,
the full step trace, dt sequence, final budget and both diagnostic CSVs.

For the optional Hong Kong 1 h CPU comparison set `NH_GATE_HK_DOMAIN` and
`NH_GATE_HK_FORCE` to the saved configuration paths, then run that test. The
compatibility runner resolves all input paths read-only, uses zero warmup for
the 1 h smoke, and writes all products under the test output directory. The
disabled case maps every class-11 component automatically; those illustrative
control points are not experimental gate locations.
