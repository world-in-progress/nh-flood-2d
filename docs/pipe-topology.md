# Pipe-network configuration and topology

The implemented coupling is documented in
[Surface / SWMM coupling](coupling-exchange-design.md), with all JSON
fields in [Configuration](configuration.md). These references describe
the current shared surface engine and continuous actor.

`src/nh_flood_2d/preprocess/pipe.py` writes `pipe_dir/pipe.fdb`. Its tables are:

| Table | Contents |
| --- | --- |
| `Node` | Node index, name, x/y and outfall flag |
| `node_primary_ei` | Primary surface cell per node; 0 means no match |
| `node_count_per_ei` | Primary-node count per surface cell |
| `PipeTopo` | Additional associated surface cell IDs |
| `topo_ptr` | CSR offsets of length node count + 1 |
| `node_count` | Number of nodes |
| `node_rim` | Junction rim elevation (invert elevation + full depth) |

Node rows and CSR offsets are zero-based. Surface cell IDs are one-based with
0 reserved as the sentinel. Primary cells are not repeated in the additional
association list. The runtime validates topology shapes and uses unique
junction/cell relationships for drainage.

The old separate surface solver, hotstart window restarts, merged source
buffers and millilitre-volume conversion are superseded by the shared engine,
explicit exchange windows and rates in m³/s.
