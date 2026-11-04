# SWMM INP to Shapefile conversion

Run from the repository root after `uv sync --locked`:

```bash
uv run python tools/inp2shp.py path/to/network.inp -o path/to/shapefiles --epsg 4547
```

The INP must be supplied locally. This command is a template; the repository
does not bundle a case network. `--epsg` assigns a CRS to existing coordinates;
it does not reproject them.

| Argument | Default | Meaning |
| --- | --- | --- |
| `inp_file` | Required | Existing SWMM INP |
| `-o`, `--output` | INP parent directory | Created if missing |
| `--epsg` | None | EPSG code assigned on export |

`tools.inp2shp.inp_to_shp(inp_path, output_dir=None, epsg=None)` returns the list
of written `.shp` paths. The converter reads sections individually using
`swmm-api` and exports nonempty layers with GeoPandas. Node and link files are
named by section, for example `junctions.shp`, `outfalls.shp`, `conduits.shp`
and `pumps.shp`; polygons are written to `subcatchments.shp`. Shapefile sidecar
files are also produced; the driver can shorten attribute names to the
Shapefile field-name limit. Nodes need coordinates, links need endpoint coordinates
(and use completed vertices), and subcatchments need polygons.

A missing INP raises `FileNotFoundError`; no spatial layers prints a warning
and returns an empty list. Optional tag extraction failures are ignored;
other parsing/export exceptions propagate to the CLI's failure handler.
