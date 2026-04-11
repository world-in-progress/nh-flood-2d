# INP → SHP Conversion Tool Design

## Problem

The project needs a utility to convert SWMM `.inp` files into ESRI Shapefiles (`.shp`) for GIS visualization and analysis. The tool should automatically detect available feature types in the INP file and produce one Shapefile per geometry type.

## Approach

Use the `swmm-api` library's built-in GIS macros to parse INP files and extract GeoDataFrames, then export each as a Shapefile with user-specified CRS.

## Architecture

Single-file tool at `tools/inp2shp.py` serving as both an importable module and a CLI entry point.

### Dependencies

- `swmm-api` (new addition to `pyproject.toml`)
- `geopandas` (already present)
- `shapely` (already present)

### Core Flow

```
INP file path
  → SwmmInput.read_file(path)
  → Extract GeoDataFrames:
      - nodes_data_frame(inp)     → Point geometries (junctions, outfalls, storage, dividers)
      - links_data_frame(inp)     → LineString geometries (conduits, pumps, weirs, orifices)
      - subcatchments_data_frame(inp) → Polygon geometries (subcatchments)
  → Set CRS if --epsg provided
  → Export each non-empty GeoDataFrame to .shp
```

### Output Files

| File | Geometry | Contents |
|------|----------|----------|
| `nodes.shp` | Point | Junctions, Outfalls, Storage, Dividers |
| `links.shp` | LineString | Conduits, Pumps, Weirs, Orifices |
| `subcatchments.shp` | Polygon | Subcatchments |

Only non-empty layers are written. If an INP file has no subcatchments, `subcatchments.shp` is not created.

### CLI Interface

```bash
python tools/inp2shp.py <inp_file> [-o OUTPUT_DIR] [--epsg EPSG_CODE]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `inp_file` | Yes | — | Path to the SWMM .inp file |
| `-o/--output` | No | Same directory as inp_file | Output directory for shapefiles |
| `--epsg` | No | None (no CRS set) | EPSG code for coordinate reference system |

### Error Handling

- INP file not found → clear error message and exit code 1
- INP file has no spatial data (no COORDINATES section) → warning and exit
- Individual layer extraction failure → warn and continue with remaining layers
- Output directory creation → auto-create if it does not exist

### Example Usage

```bash
# Basic conversion
python tools/inp2shp.py ./data/network.inp

# With CRS and custom output directory
python tools/inp2shp.py ./data/network.inp -o ./gis_output/ --epsg 4547

# Via uv
uv run python tools/inp2shp.py ./data/network.inp --epsg 4326
```
