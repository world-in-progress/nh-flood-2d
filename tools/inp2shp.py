"""
Convert SWMM .inp files to ESRI Shapefiles.

Usage:
    python tools/inp2shp.py <inp_file> [-o OUTPUT_DIR] [--epsg EPSG_CODE]

Each geometry type (nodes, links, subcatchments) is exported as a separate
Shapefile. Only non-empty layers are written.

Requires: swmm-api, geopandas, shapely (all in project dependencies).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def inp_to_shp(
    inp_path: str | Path,
    output_dir: str | Path | None = None,
    epsg: int | None = None,
) -> list[Path]:
    """Convert a SWMM .inp file to Shapefiles.

    Args:
        inp_path: Path to the SWMM .inp file.
        output_dir: Directory for output shapefiles. Defaults to inp_path's parent.
        epsg: EPSG code for the coordinate reference system.

    Returns:
        List of created Shapefile paths.
    """
    from swmm_api.input_file import SwmmInput
    from swmm_api.input_file.macros.gis import set_crs
    from swmm_api.input_file.section_labels import (
        COORDINATES, VERTICES, POLYGONS,
    )

    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"INP file not found: {inp_path}")

    out = Path(output_dir) if output_dir else inp_path.parent
    out.mkdir(parents=True, exist_ok=True)

    inp = SwmmInput.read_file(str(inp_path))

    if epsg is not None:
        set_crs(inp, crs=f"EPSG:{epsg}")

    created: list[Path] = []

    # --- Nodes (Point) ---
    if COORDINATES in inp:
        gdf = _build_nodes(inp)
        if gdf is not None and not gdf.empty:
            p = out / "nodes.shp"
            _save(gdf, p, epsg)
            created.append(p)
            print(f"  ✓ nodes.shp  ({len(gdf)} features)")

    # --- Links (LineString) ---
    if VERTICES in inp or COORDINATES in inp:
        gdf = _build_links(inp)
        if gdf is not None and not gdf.empty:
            p = out / "links.shp"
            _save(gdf, p, epsg)
            created.append(p)
            print(f"  ✓ links.shp  ({len(gdf)} features)")

    # --- Subcatchments (Polygon) ---
    if POLYGONS in inp:
        gdf = _build_subcatchments(inp)
        if gdf is not None and not gdf.empty:
            p = out / "subcatchments.shp"
            _save(gdf, p, epsg)
            created.append(p)
            print(f"  ✓ subcatchments.shp  ({len(gdf)} features)")

    if not created:
        print("  ⚠ No spatial data found in INP file.")

    return created


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_nodes(inp):
    """Build a GeoDataFrame of all node types."""
    try:
        from swmm_api.input_file.macros.gis import nodes_geo_data_frame
        return nodes_geo_data_frame(inp, label_sep="_")
    except Exception as e:
        print(f"  ⚠ Failed to build nodes layer: {e}")
        return None


def _build_links(inp):
    """Build a GeoDataFrame of all link types."""
    try:
        from swmm_api.input_file.macros.gis import links_geo_data_frame
        return links_geo_data_frame(inp, label_sep="_")
    except Exception as e:
        print(f"  ⚠ Failed to build links layer: {e}")
        return None


def _build_subcatchments(inp):
    """Build a GeoDataFrame of all subcatchments."""
    try:
        from swmm_api.input_file.macros.gis import subcatchment_geo_data_frame
        return subcatchment_geo_data_frame(inp, label_sep="_")
    except Exception as e:
        print(f"  ⚠ Failed to build subcatchments layer: {e}")
        return None


def _save(gdf, path: Path, epsg: int | None):
    """Write a GeoDataFrame to Shapefile, setting CRS if needed."""
    if epsg is not None and gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg)
    gdf.to_file(str(path), driver="ESRI Shapefile", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SWMM .inp to ESRI Shapefiles",
    )
    parser.add_argument("inp_file", help="Path to the SWMM .inp file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: same as inp file)",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="EPSG code for the coordinate reference system",
    )

    args = parser.parse_args()

    print(f"Converting: {args.inp_file}")
    try:
        created = inp_to_shp(args.inp_file, args.output, args.epsg)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if created:
        print(f"\nDone. {len(created)} shapefile(s) written to: {created[0].parent}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
