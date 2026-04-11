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
    import geopandas as gpd
    from swmm_api.input_file import SwmmInput
    from swmm_api.input_file.macros.gis import (
        complete_vertices, set_crs, NODE_SECTIONS, LINK_SECTIONS,
        get_node_tags, get_link_tags, get_subcatchment_tags,
    )
    from swmm_api.input_file.section_labels import (
        COORDINATES, VERTICES, POLYGONS,
        SUBCATCHMENTS, SUBAREAS, INFILTRATION,
        XSECTIONS, LOSSES,
    )

    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"INP file not found: {inp_path}")

    out = Path(output_dir) if output_dir else inp_path.parent
    out.mkdir(parents=True, exist_ok=True)

    inp = SwmmInput.read_file(str(inp_path))

    crs_str = f"EPSG:{epsg}" if epsg is not None else None
    if crs_str is not None:
        set_crs(inp, crs=crs_str)

    created: list[Path] = []

    # --- Nodes (Point) — per-section to work around swmm-api bugs ---
    if COORDINATES in inp:
        node_geo = inp[COORDINATES].geo_series
        node_tags = _safe_call(lambda: get_node_tags(inp))
        for sec in NODE_SECTIONS:
            if sec not in inp:
                continue
            frame = inp[sec].frame
            if frame.empty:
                continue
            df = frame.rename(columns=lambda c: f"{sec}.{c}")
            df = df.join(node_geo)
            if node_tags is not None:
                df = df.join(node_tags)
            gdf = gpd.GeoDataFrame(df.dropna(subset=["geometry"]))
            if gdf.empty:
                continue
            p = out / f"{sec.lower()}.shp"
            _save(gdf, p, epsg)
            created.append(p)
            print(f"  ✓ {p.name}  ({len(gdf)} features)")

    # --- Links (LineString) — per-section, call complete_vertices first ---
    if COORDINATES in inp:
        complete_vertices(inp, crs_str)
        link_geo = inp[VERTICES].geo_series
        link_tags = _safe_call(lambda: get_link_tags(inp))
        xsec_frame = inp[XSECTIONS].frame if XSECTIONS in inp else None
        loss_frame = inp[LOSSES].frame if LOSSES in inp else None
        for sec in LINK_SECTIONS:
            if sec not in inp:
                continue
            frame = inp[sec].frame
            if frame.empty:
                continue
            df = frame.rename(columns=lambda c: f"{sec}.{c}")
            if xsec_frame is not None and not xsec_frame.empty:
                df = df.join(
                    xsec_frame.rename(columns=lambda c: f"XSECTIONS.{c}"),
                )
            if loss_frame is not None and not loss_frame.empty:
                df = df.join(
                    loss_frame.rename(columns=lambda c: f"LOSSES.{c}"),
                )
            df = df.join(link_geo)
            if link_tags is not None:
                df = df.join(link_tags)
            gdf = gpd.GeoDataFrame(df.dropna(subset=["geometry"]))
            if gdf.empty:
                continue
            p = out / f"{sec.lower()}.shp"
            _save(gdf, p, epsg)
            created.append(p)
            print(f"  ✓ {p.name}  ({len(gdf)} features)")

    # --- Subcatchments (Polygon) ---
    if POLYGONS in inp and SUBCATCHMENTS in inp:
        frame = inp[SUBCATCHMENTS].frame
        if not frame.empty:
            df = frame.rename(columns=lambda c: f"SUBCATCHMENTS.{c}")
            if SUBAREAS in inp and not inp[SUBAREAS].frame.empty:
                df = df.join(
                    inp[SUBAREAS].frame.rename(columns=lambda c: f"SUBAREAS.{c}"),
                )
            if INFILTRATION in inp and not inp[INFILTRATION].frame.empty:
                df = df.join(
                    inp[INFILTRATION].frame.rename(columns=lambda c: f"INFILTRATION.{c}"),
                )
            sub_tags = _safe_call(lambda: get_subcatchment_tags(inp))
            if sub_tags is not None:
                df = df.join(sub_tags)
            df = df.join(inp[POLYGONS].geo_series)
            gdf = gpd.GeoDataFrame(df.dropna(subset=["geometry"]))
            if not gdf.empty:
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

def _safe_call(fn):
    """Call *fn* and return None on any exception."""
    try:
        return fn()
    except Exception:
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
