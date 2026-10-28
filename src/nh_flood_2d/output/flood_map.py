import shutil
import rasterio
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import no_type_check
from rasterio.transform import from_origin

from ..input import DomainConfig, ForceConfig
from ..preprocess.force import MeshGeometry, RainField
from ..util import init_taichi, copy_to_taichi
from ..schema.feature import Ne, Ns, IndexLike
from ..core.output import BedTimeline, list_depth_frames, read_depth_frame, read_extrema
    
# Check if grid is too large for single Taichi field (limit near 2^31 elements, but practical limit lower)
# Using 20000x20000 as a safe threshold for tiling
TILE_SIZE = 4096


def _load_rainfall_plot_series(
    domain_cfg: DomainConfig, force_cfg: ForceConfig,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """Return clipped interval starts, mesh-mean mm/hr, and widths in seconds.

    Frame wall times derive from ``domain_cfg.start``. Matplotlib receives
    naive wall times for both rainfall and model series.
    """
    ne_db = fdb.ORM.load(domain_cfg.ne_fdb, from_file=True)
    ns_db = fdb.ORM.load(domain_cfg.ns_fdb, from_file=True)
    mesh = MeshGeometry.from_fdb(ne_db, ns_db)
    rain = RainField.load(force_cfg.rainfall, mesh, domain_cfg.epsg_code,
                          domain_cfg.start, 0, domain_cfg.duration)
    area = mesh.area[1:]
    pixel_area = np.bincount(rain.pixel_index[1:], weights=area,
                             minlength=rain.millimetres.shape[1])
    total_area = pixel_area.sum(dtype=np.float64)
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError('Rainfall plot requires positive finite mesh area')
    mean_mm = rain.millimetres @ pixel_area / total_area
    starts = np.maximum(rain.starts, 0.)
    ends = np.minimum(rain.ends, float(domain_cfg.duration))
    selected = ends > starts
    widths = ends[selected] - starts[selected]
    rates = mean_mm[selected] * (3600. / (rain.ends[selected] - rain.starts[selected]))
    dates = [(domain_cfg.start + timedelta(seconds=float(t))).replace(tzinfo=None)
             for t in starts[selected]]
    return dates, rates, widths


def _masked_depth(depth: np.ndarray, min_depth: float, invalid_data: float) -> np.ndarray:
    """Apply the display threshold to stored water depth without WSE subtraction."""
    return np.where(depth >= min_depth, depth, invalid_data).astype(np.float32)

def get_area_meta(ne_fdb_fn: str, ns_fdb_fn: str):
    """
    Compute area metadata (bounding box, resolution, element half-sizes) from ne.fdb and ns.fdb.
    """
    init_taichi()
    
    ne_fdb = fdb.ORM.load(ne_fdb_fn, from_file=True)
    ns_fdb = fdb.ORM.load(ns_fdb_fn, from_file=True)
    
    nes = ne_fdb[Ne][Ne]
    nss = ns_fdb[Ns][Ns]
    
    e_cnt = len(nes)
    
    exs = copy_to_taichi(nes.column.x, ti.f32, None)
    eys = copy_to_taichi(nes.column.y, ti.f32, None)
    sxs = copy_to_taichi(nss.column.x, ti.f32, None)
    sys = copy_to_taichi(nss.column.y, ti.f32, None)
    isl_data  = copy_to_taichi(ne_fdb[IndexLike]['isl_data'].column.index,  ti.i32, None)
    isl_ptr_l = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_l'].column.index, ti.i32, None)
    isl_ptr_b = copy_to_taichi(ne_fdb[IndexLike]['isl_ptr_b'].column.index, ti.i32, None)
    
    vr = ti.field(dtype=ti.f32, shape=())           # vertical resolution of cell in finest level
    hr = ti.field(dtype=ti.f32, shape=())           # horizontal resolution of cell in finest level
    bbox = ti.field(dtype=ti.f32, shape=4)          # xmin, ymin, xmax, ymax
    hws = ti.field(dtype=ti.f32, shape=e_cnt)       # half-width of each element
    hhs = ti.field(dtype=ti.f32, shape=e_cnt)       # half-height of each element
    
    @ti.kernel
    def compute_meta():
        bbox[0] = bbox[1] = 99999999.0
        bbox[2] = bbox[3] = 0.0
        vr[None] = hr[None] = 99999999.0
        
        for ei in range(1, ti.i32(e_cnt)):
            lsi0 = isl_data[isl_ptr_l[ei]]   # first left side
            lsi2 = isl_data[isl_ptr_b[ei]]   # first bottom side
            hw = ti.floor(exs[ei] - sxs[lsi0] + 0.5)
            hh = ti.floor(eys[ei] - sys[lsi2] + 0.5)
            
            # Update resolution
            ti.atomic_min(hr[None], hw * 2.0)
            ti.atomic_min(vr[None], hh * 2.0)
            
            # Update bbox
            xmin = exs[ei] - hw
            xmax = exs[ei] + hw
            ymin = eys[ei] - hh
            ymax = eys[ei] + hh
            ti.atomic_min(bbox[0], xmin)
            ti.atomic_min(bbox[1], ymin)
            ti.atomic_max(bbox[2], xmax)
            ti.atomic_max(bbox[3], ymax)
            
            # Store half-sizes
            hws[ei] = hw
            hhs[ei] = hh
            
    compute_meta()
    return bbox.to_numpy(), (vr.to_numpy(), hr.to_numpy()), hws.to_numpy()[1:], hhs.to_numpy()[1:]  # skip virtual element 0

def generate_flood_map(cfg: DomainConfig):
    """
    Generate GeoTIFF flood maps from 900-second depth frames.
    """
    init_taichi()
    
    no_data = -9999.0  # nodata value for dry pixels or outside domain
    
    min_h = cfg.min_h
    ne_fdb_fn = cfg.ne_fdb
    ns_fdb_fn = cfg.ns_fdb
    frames_dir = Path(cfg.domain_dir) / 'output' / 'frames'
    epsg_code = cfg.epsg_code
    output_dir = cfg.flood_map_dir
    
    # Check input files
    ne_fdb_path = Path(ne_fdb_fn)
    ns_fdb_path = Path(ns_fdb_fn)
    if not ne_fdb_path.exists() or not ns_fdb_path.exists() or not frames_dir.exists():
        raise FileNotFoundError(f'Mesh or depth frames not found: {ne_fdb_path}, {ns_fdb_path}, {frames_dir}')
    
    # Clean output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate area metadata (bounding box, resolution, element half-sizes) from ne.fdb and ns.fdb
    (min_x, min_y, max_x, max_y), (vr_np, hr_np), half_widths, half_heights = get_area_meta(str(ne_fdb_path), str(ns_fdb_path))
    print(f'Bounding Box: ({min_x}, {min_y}) to ({max_x}, {max_y})')
    print(f'Vertical Pixel Resolution: {vr_np} m, Horizontal Pixel Resolution: {hr_np} m')
    
    # Calculate flood map dimensions
    width = int(np.ceil((max_x - min_x) / hr_np))
    height = int(np.ceil((max_y - min_y) / vr_np))
    print(f'Flood Map Size: {width} x {height}')
        
    # Load mesh data
    ne_fdb = fdb.ORM.load(str(ne_fdb_path), from_file=True)
    nes = ne_fdb[Ne][Ne]
    e_cnt = len(nes)
    
    if e_cnt == 0:
        print('No elements to process. Check if ne.fdb is correct and has elements.')
        return
    
    # Prepared Taichi Fields for elements
    x_field = copy_to_taichi(nes.column.x[1:], ti.f32, None)
    y_field = copy_to_taichi(nes.column.y[1:], ti.f32, None)
    
    # Element dimensions fields
    hw_field = copy_to_taichi(half_widths, ti.f32, None)
    hh_field = copy_to_taichi(half_heights, ti.f32, None)
    
    real_e_cnt = e_cnt - 1  # skip virtual element 0

    # Pre-allocate per-timestep Taichi fields once and reuse them to avoid
    # repeated GPU allocations when processing long depth-frame sequences.
    depth_field = ti.field(dtype=ti.f32, shape=real_e_cnt)
    tile_field = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))

    @ti.kernel
    @no_type_check
    def rasterize_tile_kernel(
        x: ti.template(), y: ti.template(),
        hw: ti.template(), hh: ti.template(),
        depth: ti.template(),
        tile: ti.template(), tile_min_x: float, tile_max_y: float,
        hr: float, vr: float,
        t_rows: int, t_cols: int
    ):
        # Loop over all elements
        for i in x:
            px = x[i]
            py = y[i]
            phw = hw[i]
            phh = hh[i]

            # Element bounds in world coords
            e_min_x = px - phw
            e_max_x = px + phw
            e_min_y = py - phh
            e_max_y = py + phh

            # Check if element overlaps with this tile (AABB check)
            # Tile: [tile_min_x, tile_max_x] x [tile_min_y, tile_max_y]
            tile_max_x = tile_min_x + t_cols * hr
            tile_min_y = tile_max_y - t_rows * vr
            if (e_max_x >= tile_min_x and e_min_x < tile_max_x and
                e_max_y >= tile_min_y and e_min_y < tile_max_y):
                # Assign pixel centres with half-open cell bounds; neighbouring
                # cells must never race to write one boundary pixel.
                start_col_f = ti.ceil((e_min_x - tile_min_x) / hr - 0.5)
                end_col_f   = (ti.ceil((e_max_x - tile_min_x) / hr - 0.5) - 1)
                start_row_f = ti.ceil((tile_max_y - e_max_y) / vr - 0.5)
                end_row_f   = (ti.ceil((tile_max_y - e_min_y) / vr - 0.5) - 1)

                col_start = int(start_col_f)
                col_end   = int(end_col_f)
                row_start = int(start_row_f)
                row_end   = int(end_row_f)

                # Clamp to tile bounds (0..t_cols-1, 0..t_rows-1)
                col_start = ti.max(0, col_start)
                col_end   = ti.min(t_cols - 1, col_end)
                row_start = ti.max(0, row_start)
                row_end   = ti.min(t_rows - 1, row_end)

                # Splat the depth value
                d = depth[i]
                for r in range(row_start, row_end + 1):
                    for c in range(col_start, col_end + 1):
                        tile[r, c] = d

    # Frames carry depth only; the run start supplies each GeoTIFF's wall time.
    for idx, (seconds, frame_path) in enumerate(list_depth_frames(frames_dir.parent)):
        print(f'Processing depth frame: {frame_path} ...')
        
        # Preserve wall-time names for the downstream video label.
        stamp = (cfg.start + timedelta(seconds=seconds)).strftime('%Y%m%d-%H%M%S')
        out_name = f'flood_map_{idx}_{stamp}.tif'
        out_path = output_dir / out_name

        depth = read_depth_frame(frame_path, expected_count=e_cnt)
        depth_field.from_numpy(_masked_depth(depth[1:], min_h, no_data))
        
        # Calculate geotransform
        transform = from_origin(min_x, max_y, hr_np, vr_np)
        
        # Use tiled processing for large grids
        # Initialize the output file with nodata
        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=height, width=width, count=1,
            dtype=rasterio.float32,
            crs=rasterio.crs.CRS.from_epsg(epsg_code),
            transform=transform,
            nodata=no_data,
            compress='lzw', # good for sparse data
            tiled=True,
            blockxsize=512, blockysize=512
        ) as dst:
            # Iterate through tiles
            for row_off in range(0, height, TILE_SIZE):
                for col_off in range(0, width, TILE_SIZE):
                    # Current Tile Size (might be smaller at edges)
                    current_h = min(TILE_SIZE, height - row_off)
                    current_w = min(TILE_SIZE, width - col_off)
                    
                    # Tile Bounds in World Coordinates
                    # Top-Left of this tile
                    # Global Row `row_off` corresponds to y = max_y - row_off * vr
                    tile_top_y = max_y - row_off * vr_np
                    # Global Col `col_off` corresponds to x = min_x + col_off * hr
                    tile_left_x = min_x + col_off * hr_np
                    
                    # Reset Tile Field
                    tile_field.fill(-9999.0)
                    
                    # Run Kernel
                    rasterize_tile_kernel(
                        x_field, y_field, hw_field, hh_field, depth_field, tile_field,
                        tile_left_x, tile_top_y, float(hr_np), float(vr_np), current_h, current_w
                    )
                    
                    # Write to GeoTIFF
                    # Extract the valid part of tile_data [0:current_h, 0:current_w]
                    # rasterio window: ((row_start, row_stop), (col_start, col_stop))
                    tile_data = tile_field.to_numpy()
                    dt = tile_data[0:current_h, 0:current_w]
                    window = rasterio.windows.Window(col_off, row_off, current_w, current_h)
                    dst.write(dt, 1, window=window)
            
        print(f'    Saved flood map to: {out_path}')

def generate_max_inundation_extent_map(cfg: DomainConfig, min_depth: float = 0.05, invalid_data: float = -9999.0, is_absolute: bool = False):
    """
    Generate a GeoTIFF from the per-step maximum water depth.

    Maximum depth is accumulated by the solver on every completed step.
    Pixels whose maximum depth never reaches `min_depth` are set to `invalid_data` (nodata).
    Pixels that do reach `min_depth` retain their maximum depth value.

    If `is_absolute` is True, the output is a binary int8 map:
      - 1 for wet pixels (max depth >= min_depth)
      - 0 for dry pixels (max depth < min_depth)
      - `invalid_data` for pixels outside the domain (e.g., -128)

    Uses the same Taichi-accelerated tiled rasterization approach as generate_flood_map.
    Rasterizes the stored maximum once.
    Output: <domain_dir>/max_inundation/max_inundation.tif
    """
    init_taichi()

    ne_fdb_fn = cfg.ne_fdb
    ns_fdb_fn = cfg.ns_fdb
    extrema_path = Path(cfg.domain_dir) / 'output' / 'extrema.fdb'
    epsg_code = cfg.epsg_code
    output_path = Path(cfg.max_inundation_dir) / 'max_inundation.tif'

    # Check input files
    ne_fdb_path = Path(ne_fdb_fn)
    ns_fdb_path = Path(ns_fdb_fn)
    if not ne_fdb_path.exists() or not ns_fdb_path.exists() or not extrema_path.exists():
        raise FileNotFoundError(f'Mesh or extrema not found: {ne_fdb_path}, {ns_fdb_path}, {extrema_path}')

    # Calculate area metadata
    (min_x, min_y, max_x, max_y), (vr_np, hr_np), half_widths, half_heights = get_area_meta(str(ne_fdb_path), str(ns_fdb_path))
    print(f'Bounding Box: ({min_x}, {min_y}) to ({max_x}, {max_y})')
    print(f'Vertical Pixel Resolution: {vr_np} m, Horizontal Pixel Resolution: {hr_np} m')

    width = int(np.ceil((max_x - min_x) / hr_np))
    height = int(np.ceil((max_y - min_y) / vr_np))
    print(f'Max Inundation Map Size: {width} x {height}')

    # Load mesh data into Taichi fields
    ne_fdb = fdb.ORM.load(str(ne_fdb_path), from_file=True)
    nes = ne_fdb[Ne][Ne]
    e_cnt = len(nes)

    if e_cnt == 0:
        print('No elements to process. Check if ne.fdb is correct and has elements.')
        return

    x_field  = copy_to_taichi(nes.column.x[1:], ti.f32, None)
    y_field  = copy_to_taichi(nes.column.y[1:], ti.f32, None)
    hw_field = copy_to_taichi(half_widths, ti.f32, None)
    hh_field = copy_to_taichi(half_heights, ti.f32, None)

    real_e_cnt = e_cnt - 1  # skip virtual element 0
    max_depth_field = ti.field(dtype=ti.f32, shape=real_e_cnt)
    max_depth_field.from_numpy(read_extrema(extrema_path, expected_count=e_cnt).max_depth[1:])

    # Pass 2: threshold maximum depth once and rasterize once
    depth_field = ti.field(dtype=ti.f32, shape=real_e_cnt)

    if is_absolute:
        @ti.kernel
        @no_type_check
        def compute_depth_max(maximum: ti.template(), depth: ti.template()):
            for i in maximum:
                if maximum[i] >= min_depth:
                    depth[i] = 1.0
                else:
                    depth[i] = 0.0
    else:
        @ti.kernel
        @no_type_check
        def compute_depth_max(maximum: ti.template(), depth: ti.template()):
            for i in maximum:
                depth[i] = maximum[i]
                if maximum[i] < min_depth:
                    depth[i] = invalid_data

    compute_depth_max(max_depth_field, depth_field)

    tile_field = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))

    @ti.kernel
    @no_type_check
    def rasterize_tile_kernel(
        x: ti.template(), y: ti.template(),
        hw: ti.template(), hh: ti.template(),
        depth: ti.template(),
        tile: ti.template(), tile_min_x: float, tile_max_y: float,
        hr: float, vr: float,
        t_rows: int, t_cols: int
    ):
        for i in x:
            px = x[i]
            py = y[i]
            phw = hw[i]
            phh = hh[i]

            e_min_x = px - phw
            e_max_x = px + phw
            e_min_y = py - phh
            e_max_y = py + phh

            tile_max_x = tile_min_x + t_cols * hr
            tile_min_y = tile_max_y - t_rows * vr
            if (e_max_x >= tile_min_x and e_min_x < tile_max_x and
                    e_max_y >= tile_min_y and e_min_y < tile_max_y):
                start_col_f = ti.ceil((e_min_x - tile_min_x) / hr - 0.5)
                end_col_f   = (ti.ceil((e_max_x - tile_min_x) / hr - 0.5) - 1)
                start_row_f = ti.ceil((tile_max_y - e_max_y) / vr - 0.5)
                end_row_f   = (ti.ceil((tile_max_y - e_min_y) / vr - 0.5) - 1)

                col_start = ti.max(0, int(start_col_f))
                col_end   = ti.min(t_cols - 1, int(end_col_f))
                row_start = ti.max(0, int(start_row_f))
                row_end   = ti.min(t_rows - 1, int(end_row_f))

                d = depth[i]
                for r in range(row_start, row_end + 1):
                    for c in range(col_start, col_end + 1):
                        tile[r, c] = d

    result = np.full((height, width), -128 if is_absolute else invalid_data, dtype=np.int8 if is_absolute else np.float32)

    print('Rasterizing max depth ...')
    for row_off in range(0, height, TILE_SIZE):
        for col_off in range(0, width, TILE_SIZE):
            current_h = min(TILE_SIZE, height - row_off)
            current_w = min(TILE_SIZE, width - col_off)

            tile_top_y  = max_y - row_off * vr_np
            tile_left_x = min_x + col_off * hr_np

            tile_field.fill(invalid_data)
            rasterize_tile_kernel(
                x_field, y_field, hw_field, hh_field, depth_field, tile_field,
                tile_left_x, tile_top_y, float(hr_np), float(vr_np), current_h, current_w
            )

            tile_np = tile_field.to_numpy()[:current_h, :current_w]
            if is_absolute:
                tile_np = tile_np.astype(np.int8)
            result[row_off:row_off + current_h, col_off:col_off + current_w] = tile_np

    # Apply minimum depth threshold: pixels below min_depth → invalid_data
    if is_absolute:
        result = result.astype(np.int8)
    else:
        result = np.where(result >= min_depth, result, invalid_data).astype(np.float32)

    # Write single GeoTIFF
    out_dtype = rasterio.int8 if is_absolute else rasterio.float32
    out_nodata = -128 if is_absolute else invalid_data
    transform = from_origin(min_x, max_y, hr_np, vr_np)
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height, width=width, count=1,
        dtype=out_dtype,
        crs=rasterio.crs.CRS.from_epsg(epsg_code),
        transform=transform,
        nodata=out_nodata,
        compress='lzw',
        tiled=True,
        blockxsize=512, blockysize=512
    ) as dst:
        dst.write(result, 1)

    print(f'Saved max inundation map to: {output_path}')

def generate_f1_score_map(
    truth_path: str,
    pred_path: str,
    output_path: str,
    invalid_data: int = -128,
) -> dict[str, float]:
    """
    Generate a spatial F1-Score classification map from two binary inundation extent maps.

    Both inputs should be binary int8 GeoTIFFs produced by generate_max_inundation_extent_map
    with is_absolute=True (values: 0=dry, 1=wet, invalid_data=outside domain).

    The prediction raster is reprojected to the truth raster grid (nearest-neighbour) before
    comparison, so the two inputs do not need to share the same resolution or extent.

    Output pixel categories (written as classified int8 GeoTIFF with embedded colormap):
      - 1  → TP: both truth and prediction flooded  (blue)
      - 2  → FP: prediction flooded, truth dry      (red)
      - 3  → FN: truth flooded, prediction dry       (gray)
      - 0  → TN: both dry                            (transparent)
      - invalid_data → either input is outside its domain

    Parameters
    ----------
    truth_path   : Path to the truth binary inundation GeoTIFF (treated as ground truth).
    pred_path    : Path to the prediction binary inundation GeoTIFF.
    output_path  : Destination path for the classified GeoTIFF.
    invalid_data : Nodata value used in both input rasters (default -128 for int8).

    Returns
    -------
    dict with keys 'f1', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject as rio_reproject

    # ── Read truth raster ──────────────────────────────────────────────────────
    with rasterio.open(truth_path) as src:
        truth = src.read(1).astype(np.int16)
        truth_transform = src.transform
        truth_crs = src.crs
        truth_shape = src.shape

    # ── Reproject prediction onto the truth grid ───────────────────────────────
    pred_aligned = np.full(truth_shape, invalid_data, dtype=np.int16)
    with rasterio.open(pred_path) as src:
        rio_reproject(
            source=rasterio.band(src, 1),
            destination=pred_aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=truth_transform,
            dst_crs=truth_crs,
            resampling=Resampling.nearest,
            src_nodata=invalid_data,
            dst_nodata=invalid_data,
        )

    # ── Classify pixels ────────────────────────────────────────────────────────
    invalid_mask = (truth == invalid_data) | (pred_aligned == invalid_data)
    tp_mask = (truth == 1) & (pred_aligned == 1) & ~invalid_mask
    fp_mask = (truth == 0) & (pred_aligned == 1) & ~invalid_mask
    fn_mask = (truth == 1) & (pred_aligned == 0) & ~invalid_mask
    tn_mask = (truth == 0) & (pred_aligned == 0) & ~invalid_mask

    result = np.zeros(truth_shape, dtype=np.int8)
    result[tp_mask] = 1
    result[fp_mask] = 2
    result[fn_mask] = 3
    result[tn_mask] = 0
    result[invalid_mask] = np.int8(invalid_data)

    # ── Compute metrics ────────────────────────────────────────────────────────
    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tn = int(tn_mask.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f'F1: {f1:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}')
    print(f'TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}')

    # ── Write classified GeoTIFF with embedded colormap ────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=truth_shape[0], width=truth_shape[1], count=1,
        dtype=rasterio.int8,
        crs=truth_crs,
        transform=truth_transform,
        nodata=invalid_data,
        compress='lzw',
        tiled=True,
        blockxsize=512, blockysize=512,
    ) as dst:
        dst.write(result, 1)
        # RGBA colormap embedded in the GeoTIFF (readable by QGIS, ArcGIS, etc.)
        dst.write_colormap(1, {
            0: (255, 255, 255, 0),    # TN  — transparent white
            1: (70,  130, 180, 255),  # TP  — steel blue
            2: (205,  92,  92, 255),  # FP  — indian red
            3: (169, 169, 169, 255),  # FN  — gray
        })

    print(f'Saved F1-score map to: {out_path}')

    return {'f1': f1, 'precision': precision, 'recall': recall,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def plot_spatial_mae_curve(
    cfg_ref: DomainConfig,
    cfg_cmp: DomainConfig,
    force_cfg: ForceConfig | None = None,
    min_depth: float = 0.05,
    output_path: str | None = None,
    show: bool = True,
) -> list[float]:
    """
    Plot Spatial Mean Absolute Error (MAE) of Water Surface Elevation (WSE)
    between two simulation configurations across shared depth-frame times.

    For every timestep t that both configs have produced, computes:

        MAE(t) = (1 / N_t) * sum |WSE_ref(t) - WSE_cmp(t)|

    where N_t is the number of raster pixels simultaneously wet (depth >= min_depth)
    in both grids.  The result is a 1-D time-series curve.

    If *force_cfg* is provided, an area-weighted rainfall-intensity hyetograph
    from its raster bands and interval CSV is overlaid on an inverted secondary
    Y-axis (meteorological convention: bars grow downward from the top).

    Uses Taichi GPU-accelerated tiled rasterisation (same TILE_SIZE as the rest of
    this module) and atomic double-precision reduction for the MAE accumulation.

    Parameters
    ----------
    cfg_ref      : Reference DomainConfig  (e.g., high-resolution 4 m baseline)
    cfg_cmp      : Comparison DomainConfig (e.g., MRCG)
    force_cfg    : Optional ForceConfig supplying rainfall raster and intervals.
    min_depth    : Minimum water depth (m) threshold for a cell to be considered wet.
    output_path  : If given, the figure is saved to this file path.
    show         : If True, plt.show() is called after rendering.

    Returns
    -------
    List of per-timestep spatial MAE values (metres), in chronological order.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    init_taichi()

    ne_path_ref  = Path(cfg_ref.ne_fdb)
    ns_path_ref  = Path(cfg_ref.ns_fdb)
    ne_path_cmp  = Path(cfg_cmp.ne_fdb)
    ns_path_cmp  = Path(cfg_cmp.ns_fdb)
    output_ref = Path(cfg_ref.domain_dir) / 'output'
    output_cmp = Path(cfg_cmp.domain_dir) / 'output'

    for p in [ne_path_ref, ns_path_ref, ne_path_cmp, ns_path_cmp, output_ref, output_cmp]:
        if not p.exists():
            raise FileNotFoundError(f'Required path not found: {p}')

    # ── Area metadata for both grids ───────────────────────────────────────────
    (xmin_r, ymin_r, xmax_r, ymax_r), (vr_r, hr_r), hws_r, hhs_r = get_area_meta(str(ne_path_ref), str(ns_path_ref))
    (xmin_c, ymin_c, xmax_c, ymax_c), (vr_c, hr_c), hws_c, hhs_c = get_area_meta(str(ne_path_cmp), str(ns_path_cmp))

    # Common spatial domain: bounding-box intersection, finest resolution
    gmin_x = float(max(xmin_r, xmin_c))
    gmin_y = float(max(ymin_r, ymin_c))
    gmax_x = float(min(xmax_r, xmax_c))
    gmax_y = float(min(ymax_r, ymax_c))
    if gmin_x >= gmax_x or gmin_y >= gmax_y:
        raise ValueError('The bounding boxes of cfg_ref and cfg_cmp do not overlap.')

    g_hr = float(min(hr_r, hr_c))
    g_vr = float(min(vr_r, vr_c))
    g_width  = int(np.ceil((gmax_x - gmin_x) / g_hr))
    g_height = int(np.ceil((gmax_y - gmin_y) / g_vr))
    print(f'Common raster grid: {g_width} x {g_height} px  @  {g_hr} m x {g_vr} m')

    # ── Load element geometry into Taichi fields ───────────────────────────────
    ne_fdb_r = fdb.ORM.load(str(ne_path_ref), from_file=True)
    nes_r    = ne_fdb_r[Ne][Ne]
    ne_fdb_c = fdb.ORM.load(str(ne_path_cmp), from_file=True)
    nes_c    = ne_fdb_c[Ne][Ne]

    ex_r  = copy_to_taichi(nes_r.column.x[1:], ti.f32, None)
    ey_r  = copy_to_taichi(nes_r.column.y[1:], ti.f32, None)
    ehw_r = copy_to_taichi(hws_r,               ti.f32, None)
    ehh_r = copy_to_taichi(hhs_r,               ti.f32, None)

    ex_c  = copy_to_taichi(nes_c.column.x[1:], ti.f32, None)
    ey_c  = copy_to_taichi(nes_c.column.y[1:], ti.f32, None)
    ehw_c = copy_to_taichi(hws_c,               ti.f32, None)
    ehh_c = copy_to_taichi(hhs_c,               ti.f32, None)

    # Two tile buffers (one per mesh) for tiled GPU rasterisation
    tile_ref = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))
    tile_cmp = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))

    # Pre-allocate WSE fields once; updated each timestep with from_numpy().
    # Avoids calling ti.field() inside the loop which exhausts Metal's snode limit.
    h_r_field = ti.field(dtype=ti.f32, shape=len(nes_r) - 1)
    h_c_field = ti.field(dtype=ti.f32, shape=len(nes_c) - 1)
    depth_r_field = ti.field(dtype=ti.f32, shape=len(nes_r) - 1)
    depth_c_field = ti.field(dtype=ti.f32, shape=len(nes_c) - 1)
    bed_r = BedTimeline(output_ref, expected_count=len(nes_r))
    bed_c = BedTimeline(output_cmp, expected_count=len(nes_c))

    # ── Taichi kernels ─────────────────────────────────────────────────────────

    @ti.kernel
    @no_type_check
    def rasterize_wse(
        x: ti.template(), y: ti.template(),
        hw: ti.template(), hh: ti.template(),
        depth: ti.template(), h: ti.template(),
        tile: ti.template(),
        tile_min_x: float, tile_max_y: float,
        hr: float, vr: float,
        t_rows: int, t_cols: int,
    ):
        """Rasterise WSE (h) onto *tile*; pixels where depth < min_depth → nodata."""
        for i in x:
            px  = x[i];  py  = y[i]
            phw = hw[i]; phh = hh[i]

            e_min_x = px - phw;  e_max_x = px + phw
            e_min_y = py - phh;  e_max_y = py + phh

            tile_max_x = tile_min_x + t_cols * hr
            tile_min_y = tile_max_y - t_rows * vr

            if (e_max_x >= tile_min_x and e_min_x < tile_max_x and
                    e_max_y >= tile_min_y and e_min_y < tile_max_y):
                wse = h[i]
                if depth[i] < min_depth:
                    wse = -9999.0

                c0 = ti.max(0, int(ti.ceil((e_min_x - tile_min_x) / hr - 0.5)))
                c1 = ti.min(t_cols - 1, int((ti.ceil((e_max_x - tile_min_x) / hr - 0.5) - 1)))
                r0 = ti.max(0, int(ti.ceil((tile_max_y - e_max_y) / vr - 0.5)))
                r1 = ti.min(t_rows - 1, int((ti.ceil((tile_max_y - e_min_y) / vr - 0.5) - 1)))

                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        tile[r, c] = wse

    frames_ref = {cfg_ref.start + timedelta(seconds=seconds): (seconds, path)
                  for seconds, path in list_depth_frames(output_ref)}
    frames_cmp = {cfg_cmp.start + timedelta(seconds=seconds): (seconds, path)
                  for seconds, path in list_depth_frames(output_cmp)}
    common = sorted(set(frames_ref) & set(frames_cmp))
    if not common:
        raise ValueError('No common depth-frame times found between cfg_ref and cfg_cmp.')
    print(f'Processing {len(common)} shared depth-frame times ...')

    ref_label = Path(cfg_ref.domain_dir).name
    cmp_label = Path(cfg_cmp.domain_dir).name

    # ── Optional: load rainfall for hyetograph overlay ─────────────────────────
    rain_dts: list[datetime] | None = None
    rain_rates: np.ndarray | None = None
    rain_widths: np.ndarray | None = None
    if force_cfg is not None:
        rain_dts, rain_rates, rain_widths = _load_rainfall_plot_series(cfg_ref, force_cfg)

    # ── Per-timestep spatial MAE computation ───────────────────────────────────
    mae_list:  list[float]    = []
    time_axis: list[datetime] = []

    for instant in common:
        seconds_r, path_r = frames_ref[instant]
        seconds_c, path_c = frames_cmp[instant]
        depth_r = read_depth_frame(path_r, expected_count=len(nes_r))
        depth_c = read_depth_frame(path_c, expected_count=len(nes_c))
        working_bed_r = bed_r.at(seconds_r)
        working_bed_c = bed_c.at(seconds_c)

        h_r_field.from_numpy((working_bed_r[1:] + depth_r[1:]).astype(np.float32))
        h_c_field.from_numpy((working_bed_c[1:] + depth_c[1:]).astype(np.float32))
        depth_r_field.from_numpy(depth_r[1:])
        depth_c_field.from_numpy(depth_c[1:])

        mae_sum_acc = 0.0
        mae_cnt_acc = 0

        for row_off in range(0, g_height, TILE_SIZE):
            for col_off in range(0, g_width, TILE_SIZE):
                t_rows  = min(TILE_SIZE, g_height - row_off)
                t_cols  = min(TILE_SIZE, g_width  - col_off)
                top_y   = gmax_y - row_off * g_vr
                left_x  = gmin_x + col_off * g_hr

                tile_ref.fill(-9999.0)
                tile_cmp.fill(-9999.0)

                rasterize_wse(
                    ex_r, ey_r, ehw_r, ehh_r, depth_r_field, h_r_field,
                    tile_ref, left_x, top_y, g_hr, g_vr, t_rows, t_cols,
                )
                rasterize_wse(
                    ex_c, ey_c, ehw_c, ehh_c, depth_c_field, h_c_field,
                    tile_cmp, left_x, top_y, g_hr, g_vr, t_rows, t_cols,
                )

                # MAE reduction on CPU; tiles are small (≤ TILE_SIZE²) so this is fast.
                # Avoids ti.f64 / ti.i64 which are unsupported on Metal (macOS GPU).
                tr_np = tile_ref.to_numpy()[:t_rows, :t_cols]
                tc_np = tile_cmp.to_numpy()[:t_rows, :t_cols]
                mask  = (tr_np > -9000.0) & (tc_np > -9000.0)
                if mask.any():
                    mae_sum_acc += float(np.sum(np.abs(tr_np[mask] - tc_np[mask])))
                    mae_cnt_acc += int(mask.sum())

        mae = mae_sum_acc / mae_cnt_acc if mae_cnt_acc > 0 else float('nan')
        mae_list.append(mae)
        time_axis.append(instant.replace(tzinfo=None))
        print(f'  {instant.isoformat()}: MAE = {mae:.4f} m  (N_wet = {mae_cnt_acc} px)')

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(time_axis, mae_list, color='steelblue', linewidth=2, label='Spatial MAE (m)')
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.set_ylabel('Spatial MAE of WSE (m)', color='steelblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()

    if rain_dts is not None and rain_rates is not None and rain_widths is not None:
        if rain_dts:
            ax2 = ax1.twinx()
            ax2.bar(
                rain_dts, rain_rates,
                width=rain_widths / 86400., align='edge',
                alpha=0.35, color='dimgray', label='Rainfall (mm/hr)',
            )
            ax2.set_ylabel('Rainfall (mm/hr)', color='dimgray', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='dimgray')
            ax2.invert_yaxis()   # meteorological convention: bars grow downward from top
            ax2.legend(loc='upper right')

    ax1.set_title(f'Spatial MAE of WSE — {ref_label}  vs  {cmp_label}  (min_depth = {min_depth} m)', fontweight='bold')
    ax1.legend(loc='upper left')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to: {output_path}')
    if show:
        plt.show()
    plt.close(fig)

    return mae_list


def generate_flood_video(
    cfg: DomainConfig,
    output_path: str,
    interval_second: float = 0.2,
) -> None:
    """
    Generate an MP4 animation from all GeoTIFF flood maps with depth-dependent coloring.

    A two-pass approach is used:
      Pass 1 — scan all TIF frames to determine the global min/max water depth.
      Pass 2 — render each frame with a linear colour gradient from light-blue
               (shallow) to dark-blue (deep).  Dry / noData pixels remain black.

    The current simulation timestamp is drawn in the top-right corner of every frame.

    Parameters
    ----------
    cfg             : DomainConfig — used to locate the flood_maps directory.
    output_path     : Destination file path for the output MP4 (e.g. 'output/flood.mp4').
    interval_second : Duration of each frame in seconds (default 0.2 → 5 fps).
    """
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    from datetime import datetime

    flood_map_dir = Path(cfg.flood_map_dir)
    tif_paths = sorted(
        flood_map_dir.glob('flood_map_*.tif'),
        key=lambda p: int(p.stem.split('_')[2]),  # sort by embedded frame index
    )

    if not tif_paths:
        print(f'No flood map TIFFs found in {flood_map_dir}. Run generate_flood_map first.')
        return

    fps = 1.0 / interval_second

    # Read the first frame to determine video resolution
    with rasterio.open(str(tif_paths[0])) as src:
        nodata = src.nodata
        height, width = src.height, src.width

    # --- Pass 1: scan all TIFs to find global depth range ---
    global_min = np.inf
    global_max = -np.inf
    for tif_path in tif_paths:
        with rasterio.open(str(tif_path)) as src:
            data = src.read(1)
        if nodata is not None:
            valid = (data != nodata) & np.isfinite(data)
        else:
            valid = np.isfinite(data)
        if valid.any():
            vals = data[valid]
            global_min = min(global_min, float(vals.min()))
            global_max = max(global_max, float(vals.max()))

    if global_min >= global_max:
        global_min, global_max = 0.0, 1.0

    print(f'Generating flood video: {len(tif_paths)} frames, {width}x{height} px, {fps:.1f} fps')
    print(f'  Depth range: {global_min:.3f} – {global_max:.3f} m')

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Colour gradient endpoints: light-blue (shallow) → dark-blue (deep)
    color_shallow = np.array([173, 216, 230], dtype=np.float32)
    color_deep = np.array([0, 0, 139], dtype=np.float32)

    # Scale font size relative to image width (roughly 1.5% of width, min 14px)
    font_size = max(14, int(width * 0.015))
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()

    padding = max(6, font_size // 3)

    with imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=8,
                            macro_block_size=1) as writer:
        for idx, tif_path in enumerate(tif_paths):
            with rasterio.open(str(tif_path)) as src:
                data = src.read(1)  # shape (H, W), float32

            # Build RGB frame: black background, depth-coloured wet pixels
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if nodata is not None:
                valid_mask = (data != nodata) & np.isfinite(data)
            else:
                valid_mask = np.isfinite(data)

            if valid_mask.any():
                depths = data[valid_mask]
                t = np.clip((depths - global_min) / (global_max - global_min), 0.0, 1.0)
                colors = (1.0 - t[:, None]) * color_shallow + t[:, None] * color_deep
                frame[valid_mask] = colors.astype(np.uint8)

            # Parse timestamp from filename: flood_map_{idx}_{YYYYMMDD-HHMMSS}.tif
            ts_str = tif_path.stem.split('_')[-1]
            dt = datetime.strptime(ts_str, '%Y%m%d-%H%M%S')
            label = dt.strftime('%Y-%m-%d  %H:%M:%S')

            # Draw timestamp in top-right corner using Pillow
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = width - text_w - padding * 2
            y = padding
            # Semi-transparent dark background behind text for readability
            draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=(0, 0, 0, 160))
            draw.text((x, y), label, font=font, fill=(255, 255, 255))
            frame = np.array(img)

            writer.append_data(frame)

            if (idx + 1) % 10 == 0 or idx == len(tif_paths) - 1:
                print(f'  Encoded frame {idx + 1}/{len(tif_paths)} ...')

    print(f'Saved flood video to: {out_path}')
