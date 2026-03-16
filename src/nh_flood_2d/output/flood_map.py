import shutil
import rasterio
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from datetime import datetime, timezone
from typing import no_type_check
from rasterio.transform import from_origin

from ..input import DomainConfig, ForceConfig
from ..util import init_taichi, copy_to_taichi
from ..schema.feature import Ne, UVH, Ns, IndexLike, Rainfall
    
# Check if grid is too large for single Taichi field (limit near 2^31 elements, but practical limit lower)
# Using 20000x20000 as a safe threshold for tiling
TILE_SIZE = 4096

@ti.kernel
@no_type_check
def compute_depth(h: ti.template(), z: ti.template(), depth: ti.template()):
    """
    Compute water depth for each element.
    Water depth = max(h - z, 0)
    """
    for i in h:
        # depth[i] = ti.max(h[i] - z[i], 0.0)
        depth[i] = h[i]
        if (h[i] - z[i]) < 0.2:  # if depth < 0.2m, consider it dry
            depth[i] = -9999.0  # mark as invalid if max depth < threshold

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
    Generate a GeoTIFF flood map from UVH calculation results.
    """
    init_taichi()
    
    ne_fdb_fn = cfg.ne_fdb
    ns_fdb_fn = cfg.ns_fdb
    uvhs_dir = cfg.uvh_dir
    epsg_code = cfg.epsg_code
    output_dir = cfg.flood_map_dir
    
    # Check input files
    uvhs_path = Path(uvhs_dir)
    ne_fdb_path = Path(ne_fdb_fn)
    ns_fdb_path = Path(ns_fdb_fn)
    if not ne_fdb_path.exists() or not ns_fdb_path.exists() or not uvhs_path.exists():
        raise FileNotFoundError(f'ne.fdb or ns.fdb or uvh directory not found at {ne_fdb_path} or {ns_fdb_path} or {uvhs_path}')
    
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
    z_field = copy_to_taichi(nes.column.z[1:], ti.f32, None)
    
    # Element dimensions fields
    hw_field = copy_to_taichi(half_widths, ti.f32, None)
    hh_field = copy_to_taichi(half_heights, ti.f32, None)
    
    # Make flood map for all UVH files in the directory
    uvh_paths = list(uvhs_path.glob('uvh_*.fdb'))
    uvh_paths.sort(
        key=lambda p: datetime.strptime(p.stem.split('_')[-1], '%Y%m%d-%H%M%S').timestamp()
    )
    for idx, uvh_file in enumerate(uvh_paths):
        print(f'Processing UVH file: {uvh_file} ...')
        
        # Output file name based on UVH file name (e.g., flood_map_0_20240101-120000.tif)
        out_name = f'flood_map_{idx}_{uvh_file.stem.split("_")[-1]}.tif'
        out_path = output_dir / out_name

        # Load UVH data
        uvh_fdb = fdb.ORM.load(str(uvh_file), from_file=True)
        uvhs = uvh_fdb[UVH][UVH]
        
        # Compute water depth on GPU
        h_field = copy_to_taichi(uvhs.column.h[1:], ti.f32, None)
        depth_field = ti.field(dtype=ti.f32, shape=e_cnt)
        compute_depth(h_field, z_field, depth_field)
        
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
            nodata=-9999.0,
            compress='lzw', # good for sparse data
            tiled=True,
            blockxsize=512, blockysize=512
        ) as dst:
            # Make tile
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
                        # Compute integer grid index range to loop over within the tile
                        # The grid indices (r, c) correspond to:
                        # x_center = tile_min_x + (c + 0.5) * h_res
                        # y_center = tile_max_y - (r + 0.5) * v_res
                        
                        # We want to cover the range [e_min_x, e_max_x] and [e_min_y, e_max_y]
                        
                        # Convert world bounds to tile-relative grid coordinates
                        # col = (x - tile_min_x) / h_res
                        # row = (tile_max_y - y) / v_res
                        
                        # For X (Columns):
                        # We want smallest col c such that cell right edge > e_min_x
                        # cell_right_edge = (c + 1) * h_res
                        # (c+1)*h_res > e_min_x - tile_min_x  => c > (e_min_x - tile_min_x)/h_res - 1
                        # So start_col = floor((e_min_x - tile_min_x) / h_res)
                        
                        # We want largest col c such that cell left edge < e_max_x
                        # cell_left_edge = c * h_res
                        # c * h_res < e_max_x - tile_min_x => c < (e_max_x - tile_min_x)/h_res
                        
                        start_col_f = ti.floor((e_min_x - tile_min_x) / hr)
                        end_col_f   = ti.floor((e_max_x - tile_min_x) / hr)
                        start_row_f = ti.floor((tile_max_y - e_max_y) / vr)
                        end_row_f   = ti.floor((tile_max_y - e_min_y) / vr)
                        
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
    Generate a single GeoTIFF showing the maximum inundation extent across all UVH timesteps.

    For each pixel, the maximum water depth over all output steps is computed.
    Pixels whose maximum depth never reaches `min_depth` are set to `invalid_data` (nodata).
    Pixels that do reach `min_depth` retain their maximum depth value.

    If `is_absolute` is True, the output is a binary int8 map:
      - 1 for wet pixels (max depth >= min_depth)
      - 0 for dry pixels (max depth < min_depth)
      - `invalid_data` for pixels outside the domain (e.g., -128)

    Uses the same Taichi-accelerated tiled rasterization approach as generate_flood_map.
    Optimized: accumulates per-element max h on GPU across all timesteps, then rasterizes once.
    Output: <domain_dir>/max_inundation/max_inundation.tif
    """
    init_taichi()

    ne_fdb_fn = cfg.ne_fdb
    ns_fdb_fn = cfg.ns_fdb
    uvhs_dir = cfg.uvh_dir
    epsg_code = cfg.epsg_code
    output_path = Path(cfg.max_inundation_dir) / 'max_inundation.tif'

    # Check input files
    uvhs_path = Path(uvhs_dir)
    ne_fdb_path = Path(ne_fdb_fn)
    ns_fdb_path = Path(ns_fdb_fn)
    if not ne_fdb_path.exists() or not ns_fdb_path.exists() or not uvhs_path.exists():
        raise FileNotFoundError(f'ne.fdb or ns.fdb or uvh directory not found at {ne_fdb_path} or {ns_fdb_path} or {uvhs_path}')

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
    z_field  = copy_to_taichi(nes.column.z[1:], ti.f32, None)
    hw_field = copy_to_taichi(half_widths, ti.f32, None)
    hh_field = copy_to_taichi(half_heights, ti.f32, None)

    # Sort UVH files chronologically
    uvh_paths = list(uvhs_path.glob('uvh_*.fdb'))
    uvh_paths.sort(
        key=lambda p: datetime.strptime(p.stem.split('_')[-1], '%Y%m%d-%H%M%S').timestamp()
    )

    if not uvh_paths:
        print('No UVH files found. Nothing to process.')
        return

    real_e_cnt = e_cnt - 1  # skip virtual element 0

    # Per-element max-h accumulator on GPU, initialized to -inf so any real h wins
    max_h_field = ti.field(dtype=ti.f32, shape=real_e_cnt)
    max_h_field.fill(-1e10)

    @ti.kernel
    @no_type_check
    def update_max_h(h: ti.template(), max_h: ti.template()):
        for i in h:
            ti.atomic_max(max_h[i], h[i])

    # Pass 1: scan all timesteps, keep per-element maximum h on GPU
    for uvh_file in uvh_paths:
        print(f'Processing UVH file: {uvh_file} ...')
        uvh_fdb = fdb.ORM.load(str(uvh_file), from_file=True)
        uvhs = uvh_fdb[UVH][UVH]
        h_field = copy_to_taichi(uvhs.column.h[1:], ti.f32, None)
        update_max_h(h_field, max_h_field)

    # Pass 2: compute max depth once and rasterize once
    depth_field = ti.field(dtype=ti.f32, shape=real_e_cnt)

    if is_absolute:
        @ti.kernel
        @no_type_check
        def compute_depth_max(h: ti.template(), z: ti.template(), depth: ti.template()):
            for i in h:
                if (h[i] - z[i]) >= min_depth:
                    depth[i] = 1.0
                else:
                    depth[i] = 0.0
    else:
        @ti.kernel
        @no_type_check
        def compute_depth_max(h: ti.template(), z: ti.template(), depth: ti.template()):
            for i in h:
                depth[i] = h[i]
                if (h[i] - z[i]) < min_depth:
                    depth[i] = -9999.0  # mark as invalid if max depth < threshold

    compute_depth_max(max_h_field, z_field, depth_field)

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
                start_col_f = ti.floor((e_min_x - tile_min_x) / hr)
                end_col_f   = ti.floor((e_max_x - tile_min_x) / hr)
                start_row_f = ti.floor((tile_max_y - e_max_y) / vr)
                end_row_f   = ti.floor((tile_max_y - e_min_y) / vr)

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
    between two simulation configurations across all shared UVH timesteps.

    For every timestep t that both configs have produced, computes:

        MAE(t) = (1 / N_t) * sum |WSE_ref(t) - WSE_cmp(t)|

    where N_t is the number of raster pixels simultaneously wet (h - z > min_depth)
    in both grids.  The result is a 1-D time-series curve.

    If *force_cfg* is provided and its rain.fdb exists, a rainfall hyetograph is
    overlaid on an inverted secondary Y-axis (meteorological convention: bars grow
    downward from the top of the figure).

    Uses Taichi GPU-accelerated tiled rasterisation (same TILE_SIZE as the rest of
    this module) and atomic double-precision reduction for the MAE accumulation.

    Parameters
    ----------
    cfg_ref      : Reference DomainConfig  (e.g., high-resolution 4 m baseline)
    cfg_cmp      : Comparison DomainConfig (e.g., MRCG)
    force_cfg    : Optional ForceConfig; when provided the rainfall hyetograph is
                   overlaid using rain.fdb from its preprocessed directory.
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
    uvh_path_ref = Path(cfg_ref.uvh_dir)
    uvh_path_cmp = Path(cfg_cmp.uvh_dir)

    for p in [ne_path_ref, ns_path_ref, ne_path_cmp, ns_path_cmp, uvh_path_ref, uvh_path_cmp]:
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
    ez_r  = copy_to_taichi(nes_r.column.z[1:], ti.f32, None)
    ehw_r = copy_to_taichi(hws_r,               ti.f32, None)
    ehh_r = copy_to_taichi(hhs_r,               ti.f32, None)

    ex_c  = copy_to_taichi(nes_c.column.x[1:], ti.f32, None)
    ey_c  = copy_to_taichi(nes_c.column.y[1:], ti.f32, None)
    ez_c  = copy_to_taichi(nes_c.column.z[1:], ti.f32, None)
    ehw_c = copy_to_taichi(hws_c,               ti.f32, None)
    ehh_c = copy_to_taichi(hhs_c,               ti.f32, None)

    # Two tile buffers (one per mesh) for tiled GPU rasterisation
    tile_ref = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))
    tile_cmp = ti.field(dtype=ti.f32, shape=(TILE_SIZE, TILE_SIZE))

    # Pre-allocate h fields once; updated each timestep with from_numpy().
    # Avoids calling ti.field() inside the loop which exhausts Metal's snode limit.
    h_r_field = ti.field(dtype=ti.f32, shape=len(nes_r) - 1)
    h_c_field = ti.field(dtype=ti.f32, shape=len(nes_c) - 1)

    # ── Taichi kernels ─────────────────────────────────────────────────────────

    @ti.kernel
    @no_type_check
    def rasterize_wse(
        x: ti.template(), y: ti.template(),
        hw: ti.template(), hh: ti.template(),
        z: ti.template(), h: ti.template(),
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
                if (wse - z[i]) < min_depth:
                    wse = -9999.0

                c0 = ti.max(0, int(ti.floor((e_min_x - tile_min_x) / hr)))
                c1 = ti.min(t_cols - 1, int(ti.floor((e_max_x - tile_min_x) / hr)))
                r0 = ti.max(0, int(ti.floor((tile_max_y - e_max_y) / vr)))
                r1 = ti.min(t_rows - 1, int(ti.floor((tile_max_y - e_min_y) / vr)))

                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        tile[r, c] = wse

    # ── Discover shared UVH timesteps ──────────────────────────────────────────
    ts_ref = {p.stem.split('_')[-1]: p for p in uvh_path_ref.glob('uvh_*.fdb')}
    ts_cmp = {p.stem.split('_')[-1]: p for p in uvh_path_cmp.glob('uvh_*.fdb')}
    common = sorted(
        set(ts_ref) & set(ts_cmp),
        key=lambda t: datetime.strptime(t, '%Y%m%d-%H%M%S').timestamp(),
    )
    if not common:
        raise ValueError('No common UVH timestamps found between cfg_ref and cfg_cmp.')
    print(f'Processing {len(common)} shared UVH timesteps ...')

    # ── Optional: load rainfall for hyetograph overlay ─────────────────────────
    rain_dts: list[datetime] | None = None
    rain_qty: np.ndarray | None = None
    if force_cfg is not None:
        rain_fdb_path = Path(force_cfg.rain_fdb)
        if rain_fdb_path.exists():
            _rfdb   = fdb.ORM.load(str(rain_fdb_path), from_file=True)
            _rfall  = _rfdb[Rainfall][Rainfall]
            rain_dts = [datetime.fromtimestamp(float(t), tz=timezone.utc).replace(tzinfo=None) for t in _rfall.column.time]
            rain_qty = _rfall.column.quantity.copy()   # mm per interval
        else:
            print(f'Warning: rain.fdb not found at {rain_fdb_path}, skipping hyetograph.')

    # ── Per-timestep spatial MAE computation ───────────────────────────────────
    mae_list:  list[float]    = []
    time_axis: list[datetime] = []

    for ts in common:
        uvh_r = fdb.ORM.load(str(ts_ref[ts]), from_file=True)[UVH][UVH]
        uvh_c = fdb.ORM.load(str(ts_cmp[ts]), from_file=True)[UVH][UVH]

        # Reuse pre-allocated fields; no new ti.field() allocation here
        h_r_field.from_numpy(uvh_r.column.h[1:].astype(np.float32))
        h_c_field.from_numpy(uvh_c.column.h[1:].astype(np.float32))

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
                    ex_r, ey_r, ehw_r, ehh_r, ez_r, h_r_field,
                    tile_ref, left_x, top_y, g_hr, g_vr, t_rows, t_cols,
                )
                rasterize_wse(
                    ex_c, ey_c, ehw_c, ehh_c, ez_c, h_c_field,
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

        mae = mae_sum_acc / mae_cnt_acc if mae_cnt_acc > 0 else 0.0
        mae_list.append(mae)
        time_axis.append(datetime.strptime(ts, '%Y%m%d-%H%M%S'))
        print(f'  {ts}: MAE = {mae:.4f} m  (N_wet = {mae_cnt_acc} px)')

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(time_axis, mae_list, color='steelblue', linewidth=2, label='Spatial MAE (m)')
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.set_ylabel('Spatial MAE of WSE (m)', color='steelblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()

    if rain_dts is not None and rain_qty is not None and len(rain_dts) > 1:
        # Clamp rainfall records to the simulation time range
        sim_start = min(time_axis)
        sim_end   = max(time_axis)
        rain_pairs = [(d, q) for d, q in zip(rain_dts, rain_qty) if sim_start <= d <= sim_end]
        if rain_pairs:
            rain_dts_clamped, rain_qty_clamped = zip(*rain_pairs)
            ax2 = ax1.twinx()
            dt_sec = (rain_dts[1] - rain_dts[0]).total_seconds()
            bar_w  = dt_sec / 86400.0   # matplotlib date unit = 1 day
            ax2.bar(
                rain_dts_clamped, rain_qty_clamped,
                width=bar_w, align='edge',
                alpha=0.35, color='dimgray', label='Rainfall (mm)',
            )
            ax2.set_ylabel('Rainfall (mm)', color='dimgray', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='dimgray')
            ax2.invert_yaxis()   # meteorological convention: bars grow downward from top
            ax2.legend(loc='upper right')

    ref_label = Path(cfg_ref.domain_dir).name
    cmp_label = Path(cfg_cmp.domain_dir).name
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
