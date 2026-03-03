import shutil
import rasterio
import numpy as np
import taichi as ti
import fastdb4py as fdb
from pathlib import Path
from datetime import datetime
from typing import no_type_check
from rasterio.transform import from_origin

from ..input import InputConfig
from ..util import init_taichi, copy_to_taichi
from ..schema.feature import Ne, UVH, Ns, IndexLike
    
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
        depth[i] = ti.max(h[i] - z[i], 0.0)

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
    isl1 = copy_to_taichi(ne_fdb[IndexLike]['isl1'].column.index, ti.i32, [e_cnt, 10])
    isl3 = copy_to_taichi(ne_fdb[IndexLike]['isl3'].column.index, ti.i32, [e_cnt, 10])
    
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
            lsi0 = isl1[ei, 0]
            lsi2 = isl3[ei, 0]
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

def generate_flood_map(cfg: InputConfig):
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