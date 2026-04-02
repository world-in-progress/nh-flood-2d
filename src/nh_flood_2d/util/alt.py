import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def modify_dem_by_lum(
    dem_path: str,
    lum_path: str,
    land_use_type: int,
    output_path: str,
    minus_value: int = 2,
) -> None:
    """Modify DEM elevations based on land-use type sampled from a LUM raster.

    For every DEM pixel whose nearest-neighbour LUM value equals
    *land_use_type*, if the elevation is greater than 3 the value is
    reduced by 2.  The result is written to *output_path* as a GeoTIFF
    that preserves the DEM's spatial reference and profile.
    """

    with rasterio.open(dem_path) as dem_ds, rasterio.open(lum_path) as lum_ds:
        dem_data = dem_ds.read(1)
        dem_profile = dem_ds.profile.copy()

        # Reproject LUM onto the DEM grid using nearest-neighbour sampling
        lum_on_dem = np.empty_like(dem_data, dtype=lum_ds.dtypes[0])
        reproject(
            source=rasterio.band(lum_ds, 1),
            destination=lum_on_dem,
            dst_transform=dem_ds.transform,
            dst_crs=dem_ds.crs,
            dst_nodata=lum_ds.nodata,
            resampling=Resampling.nearest,
        )

        # Exclude nodata pixels from modification so they stay unchanged
        src_nodata = dem_ds.nodata
        valid = dem_data != src_nodata if src_nodata is not None else np.ones(dem_data.shape, dtype=bool)

        # Apply rule: where LUM == land_use_type and elevation > 3, subtract minus_value
        mask = valid & (lum_on_dem == land_use_type) & (dem_data > 3)
        dem_data[mask] -= minus_value

        # Replace original nodata pixels with -9999
        out_nodata = -9999.0
        if src_nodata is not None:
            dem_data[~valid] = out_nodata

    dem_profile.update(driver="GTiff", nodata=out_nodata)
    with rasterio.open(output_path, "w", **dem_profile) as dst:
        dst.write(dem_data, 1)
