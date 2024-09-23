# general
import numpy as np
from tqdm.auto import tqdm
import time
from pathlib import Path

# resampling libraries
import xarray as xa
from rasterio.enums import Resampling
import xesmf
# from cdo import Cdo


# custom
from reeftruth import utils


def calc_scale_factor(initial_resolution: float, final_resolution: float) -> float:
    """
    Calculate the scale factor between initial and final resolutions.

    Args:
        initial_resolution: The initial resolution.
        final_resolution: The final resolution.

    Returns:
        The scale factor between initial and final resolutions.
    """
    return initial_resolution / final_resolution


def scaled_width_height(
    lat_scale_factor: float,
    lon_scale_factor: float,
    initial_width: int,
    initial_height: int,
) -> tuple[int, int]:
    """
    Calculate the scaled width and height based on the scale factors and initial dimensions.

    Args:
        lat_scale_factor: The scale factor for latitude.
        lon_scale_factor: The scale factor for longitude.
        initial_width: The initial width.
        initial_height: The initial height.

    Returns:
        A tuple containing the scaled width and height.
    """
    return round(initial_width * lon_scale_factor), round(
        initial_height * lat_scale_factor
    )


def resample_rasterio(
    rio_xa: xa.DataArray,
    lat_resolution: float,
    lon_resolution: float,
    method: Resampling = Resampling.bilinear,
    n_threads: int = 4,
) -> xa.DataArray:
    """
    Resample a raster using rasterio.

    Args:
        rio_xa: The input raster as a xarray DataArray.
        lat_resolution: The desired latitude resolution.
        lon_resolution: The desired longitude resolution.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The resampled raster as a xarray DataArray.
    """
    lat_scale_factor = calc_scale_factor(
        abs(rio_xa.rio.resolution()[0]), lat_resolution
    )
    lon_scale_factor = calc_scale_factor(
        abs(rio_xa.rio.resolution()[1]), lon_resolution
    )

    new_width, new_height = scaled_width_height(
        lat_scale_factor, lon_scale_factor, rio_xa.rio.width, rio_xa.rio.height
    )
    if new_width == 0 or new_height == 0:
        raise ValueError(
            f"Cannot resample to 0 width or height. width: {new_width}, height: {new_height}"
        )

    # rio_xa.rio.write_nodata(np.nan, inplace=True)
    return rio_xa.rio.reproject(
        rio_xa.rio.crs,
        shape=(new_height, new_width),
        resampling=method,
        n_threads=n_threads
    )


def resample_process_rasterio(
    rio_xa: xa.DataArray,
    lat_resolution: float,
    lon_resolution: float,
    method: Resampling = Resampling.bilinear,
) -> xa.DataArray:
    """
    Process and resample a raster using rasterio.

    Args:
        rio_xa: The input raster as a xarray DataArray.
        lat_resolution: The desired latitude resolution.
        lon_resolution: The desired longitude resolution.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The processed and resampled raster as a xarray DataArray.
    """
    return utils.process_xa_d(
        resample_rasterio(rio_xa, lat_resolution, lon_resolution, method)
    )


def rio_resample_to_other(
    xa_d: xa.DataArray,
    other_xa_d: xa.DataArray,
    resample_method: Resampling = Resampling.bilinear,
    project_first: bool = True
) -> xa.DataArray:
    """
    Resample a raster to match the resolution of another raster.

    Args:
        xa_d: The input raster as a xarray DataArray.
        other_xa_d: The other raster to match the resolution to.
        method: The resampling method to use. Defaults to Resampling.bilinear.

    Returns:
        The resampled raster as a xarray DataArray.
    """
    final_lat_resolution = abs(other_xa_d.rio.resolution()[0])
    final_lon_resolution = abs(other_xa_d.rio.resolution()[1])

    if project_first:
        # reproject first to match gridding
        reprojected = utils.process_xa_d(
            xa_d.rio.reproject_match(other_xa_d, resampling=resample_method)
        )
        # resample to correct resolution
        return resample_process_rasterio(
            reprojected, final_lat_resolution, final_lon_resolution, resample_method
        )
    else:
        resampled = resample_process_rasterio(
            xa_d, final_lat_resolution, final_lon_resolution, resample_method
        )
        return utils.process_xa_d(
            resampled.rio.reproject_match(other_xa_d, resampling=resample_method)
        )


def rio_absolute_resample(
    xa_d: xa.DataArray | xa.Dataset,
    lat_resolution: float,
    lon_resolution: float,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resample_method: Resampling = Resampling.bilinear,
    project_first: bool = True

):
    if not xa_d.rio.crs:
        xa_d = xa_d.rio.write_crs("EPSG:4326")
        print("Written raster CRS to EPSG:4326")

    common_dataset = xa.Dataset(
        coords={
            "latitude": (
                ["latitude"],
                np.arange(lat_range[0], lat_range[1] + lat_resolution, lat_resolution),
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_range[0], lon_range[1] + lon_resolution, lon_resolution),
            ),
        }
    ).rio.write_crs(xa_d.rio.crs)
    return utils.process_xa_d(rio_resample_to_other(xa_d, common_dataset, resample_method, project_first=project_first))


def coarsen_xa_d(xa_d, resolution: float = 0.1, method="sum"):
    # TODO: for now, treating lat and long with indifference (since this is how data is).
    num_points_lat = int(
        round(resolution / abs(xa_d["latitude"].diff("latitude").mean().values))
    )
    num_points_lon = int(
        round(resolution / abs(xa_d["longitude"].diff("longitude").mean().values))
    )

    return xa_d.coarsen(
        latitude=num_points_lat,
        longitude=num_points_lon,
        boundary="pad",
    ).reduce(method)


def coarsen_dataset(ds, lat_res, lon_res, method="sum"):
    """
    Resample an xarray dataset to the specified latitude and longitude resolutions.
    
    Parameters:
    ds (xarray.Dataset): The input dataset.
    lat_res (float): The desired latitude resolution.
    lon_res (float): The desired longitude resolution.
    
    Returns:
    xarray.Dataset: The resampled dataset.
    """
    # Calculate the current resolution
    current_lat_res = np.abs(ds['latitude'][1] - ds['latitude'][0])
    current_lon_res = np.abs(ds['longitude'][1] - ds['longitude'][0])
    
    # Calculate the coarsening factors
    lat_factor = int(np.round(lat_res / current_lat_res))
    lon_factor = int(np.round(lon_res / current_lon_res))
    
    # Check if the coarsening factors are greater than 0
    if lat_factor <= 0 or lon_factor <= 0:
        raise ValueError("The specified resolution is too fine compared to the original resolution.")
    
    # Coarsen the dataset
    ds_coarsened = ds.coarsen(latitude=lat_factor, longitude=lon_factor, boundary='trim').reduce(method)
    
    return ds_coarsened


def absolute_coarsen_dataset(
    xa_d: xa.DataArray | xa.Dataset,
    lat_resolution: float,
    lon_resolution: float,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resample_method: str = Resampling.bilinear,
    project_first: bool = True,
):
    coarse_d = coarsen_dataset(xa_d, lat_resolution, lon_resolution, method="sum")

    common_dataset = xa.Dataset(
        coords={
        "latitude": (
            ["latitude"],
            np.arange(lat_range[0], lat_range[1] + lat_resolution, lat_resolution),
        ),
        "longitude": (
            ["longitude"],
            np.arange(lon_range[0], lon_range[1] + lon_resolution, lon_resolution),
        ),
        }
    ).rio.write_crs(xa_d.rio.crs)
    # return xa_d.interp_like(common_dataset)
    return utils.process_xa_d(rio_resample_to_other(xa_d, common_dataset, resample_method, project_first=project_first))


class Resample:
    def __init__(self, lats, lons, resolution, 
    # resolution_unit
    ):
        self.lats = lats
        self.lons = lons
        self.resolution = resolution
        # self.resolution_unit = resolution_unit

    def get_resampled_raster(self, raster, dataset):
        # self.resolution = spatial_data.process_resolution_input(
        #     self.resolution, self.resolution_unit
        # )
        print(f"\tresampling dataset to {self.resolution} degree(s) resolution...\n")
        mean_ds_resolution = np.mean(raster.rio.resolution())

        # fetching more data than required necessary to avoid missing values at edge
        buffered_lats = utils.get_buffered_lims(self.lats, self.resolution)
        buffered_lons = utils.get_buffered_lims(self.lons, self.resolution)

        if dataset in ["unep", "unep_wcmc", "gdcr", "unep_coral_presence"]:
            # if mean_ds_resolution > self.resolution:  # TODO: better control: if desired resolution is greater than dataset resolution, nearest with rio
            #     resample_method = Resampling.nearest    # don't invent new data
            # else:
            #     resample_method = Resampling.sum
            return absolute_coarsen_dataset(
                raster.sel(latitude=slice(*buffered_lats), longitude=slice(*buffered_lons)),
                lat_resolution=self.resolution,
                lon_resolution=self.resolution,
                lat_range=self.lats,
                lon_range=self.lons,
                resample_method=Resampling.sum,
                project_first=False,
            )

        if mean_ds_resolution < self.resolution:    # if desired resolution is greater than dataset resolution
            resample_method = Resampling.nearest
        else:
            resample_method = Resampling.bilinear

        return rio_absolute_resample(
            raster.sel(latitude=slice(*buffered_lats), longitude=slice(*buffered_lons)),
            lat_resolution=self.resolution,
            lon_resolution=self.resolution,
            lat_range=self.lats,
            lon_range=self.lons,
            resample_method=resample_method,
            project_first=False,
        )


def multi_resample_xa_d(
    xa_d: xa.DataArray | xa.Dataset,
    lat_resolution: float,
    lon_resolution: float,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resample_method: str = Resampling.bilinear,
    project_first: bool = True,
):
    common_dataset = xa.Dataset(
        coords={
        "latitude": (
            ["latitude"],
            np.arange(lat_range[0], lat_range[1] + lat_resolution, lat_resolution),
        ),
        "longitude": (
            ["longitude"],
            np.arange(lon_range[0], lon_range[1] + lon_resolution, lon_resolution),
        ),
        }
    ).rio.write_crs(xa_d.rio.crs)





def xesmf_regrid(
    xa_d: xa.DataArray | xa.Dataset,
    lat_range: list[float] = None,
    lon_range: list[float] = None,
    resolution: float = 0.1,
    resampled_method: str = "bilinear",
):

    lon_range = sorted(lon_range)
    lat_range = sorted(lat_range)
    target_grid = xesmf.util.grid_2d(
        lon_range[0],
        lon_range[1],
        resolution,
        lat_range[0],
        lat_range[1],
        resolution,  # longitude range and resolution
    )  # latitude range and resolution

    regridder = xesmf.Regridder(
        xa_d.astype("float64", order="C"),
        target_grid.chunk({"y": 100, "x": 100, "y_b": 100, "x_b": 100}),
        method=resampled_method,
        parallel=True,
    )
    return process_xesmf_regridded(regridder(xa_d.astype("float64", order="C")))


def process_xesmf_regridded(
    xa_d: xa.DataArray | xa.Dataset,
):
    if isinstance(xa_d, xa.DataArray):
        xa_d = xa_d.to_dataset(name="var_name")

    xa_d["lon"] = xa_d.lon.values[0, :]
    xa_d["lat"] = xa_d.lat.values[:, 0]

    return xa_d.rename(
        {"x": "longitude", "y": "latitude", "lon": "longitude", "lat": "latitude"}
    )


def xarray_resample(xa_da, lat_resolution, lon_resolution, method):
    """
    Resample an xarray DataArray to a new resolution.
    """
    # get the new lat and lon values
    new_lats = np.arange(xa_da.latitude.min(), xa_da.latitude.max(), lat_resolution)
    new_lons = np.arange(xa_da.longitude.min(), xa_da.longitude.max(), lon_resolution)

    # create the new DataArray
    new_xa_da = xa.DataArray(
        np.zeros((len(new_lats), len(new_lons)), dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": new_lats, "longitude": new_lons},
    )

    # resample the data
    new_xa_da.values = xa_da.interp(
        latitude=new_lats, longitude=new_lons, method=method
    ).values

    return new_xa_da


def time_method_variables(
    xa_da,
    var1: str,
    var2: str,
    params_dict,
    func_code: str,
    library: str,
    method="linear",
    repeats=10,
):

    timings_dict = {
        k1: {k2: {} for k2 in params_dict[var2]} for k1 in params_dict[var1]
    }

    for v1 in tqdm(params_dict[var1], total=len(params_dict[var1]), desc=f"{var1}"):
        for v2 in tqdm(
            params_dict[var2], total=len(params_dict[var2]), desc=f"{var1}={v1}"
        ):
            if func_code == "prop_res":

                raster = xa_da.isel(
                    latitude=slice(0, round(xa_da.latitude.size * v1)),
                    longitude=slice(0, round(xa_da.longitude.size * v1)),
                )

            else:
                raster = xa_da

            tic = time.time()
            skip = False
            for _ in range(repeats):
                if not skip:
                    if func_code == "prop_res":
                        if library == "xarray":
                            xarray_resample(
                                raster,
                                lat_resolution=v2,
                                lon_resolution=v2,
                                method=method,
                            )
                        elif library == "rasterio":
                            try:
                                resample_process_rasterio(
                                    raster,
                                    lat_resolution=v2,
                                    lon_resolution=v2,
                                    method=method,
                                )
                            except ValueError as e:
                                print(
                                    f"Error: {e}\nSkipping {var1}={v1} {var2}={v2} method={method} library={library}"
                                )
                                skip = True
                        elif library == "xesmf":
                            xesmf_regrid(
                                raster,
                                lat_range=[
                                    raster.latitude.min(),
                                    raster.latitude.max(),
                                ],
                                lon_range=[
                                    raster.longitude.min(),
                                    raster.longitude.max(),
                                ],
                                resolution=v2,
                                resampled_method=method,
                            )
                    elif func_code == "method_res":
                        if library == "xarray":
                            xarray_resample(
                                raster, lat_resolution=v2, lon_resolution=v2, method=v1
                            )
                        elif library == "rasterio":
                            resample_process_rasterio(
                                raster, lat_resolution=v2, lon_resolution=v2, method=v1
                            )
                        elif library == "xesmf":
                            xesmf_regrid(
                                raster,
                                lat_range=[
                                    raster.latitude.min(),
                                    raster.latitude.max(),
                                ],
                                lon_range=[
                                    raster.longitude.min(),
                                    raster.longitude.max(),
                                ],
                                resolution=v2,
                                resampled_method=v1,
                            )
                    else:
                        raise ValueError(f"Invalid func_code: {func_code}")

            total_time = time.time() - tic
            timings_dict[v1][v2]["total_time"] = total_time
            timings_dict[v1][v2]["iteration_time"] = total_time / repeats

    return timings_dict


def cdo_regrid(input_fp, output_fp, remap_template_fp, resolution, method="remapbil"):
    cdo = Cdo()

    ds = xa.open_dataset(input_fp)
    if not remap_template_fp.exists():
        utils.generate_remapping_file(
            ds,
            remap_template_fp=remap_template_fp,
            resolution=resolution,
            out_grid="latlon",
        )
    # create output_fp
    output_name = f"{Path(input_fp).name}_{resolution}_regridded.nc"
    output_dir = (
        Path(input_fp).parent
        / f"{utils.replace_dot_with_dash(resolution):03f}_regridded"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fp = output_dir / output_name

    cdo.remapbil(
        remap_template_fp,
        input=input_fp,
        output=output_fp,
    )
