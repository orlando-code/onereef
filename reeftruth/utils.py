# general
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# spatial
import xarray as xa
import haversine

# statistical
import scipy.stats as sstats

# custom
from reeftruth import resampling


def calculate_coord_resolution(
    xa_d: xa.Dataset | xa.DataArray, coord: str
) -> tuple[float]:
    """Calculate the spatial resolution of latitude and longitude in an xarray Dataset or DataArray.

    Parameters
    ----------
    xa_d (xa.Dataset | xa.DataArray): Input xarray Dataset or DataArray.
    coord (str): Coordinate to calculate the resolution of.

    Returns
    -------
    tuple[float]: Spatial resolution of latitude and longitude.
    """
    return np.mean(np.diff(xa_d[coord].values))


def calculate_spatial_resolution(xa_d: xa.Dataset | xa.DataArray) -> tuple[float]:
    """Calculate the spatial resolution of latitude and longitude in an xarray Dataset or DataArray.

    Parameters
    ----------
    xa_d (xa.Dataset | xa.DataArray): Input xarray Dataset or DataArray.

    Returns
    -------
    tuple[float]: Spatial resolution of latitude and longitude.
    """
    # average latitudinal resolution
    lat_resolution = calculate_coord_resolution(xa_d, "latitude")
    # average longitudinal resolution
    lon_resolution = calculate_coord_resolution(xa_d, "longitude")

    return lat_resolution, lon_resolution


def degrees_to_distances(
    target_lat_res: float,
    target_lon_res: float = None,
    approx_lat: float = -18,
    approx_lon: float = 145,
) -> tuple[float]:
    """Converts target latitude and longitude resolutions from degrees to distances (in meters).

    Parameters
    ----------
        target_lat_res (float): The target latitude resolution in degrees.
        target_lon_res (float, optional): The target longitude resolution in degrees.
            If not specified, the longitude resolution will be assumed to be the same as the latitude resolution.
        approx_lat (float, optional): The approximate latitude coordinate.
            It is used as a reference for distance calculations. Default is -18.
        approx_lon (float, optional): The approximate longitude coordinate.
            It is used as a reference for distance calculations. Default is 145.

    Returns
    -------
        tuple[float]: A tuple containing the converted distances in meters
        (latitude distance, longitude distance, mean distance).

    Notes
    -----
        - It uses the haversine formula to calculate the distance between two coordinates on a sphere.
        - By default, the function assumes an approximate latitude of -18 and an approximate longitude of 145.
        - If only the latitude resolution is specified, the function assumes the longitude resolution is the same.
    """
    start_coord = (approx_lat, approx_lon)
    lat_end_coord = (approx_lat + target_lat_res, approx_lon)
    # if both lat and lon resolutions specified
    if target_lon_res:
        lon_end_coord = (approx_lat, approx_lon + target_lon_res)
    else:
        lon_end_coord = (approx_lat, approx_lon + target_lat_res)

    return (
        haversine.haversine(start_coord, lat_end_coord, unit=haversine.Unit.METERS),
        haversine.haversine(start_coord, lon_end_coord, unit=haversine.Unit.METERS),
        np.mean(
            (
                haversine.haversine(
                    start_coord, lat_end_coord, unit=haversine.Unit.METERS
                ),
                haversine.haversine(
                    start_coord, lon_end_coord, unit=haversine.Unit.METERS
                ),
            )
        ),
    )


def process_xa_d(
    xa_d: xa.Dataset | xa.DataArray,
    rename_lat_lon_grids: bool = False,
    rename_mapping: dict = {
        "lat": "latitude",
        "lon": "longitude",
        "y": "latitude",
        "x": "longitude",
        "i": "longitude",
        "j": "latitude",
        "lev": "depth",
    },
    squeeze_coords: str | list[str] = None,
    # chunk_dict: dict = {"latitude": 100, "longitude": 100, "time": 100},
    crs: str = "EPSG:4326",
):
    """
    Process the input xarray Dataset or DataArray by standardizing coordinate names, squeezing dimensions,
    chunking along specified dimensions, and sorting coordinates.

    Parameters
    ----------
        xa_d (xa.Dataset or xa.DataArray): The xarray Dataset or DataArray to be processed.
        rename_mapping (dict, optional): A dictionary specifying the mapping for coordinate renaming.
            The keys are the existing coordinate names, and the values are the desired names.
            Defaults to a mapping that standardizes common coordinate names.
        squeeze_coords (str or list of str, optional): The coordinates to squeeze by removing size-1 dimensions.
                                                      Defaults to ['band'].
        chunk_dict (dict, optional): A dictionary specifying the chunk size for each dimension.
                                     The keys are the dimension names, and the values are the desired chunk sizes.
                                     Defaults to {'latitude': 100, 'longitude': 100, 'time': 100}.

    Returns
    -------
        xa.Dataset or xa.DataArray: The processed xarray Dataset or DataArray.

    """
    temp_xa_d = xa_d.copy()

    if rename_lat_lon_grids:
        temp_xa_d = temp_xa_d.rename(
            {"latitude": "latitude_grid", "longitude": "longitude_grid"}
        )

    for coord, new_coord in rename_mapping.items():
        if new_coord not in temp_xa_d.coords and coord in temp_xa_d.coords:
            temp_xa_d = temp_xa_d.rename({coord: new_coord})
    # temp_xa_d = xa_d.rename(
    #     {coord: rename_mapping.get(coord, coord) for coord in xa_d.coords}
    # )
    if "band" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.squeeze("band")
    if squeeze_coords:
        temp_xa_d = temp_xa_d.squeeze(squeeze_coords)

    if "time" in temp_xa_d.dims:
        temp_xa_d = temp_xa_d.transpose("time", "latitude", "longitude", ...)
    else:
        temp_xa_d = temp_xa_d.transpose("latitude", "longitude")

    if "grid_mapping" in temp_xa_d.attrs:
        del temp_xa_d.attrs["grid_mapping"]

    # drop variables which will never be variables
    # TODO: add as argument with default
    drop_vars = ["time_bnds"]

    if isinstance(temp_xa_d, xa.Dataset):
        temp_xa_d = temp_xa_d.drop_vars(
            [var for var in drop_vars if var in temp_xa_d.variables]
        )

    temp_xa_d=mask_above_threshold(temp_xa_d)

    # if np.isnan(temp_xa_d).sum() > 0:
    #     temp_xa_d.rio.set_nodata(np.nan, inplace=True)

    # add crs
    temp_xa_d.rio.write_crs(crs, inplace=True)  # recently added back
    # if chunk_dict is not None:
    #     temp_xa_d = chunk_as_necessary(temp_xa_d, chunk_dict)
    # sort coords by ascending values
    return temp_xa_d.sortby(list(temp_xa_d.dims))



def get_buffered_lims(coord_vals, buffer_size):
    return [min(coord_vals) - buffer_size, max(coord_vals) + buffer_size]





def mask_above_threshold(xa_d, threshold=100000):
    """
    Mask values above a given threshold with np.nan in an xarray Dataset or DataArray.

    Parameters:
    xr_obj (xarray.Dataset or xarray.DataArray): The input xarray object.
    threshold (float): The threshold value above which the data will be masked with np.nan. Default is 5000.

    Returns:
    xarray.Dataset or xarray.DataArray: The masked xarray object.
    """
    if isinstance(xa_d, xa.Dataset):
        return xa_d.map(lambda x: x.where(x <= threshold, np.nan))
    elif isinstance(xa_d, xa.DataArray):
        return xa_d.where(xa_d <= threshold, np.nan)
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")


def random_arr_to_smaller(arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray]:
    """Randomly sample elements from arr2 to match the size of arr1.

    Parameters
    ----------
    arr1 (np.ndarray): First array.
    arr2 (np.ndarray): Second array.

    Returns
    -------
    tuple[np.ndarray]: A tuple containing the original arr1 and a randomly sampled subset of arr2 with the same size as arr1.
    """
    if len(arr1.flatten()) <= len(arr2.flatten()):
        return arr1.flatten(), np.random.choice(
            arr2.flatten(), size=len(arr1.flatten()), replace=False
        )
    else:
        return arr2.flatten(), arr1.flatten()[: len(arr2.flatten())]

    np.random.choice(arr1.flatten(), size=len(arr2.flatten()), replace=False)
    # if len(arr1.flatten()) <= len(arr2.flatten()):
    #     return arr1.flatten(), np.random.choice(
    #         arr2.flatten(), size=len(arr1.flatten()), replace=False
    #     )
    # else:
    #     return arr2.flatten(), np.random.choice(
    #         arr1.flatten(), size=len(arr2.flatten()), replace=False
    #     )


def do_wilcoxon(xa_d1: xa.DataArray, xa_d2: xa.DataArray):
    """Perform the Wilcoxon signed-rank test between xa_d1 and xa_d2.

    If the number of elements in xa_d2 is greater than or equal to the number of elements in xa_d1,
    a random sample of elements is taken from xa_d2 to match the size of xa_d1.
    Otherwise, a random sample of elements is taken from xa_d1 to match the size of xa_d2.

    Parameters
    ----------
    xa_d1 (xa.DataArray): Input xarray Dataset or DataArray.
    xa_d2 (xa.DataArray): Second input xarray Dataset or DataArray.

    Returns
    -------
    sstats.WilcoxonResult: Result of the Wilcoxon signed-rank test.
    """
    if len(xa_d2.values.flatten()) >= len(xa_d1.values.flatten()):
        arr1, arr2 = random_arr_to_smaller(xa_d1.values, xa_d2.values)
        return sstats.wilcoxon(arr1, arr2)
    else:
        arr1, arr2 = random_arr_to_smaller(xa_d2.values, xa_d1.values)
        return sstats.wilcoxon(arr1, arr2)


def do_mannwhitneyu(arr1, arr2, alternative="two-sided"):
    """
    Perform the Mann-Whitney U test on two samples.
    """
    return sstats.mannwhitneyu(
        arr1[~np.isnan(arr1)], arr2[~np.isnan(arr2)], alternative=alternative
    )


def calc_method_res_stats(xa_da, params_dict, library="rasterio"):
    if library == "rasterio":
        methods = [method.name for method in params_dict["method"]]
    else:
        methods = params_dict["method"]

    method_res_stats = {
        method: {res: {} for res in params_dict["resolutions"]} for method in methods
    }

    for m_i, method in tqdm(enumerate(methods), total=len(methods)):
        for res in tqdm(
            params_dict["resolutions"], total=len(params_dict["resolutions"])
        ):

            if library == "rasterio":
                resampled = resampling.resample_process_rasterio(
                    xa_da,
                    lat_resolution=res,
                    lon_resolution=res,
                    method=params_dict["method"][m_i],
                )
            elif library == "xesmf":
                resampled = resampling.xesmf_regrid(
                    xa_da,
                    lat_range=[
                        xa_da.latitude.min(),
                        xa_da.latitude.max(),
                    ],
                    lon_range=[
                        xa_da.longitude.min(),
                        xa_da.longitude.max(),
                    ],
                    resolution=res,
                    resampled_method=method,
                )
            elif library == "xarray":
                resampled = resampling.xarray_resample(
                    xa_da, lat_resolution=res, lon_resolution=res, method=method
                )

            # resampled_normed = resampled / resampled.max()

            try:
                ks_stats_result = sstats.ks_2samp(
                    xa_da.values.flatten(), resampled.values.flatten()
                )
                wc_stats_result = do_wilcoxon(xa_da, resampled)
                mwu_stats_result = do_mannwhitneyu(
                    xa_da.values.flatten(), resampled.values.flatten()
                )
            except ValueError:
                ks_stats_result = np.nan
                wc_stats_result = np.nan
                mwu_stats_result = np.nan
                print("Statistical test failed")
                continue

            # Wilcoxon test
            method_res_stats[method][res]["wc_statistic"] = wc_stats_result.statistic
            method_res_stats[method][res]["wc_pvalue"] = wc_stats_result.pvalue
            # Kolmogorov-Smirnov test
            method_res_stats[method][res]["ks_statistic"] = ks_stats_result.statistic
            method_res_stats[method][res]["ks_pvalue"] = ks_stats_result.pvalue
            method_res_stats[method][res][
                "ks_statistic_location"
            ] = ks_stats_result.statistic_location
            method_res_stats[method][res][
                "ks_statistic_sign"
            ] = ks_stats_result.statistic_sign
            # Mann-Whitney U test
            method_res_stats[method][res]["mwu_statistic"] = mwu_stats_result.statistic
            method_res_stats[method][res]["mwu_pvalue"] = mwu_stats_result.pvalue
            method_res_stats[method][res]["array"] = resampled
            # Chi-squared test doesn't work:
            # ValueError: For each axis slice, the sum of the observed frequencies must agree with the sum of the expected frequencies
            # to a relative tolerance of 1e-08, but the percent differences are: (bigger)
    return method_res_stats


def generate_smooth_data_square(N, seed=None, noise_level=0):
    if seed is not None:
        np.random.seed(seed)
    xs = np.linspace(0, 2 * np.pi, N)
    ys = xs
    # generate a sinx2 + cosx2 function on a NxN grid
    X, Y = np.meshgrid(xs, ys)
    z = np.sin(X) ** 2 + np.cos(Y) ** 2
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, size=(N, N))
        z += noise
    return z / np.max(z)


# CDO


def generate_remap_info(eg_nc, resolution=0.25, out_grid: str = "latlon"):
    # [-180, 180] longitudinal range
    xfirst = float(np.min(eg_nc.longitude).values) - 180
    yfirst = float(np.min(eg_nc.latitude).values)

    xsize = int(360 / resolution)
    # [smallest latitude, largest latitude] range
    ysize = int((180 / resolution) + yfirst)

    x_inc, y_inc = resolution, resolution

    return xsize, ysize, xfirst, yfirst, x_inc, y_inc


def generate_remapping_file(
    eg_xa: xa.Dataset | xa.DataArray,
    remap_template_fp: str | Path,
    resolution: float = 0.25,
    out_grid: str = "latlon",
):
    xsize, ysize, xfirst, yfirst, x_inc, y_inc = generate_remap_info(
        eg_nc=eg_xa, resolution=resolution, out_grid=out_grid
    )

    print(f"Saving regridding info to {remap_template_fp}...")
    with open(remap_template_fp, "w") as file:
        file.write(
            f"gridtype = {out_grid}\n"
            f"xsize = {xsize}\n"
            f"ysize = {ysize}\n"
            f"xfirst = {xfirst}\n"
            f"yfirst = {yfirst}\n"
            f"xinc = {x_inc}\n"
            f"yinc = {y_inc}\n"
        )


def replace_dot_with_dash(string: str) -> str:
    """
    Replace all occurrences of "." with "-" in a string.

    Parameters
    ----------
        string (str): The input string.

    Returns
    -------
        str: The modified string with "." replaced by "-".
    """
    return string.replace(".", "-")
