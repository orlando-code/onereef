import xarray as xa
import numpy as np
import haversine


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

    # add crs
    temp_xa_d.rio.write_crs(crs, inplace=True)  # recently added back
    # if chunk_dict is not None:
    #     temp_xa_d = chunk_as_necessary(temp_xa_d, chunk_dict)
    # sort coords by ascending values
    return temp_xa_d.sortby(list(temp_xa_d.dims))
