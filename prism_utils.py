# general
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

# spatial
import xarray as xa
import netCDF4 as nc4
import multiprocessing

def rearrange_prism_nc(fp: Path | str):
    """PLACEHOLDER"""
    # read group names from tempfile
    with nc4.Dataset(fp, "r") as temp_nc:
        groups = list(temp_nc.groups.keys())
        attributes = temp_nc.__dict__

    nav_data = xa.open_dataset(fp, group="navigation_data")
    derived_data = xa.open_dataset(fp, group="derived_data")
    # create raw (irregular) grid dataset from navigation_data and derived_data
    lats = nav_data["latitude"]
    lons = nav_data["longitude"]

    # 2d vars   # TODO: extend to specific vars
    vars_2d = ["coral", "algae", "sand", "depth", "GPP", "G"]
    # Initialize data_vars dictionary
    data_vars = {}

    # Populate data_vars with all variables from derived_data
    for var_name in vars_2d:
        data_vars[var_name] = (['lat', 'lon'], np.array(derived_data[var_name].data))
    
    # Create new dataset
    ds = xa.Dataset(
        data_vars=data_vars
    )
    # Assign coordinates to the dataset
    ds = ds.assign_coords({"lat": (("j", "i"), lats.data), "lon": (("j", "i"), lons.data)})
    # make i, j coordinates
    ds = ds.assign_coords({"i": np.arange(ds.dims["lon"]), "j": np.arange(ds.dims["lat"])})
    # assign attributes from nc4 
    ds.attrs = attributes
    return ds


def irregular_to_regular(ds: xa.Dataset, N_grid: int):
    """Interpolate irregular grid data to regular grid data."""
    # Extract coordinates
    lats = ds["lat"].values
    lons = ds["lon"].values

    point_centres = np.array([lats.flatten(), lons.flatten()]).T

    # new grid
    lats_new = np.linspace(lats.min(), lats.max(), N_grid)
    lons_new = np.linspace(lons.min(), lons.max(), N_grid)
    lon_new_2d, lat_new_2d = np.meshgrid(lons_new, lats_new)

    # copy to prevent overwriting: may be unnecessary    
    ds_interp = xa.Dataset()
    # for data in data_vars
    for var_name in tqdm(ds.data_vars):
        data = ds[var_name].values.flatten()

        nan_mask = ~np.isnan(data)
        points_no_nans = point_centres[nan_mask]
        data_no_nans = data[nan_mask]

        data_interp = griddata(points_no_nans, data_no_nans, (lat_new_2d, lon_new_2d), method="linear")
        ds_interp[var_name] = (["lat", "lon"], data_interp)
    
    return ds_interp
    

def interpolate_data(args):
    """Function to interpolate data for a given variable."""
    var_name, point_centres, data_vars, lat_new_2d, lon_new_2d = args
    
    # Retrieve the variable's data
    data = data_vars[var_name].data.flatten()
    
    # Mask NaN values
    nan_mask = ~np.isnan(data)
    points_no_nans = point_centres[nan_mask]
    data_no_nans = data[nan_mask]
    
    # Perform interpolation
    data_interp = griddata(points_no_nans, data_no_nans, (lat_new_2d, lon_new_2d), method="linear")
    return (var_name, data_interp)

def irregular_to_regular_parallel(ds: xa.Dataset, N_grid: int) -> xa.Dataset:
    """Interpolate irregular grid data to regular grid data."""
    # Extract coordinates
    lats = ds["lat"].values
    lons = ds["lon"].values

    point_centres = np.array([lats.flatten(), lons.flatten()]).T

    # New grid
    lats_new = np.linspace(lats.min(), lats.max(), N_grid)
    lons_new = np.linspace(lons.min(), lons.max(), N_grid)
    lon_new_2d, lat_new_2d = np.meshgrid(lons_new, lats_new)

    # Initialize an empty dataset for interpolated data
    ds_interp = xa.Dataset()

    # Prepare arguments for parallel processing
    args = [(var_name, point_centres, ds.data_vars, lat_new_2d, lon_new_2d) for var_name in ds.data_vars]
    
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        results = pool.map(interpolate_data, args)

    # Collect the results and populate the interpolated dataset
    for var_name, data_interp in results:
        ds_interp[var_name] = (["lat", "lon"], data_interp)

    return ds_interp


def test_rearrange_prism_nc(fp: Path | str) -> xa.Dataset:
    """Rearrange the NetCDF data from the given file path into a new dataset."""
    
    # Open NetCDF file and read attributes
    with nc4.Dataset(fp, "r") as temp_nc:
        attributes = temp_nc.__dict__

    # Open datasets for navigation data and derived data
    nav_data = xa.open_dataset(fp, group="navigation_data")
    derived_data = xa.open_dataset(fp, group="derived_data")
    
    # Extract coordinates
    lats = nav_data["latitude"]
    lons = nav_data["longitude"]

    # Define the variables to include
    vars_2d = ["coral", "algae", "sand", "depth", "GPP", "G"]
    
    # Initialize data_vars dictionary
    data_vars = {var_name: (['lat', 'lon'], np.array(derived_data[var_name].data)) 
                 for var_name in vars_2d}
    
    # Create new dataset with data_vars and coordinates
    ds = xa.Dataset(
        data_vars=data_vars,
        coords={
            'lat': (['lat', 'lon'], lats.data),
            'lon': (['lat', 'lon'], lons.data)
        }
    )
    
    # Assign attributes from nc4
    ds.attrs = attributes
    
    return ds


# useful shape checking for non-2d vars
def old_gpt_rearrange_prism_nc(fp: Path | str) -> xa.Dataset:
    """Rearrange the NetCDF data from the given file path into a new dataset."""
    
    # Open NetCDF file
    with nc4.Dataset(fp, "r") as temp_nc:
        # Read group names to ensure the file structure is correct
        groups = list(temp_nc.groups.keys())
        print(f"Groups in the file: {groups}")
    
    # Open datasets for navigation data and derived data
    nav_data = xa.open_dataset(fp, group="navigation_data")
    derived_data = xa.open_dataset(fp, group="derived_data")
    
    # Extract coordinates
    lats = nav_data["latitude"]
    lons = nav_data["longitude"]
    
    # Ensure coordinate arrays have the right shape
    lat_shape = lats.shape
    lon_shape = lons.shape
    print(f"Latitude shape: {lat_shape}")
    print(f"Longitude shape: {lon_shape}")
    
    # Initialize data_vars dictionary
    data_vars = {}
    
    # Populate data_vars with all variables from derived_data
    for var_name in derived_data.data_vars:
        var_data = derived_data[var_name].data
        
        # Debugging: Check the shape of the variable data
        print(f"{var_name} data shape: {var_data.shape}")
        
        # Ensure the variable data has the correct shape
        if var_data.shape[-2:] != lat_shape or var_data.shape[-2:] != lon_shape:
            raise ValueError(f"Shape mismatch for variable '{var_name}'")

        # Convert NaN values to a compatible format for xarray
        var_data = np.array(var_data)  # Ensure data is a NumPy array to handle NaNs
        
        data_vars[var_name] = (['lat', 'lon'], var_data)
    
    # Create new dataset
    ds = xa.Dataset(
        data_vars=data_vars
    )
    
    # Assign coordinates to the dataset
    ds = ds.assign_coords(
        lat=(("lat", "lon"), lats),
        lon=(("lat", "lon"), lons)
    )
    
    return ds