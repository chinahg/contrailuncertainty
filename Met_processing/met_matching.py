########################################################################################################################################
# Imports
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from datetime import datetime, timedelta
import tqdm
import importlib.util
# Import functions from the pipeline_fxn_lib.py script
function_library_path = "/home/chinahg/GCresearch/contrailuncertainty/start_here/pipeline_fxn_lib.py"
spec = importlib.util.spec_from_file_location("fxn", function_library_path)
fxn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fxn)

# Import the user defined variables from the preprocess_pipeline.py script
import_path = "/home/chinahg/GCresearch/contrailuncertainty/preprocess_pipeline.py"
pipeline = importlib.util.spec_from_file_location("preprocess_pipeline", import_path)
preprocess_pipeline = importlib.util.module_from_spec(pipeline)
pipeline.loader.exec_module(preprocess_pipeline)

# Import only the specific variables you need
start_file = getattr(preprocess_pipeline, 'start_file', None)  # Fallback to None if not found
end_file = getattr(preprocess_pipeline, 'end_file', None)
era5_file_paths = getattr(preprocess_pipeline, 'era5_file_paths', None)
gruan_file_paths = getattr(preprocess_pipeline, 'gruan_file_paths', None)
batch = getattr(preprocess_pipeline, 'batch', None)
########################################################################################################################################

### Begin main script
print(f"Processing files from {start_file} to {end_file}...")
print(f"First ERA5 file: {era5_file_paths[start_file]}")

# Define the dimensions
days = pd.date_range('2005-01-01', '2021-12-31') # From 2005 to the end of 2021
E_latitudes = np.linspace(30, 60, int((60 - 30) / 0.25) + 1)  # 0.25 degree increments between 30 and 60 degrees
E_longitudes = np.linspace(-180, 180, int(360 / 0.25) + 1)  # 0.25 degree increments

# Initialize lists to store data
valid_combinations = []
G_data_list = []

for j in tqdm.tqdm(range(start_file, end_file)):  # Iterate through the dates
    # Initialize lists to store data for the current file
    E_RHi_regrid = []

    # Open the ERA5 file
    E_file_path = era5_file_paths[j]
    E_data = xr.open_dataset(E_file_path)

    # Open the GRUAN file
    G_file_path = gruan_file_paths[j]
    G_data = nc.Dataset(G_file_path)

    # Extract the base time from the G_data attributes
    base_time_str = G_data.variables['time'].units.split('since ')[1]
    base_time = datetime.strptime(base_time_str, '%Y-%m-%dT%H:%M:%S')

    G_datetime = [base_time + timedelta(seconds=float(sec)) for sec in G_data.variables['time'][:]]  # Convert the time variable from seconds since base_time to datetime objects
    G_site = os.path.basename(G_file_path).split('_')[0].split('-')[0]
    G_lat = fxn.fill_nan_with_next(G_data.variables['lat'][:])
    G_lon = fxn.fill_nan_with_next(G_data.variables['lon'][:])
    G_alt = fxn.fill_nan_with_next(G_data.variables['alt'][:])
    G_T = fxn.fill_nan_with_next(G_data.variables['temp'][:])
    G_RHi = G_data.variables['rh_i'][:]
    G_pres = fxn.fill_nan_with_next(G_data.variables['press'][:])
    G_MLD = fxn.calculate_MLD(G_alt, G_pres, G_RHi, G_T, G_lat, G_lon, "GRUAN")

    E_datetime = E_data.variables['time'][:].values
    E_pres = np.array(E_data.variables['isobaricInhPa'][:].values)
    E_T = np.array(E_data.sel(latitude=G_lat[0], longitude=G_lon[0], time=G_datetime[0], method='nearest')['t'])
    E_alt = np.array(fxn.press2alt(E_data.sel(latitude=G_lat[0], longitude=G_lon[0], time=G_datetime[0], method='nearest')['isobaricInhPa']))
    E_lat = float(E_data.latitude.sel(latitude=G_lat[0], method='nearest').values)
    E_lon = float(E_data.longitude.sel(longitude=G_lon[0], method='nearest').values)
    E_RHi = np.array(E_data.sel(latitude=G_lat[0], longitude=G_lon[0], time=G_datetime[0], method='nearest')['RH_i'])

    # Regrid the E_RHi data to match the G_alt data dimension 
    indexer = len(G_alt)
    for i in range(indexer):
        E_RHi_regrid.append(np.array(E_data.sel(latitude=G_lat[i], longitude=G_lon[i], isobaricInhPa=fxn.alt2press(G_alt[i]), time=G_datetime[i], method='nearest')['RH_i']))
    E_RHi_regrid = np.array(E_RHi_regrid)

    # Linearly diffuse the data as a user would when using the ERA5 data
    E_RHi_diffused = fxn.linear_diffusion(E_RHi_regrid)

    E_MLD = np.array(fxn.calculate_MLD(E_alt, E_pres, E_RHi, E_T, E_lat, E_lon, "ERA5"))
    E_alt = np.array(np.unique(E_alt))

    # Create a dictionary for the current data
    current_data = {
        'G_site': G_site,
        'G_lat': G_lat,
        'G_lon': G_lon,
        'G_alt': G_alt,
        'G_T': G_T,
        'G_RHi': G_RHi,
        'G_MLD': G_MLD,
        'G_dt': G_datetime,
        'E_alt': E_alt,
        'E_RHi': E_RHi,
        'E_RHi_diffused': E_RHi_diffused,
        'E_T': E_T,
        'E_MLD': E_MLD,
        'E_dt': E_datetime
    }

    G_data_list.append(current_data)

    # Create a list of non-empty (day, lat, lon) combinations
    E_latitude = float(E_data.latitude.sel(latitude=G_lat[0], method='nearest').values)
    E_longitude = float(E_data.longitude.sel(longitude=G_lon[0], method='nearest').values)

    combo = (G_datetime[0].strftime('%Y-%m-%d'), E_latitude, E_longitude)
    valid_combinations.append(combo)

# Convert G_data_list to a DataFrame
df = pd.DataFrame(G_data_list)

# Squash the MLD binary array into a single string for each row (to allow for parquet conversion)
df['G_MLD'] = df['G_MLD'].apply(lambda x: ','.join(map(str, x)))
df['E_MLD'] = df['E_MLD'].apply(lambda x: ','.join(map(str, x)))

# Convert valid combinations into a MultiIndex
index = pd.MultiIndex.from_tuples(valid_combinations, names=['day', 'E_latitude', 'E_longitude'])

# Check for duplicates in the MultiIndex
duplicates = index.duplicated(keep=False)

# Report duplicates if any
if duplicates.any():
    print("Duplicates found in index:")
    print(index[duplicates])
else:
    print("No duplicates found in index.")

# Set the MultiIndex to the DataFrame
df.set_index(index, inplace=True)

# Save the DataFrame to a parquet file
df.to_parquet('/home/chinahg/GCresearch/contrailuncertainty/Met_processing/final_met_data_'+str(batch)+'.parquet')
print("Data saved to final_met_data.parquet.")