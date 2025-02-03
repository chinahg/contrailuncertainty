import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from datetime import datetime, timedelta
import tqdm
import met_matching_fxnlib as fxn

# Base directories
gruan_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/'
era5_base_dir = '/home/chinahg/GCresearch/ERA5_downloads/'

# Arrays to store the paths
gruan_file_paths = []
era5_file_paths = []

# Recursively find all GRUAN files and construct corresponding ERA5 file paths
for root, dirs, files in os.walk(gruan_base_dir):
    for file in files:
        if file.endswith('.nc'):
            gruan_file_path = os.path.join(root, file)
            era5_file_path = fxn.construct_era5_path(gruan_file_path)
            gruan_file_paths.append(gruan_file_path)
            era5_file_paths.append(era5_file_path)

# Sort the paths
gruan_file_paths.sort()
era5_file_paths.sort()
start_file = 1
end_file = 1 #16132 #24198 #len(gruan_file_paths) # Number of files to process per slurm batch

# Define the dimensions
days = pd.date_range('2005-01-01', '2021-12-31')  # From 2005 to the end of 2021
E_latitudes = np.linspace(30, 60, int((60 - 30) / 0.25) + 1)  # 0.25 degree increments between 30 and 60 degrees
E_longitudes = np.linspace(-180, 180, int(360 / 0.25) + 1)  # 0.25 degree increments

# Initialize lists to store data
valid_combinations = []
G_data_list = []
E_RHi = []

for j in tqdm.tqdm(range(start_file, end_file)):  # Iterate through the dates
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
    G_T = fxn.fill_nan_with_next(G_data.variables['T'][:])
    G_RHi = G_data.variables['rh_i'][:]
    G_pres = fxn.fill_nan_with_next(G_data.variables['pres'][:])
    G_MLD = fxn.calculate_MLD(G_alt, G_pres, G_RHi, G_T, "GRUAN")

    E_datetime = E_data.variables['time'][:].values
    E_pres = E_data.variables['level'][:].values
    E_T = E_data.variables['temp'][:].values
    E_alt = np.array(fxn.press2alt(E_data.sel(latitude=G_lat[0], longitude=G_lon[0], time=G_datetime[0], method='nearest')['isobaricInhPa']))

    # Have to average over the GRUAN data to regrid it to ERA5 size 
    indexer = len(G_alt)
    for i in range(indexer):
        E_RHi.append(np.array(E_data.sel(latitude=G_lat[i], longitude=G_lon[i], isobaricInhPa=fxn.alt2press(G_alt[i]), time=G_datetime[i], method='nearest')['RH_i']))

    # Find the index ranges for where the G_alt values fall within the E_alt values
    index_ranges = []
    for i in range(len(G_alt)):
        for j in range(len(E_alt) - 1):
            if G_alt[i] >= E_alt[j] and G_alt[i] < E_alt[j + 1]:
                index_ranges.append(j)
                break
        if G_alt[i] >= E_alt[-1]:
            index_ranges.append(j)

    # Average the RHi values in E_RHi based on index_ranges
    E_RHi_avg = []
    for i in range(len(E_alt)):
        indices = [idx for idx, val in enumerate(index_ranges) if val == i]
        if indices:
            avg_rhi = np.mean([E_RHi[idx] for idx in indices])
            E_RHi_avg.append(avg_rhi)
        else:
            E_RHi_avg.append(np.nan)  # If no indices found, append NaN

    E_RHi_avg = np.array(E_RHi_avg)

    E_MLD = np.array(fxn.calculate_MLD(E_alt, E_pres, E_RHi_avg, E_T, "ERA5"))
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
        'E_RHi': E_RHi_avg,
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
df.to_parquet('/home/chinahg/GCresearch/contrailuncertainty/Met_processing/final_met_data.parquet')
print("Data saved to final_met_data.parquet.")