########################################################################################################################################
# Imports
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import tqdm
import json
import importlib.util

# Import functions from the pipeline_fxn_lib.py script
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib

# Import the saved variables required from the pipeline.py script
args_file_path = sys.argv[1]  # Path to the JSON file containing the arguments
with open(args_file_path, "r") as f:
    args = json.load(f)

# Assign variables
start_file = args['start_index']
end_file = args['end_index']
era5_file_paths = args['era5_file_paths']
gruan_file_paths = args['gruan_file_paths']
batch = args['batch']

###############################################################################################

### Begin main script
print(f"Processing files from {start_file} to {end_file}...")

# Initialize lists to store data
valid_combinations = []
G_data_list = []

for j in tqdm.tqdm(range(start_file, end_file)):  # Iterate through the dates
    print(f"Processing file {j+1}/{end_file}...")
    try:
        # Initialize lists to store data for the current file
        E_RHi_regrid = []

        # Open the ERA5 file
        E_file_path = era5_file_paths[j]
        G_file_path = gruan_file_paths[j]

        # Check if the paths are pointing to equivalent files
        # Extract the date part and insert '_' between year and month (e.g., '202101' -> '2021_01')
        GRUAN_file_date = (
            (G_file_path.split('/')[-1].split('_')[4])[0:4] + '_' +
            (G_file_path.split('/')[-1].split('_')[4])[4:6] + '_' +
            (G_file_path.split('/')[-1].split('_')[4])[6:8]
        )
        ERA5_file_date = (E_file_path.split('/')[-1])[0:10]

        if GRUAN_file_date != ERA5_file_date:
            print(f"Warning: GRUAN file date {GRUAN_file_date} does not match ERA5 file date {ERA5_file_date}.")
            continue
        else:
            print(f"Processing date: {ERA5_file_date}...")

        # Open the datasets
        E_data = xr.open_dataset(E_file_path)
        G_data = xr.open_dataset(G_file_path)

        G_site = os.path.basename(G_file_path).split('_')[0].split('-')[0]
        G_datetime = G_data.variables['time'].values
        print(f"GRUAN initial time: {G_datetime[0]}")
        G_lat = lib.fill_nan_with_next(G_data.variables['lat'][:])
        G_lon = lib.fill_nan_with_next(G_data.variables['lon'][:])
        G_alt = lib.fill_nan_with_next(G_data.variables['alt'][:])
        G_T = lib.fill_nan_with_next(G_data.variables['temp'][:])
        G_RHi = np.array(G_data.variables['rh_i'][:] * 100)  # Convert relative humidity to percentage
        G_pres = lib.fill_nan_with_next(G_data.variables['press'][:])
        G_MLD = lib.calculate_MLD(G_alt, G_pres, G_RHi, G_T, G_lat, G_lon, "GRUAN")

        E_lat = float(E_data.latitude.sel(latitude=G_lat[0], method='nearest').values)
        E_lon = float(E_data.longitude.sel(longitude=G_lon[0], method='nearest').values)
        # E_datetime = np.array(E_data.sel(latitude=E_lat, longitude=E_lon, valid_time=G_datetime[0], method='nearest').valid_time.values)

        # Create a list of non-empty (day, lat, lon) combinations
        E_latitude = float(E_data.latitude.sel(latitude=G_lat[0], method='nearest').values)
        E_longitude = float(E_data.longitude.sel(longitude=G_lon[0], method='nearest').values)

        # Convert the npdatetime object to a datetime object, which can then be converted to string and used as a tuple in the MultiIndex
        combo_date = str(G_datetime[0].astype('M8[ms]').astype(datetime))
        combo = (combo_date, E_latitude, E_longitude)
        
        # Check if the date is a duplicate
        if combo in valid_combinations:
            print(f"Duplicate combination found: {combo}. Skipping...")
            E_data.close()
            G_data.close()
            continue

        # Regrid the E_RHi data to match the G_alt data dimension
        print("Regridding ERA5 data to match GRUAN data...")

        # Initialize arrays to store the regridded data, fill with NaNs to help with debugging
        E_T_regrid = np.full(len(G_alt), np.nan)
        E_RHi_regrid = np.full(len(G_alt), np.nan)
        E_pres_regrid = np.full(len(G_alt), np.nan)

        indexer = len(G_alt) # Number of GRUAN altitudes to map ERA5 data to
        # For every GRUAN altitude/lat/lon/time combination, find the nearest ERA5 data point using interpolation
        for i in range(indexer):

            E_RHi_regrid[i] = (E_data.sel(latitude=G_lat[i], 
                                          longitude=G_lon[i], 
                                          pressure_level=G_pres[i], 
                                          valid_time=G_datetime[i], 
                                          method='nearest')['RH_i'].values)
            E_pres_regrid[i] = (E_data.sel(pressure_level=G_pres[i], 
                                           method='nearest')['pressure_level'].values)

        # Check if the E_T and G_T arrays are the same length: Sanity check
        if len(E_T_regrid) != len(G_T):
            print(f"Warning: Length mismatch between E_T ({len(E_T_regrid)}) and G_T ({len(G_T)}) for {G_site}.")
            continue
        
        # Linearly diffuse the data as a user would when using the ERA5 data
        E_T_diffused = lib.linear_diffusion(E_T_regrid)
        E_RHi_diffused = lib.linear_diffusion(E_RHi_regrid)
        E_pres_diffused = lib.linear_diffusion(E_pres_regrid)
        E_alt_diffused = np.array(lib.press2alt(E_pres_diffused))


        print("Regridded data shape:", E_T_regrid.shape)
        print("Diffused data shape:", E_T_diffused.shape)

        # Check if the E_T and G_T arrays are the same length: Sanity check
        if len(E_T_regrid) != len(G_T):
            print(f"Warning: Length mismatch between E_T ({len(E_T_regrid)}) and G_T ({len(G_T)}) for {G_site}.")
            continue
        
        # Calculate the MLD using the diffused data, returns a binary array with a length of the number of GRUAN altitudes
        E_MLD_diffused = np.array(lib.calculate_MLD(E_alt_diffused, E_pres_diffused, E_RHi_diffused, E_T_diffused, E_lat, E_lon, "ERA5"))

        # Create a dictionary for the current data
        current_data = {
            'G_site': G_site,

            # Arrays have a length equal to the number of sampled datapoints by the RS92 on a single experiment
            'G_lat': G_lat,
            'G_lon': G_lon,
            'G_alt': G_alt,

            'G_T': G_T,
            'G_RHi': G_RHi,
            'G_MLD': G_MLD,
            'G_dt': G_datetime,

            'E_T_diffused': E_T_diffused,
            'E_RHi_diffused': E_RHi_diffused,
            'E_MLD_diffused': E_MLD_diffused
        }

        G_data_list.append(current_data)

        valid_combinations.append(combo)

        print(f"Processed file {j+1}/{end_file} - {G_site} - {combo_date}")
    except Exception as e:
        print(f"Error processing file {j+1}/{end_file}: {e}")
        continue

# Convert G_data_list to a DataFrame
df = pd.DataFrame(G_data_list)

# Prepare the DataFrame for saving
# Squash the MLD binary array into a single string for each row (to allow for parquet conversion)
df['G_MLD'] = df['G_MLD'].apply(lambda x: ','.join(map(str, x)))
df['E_MLD_diffused'] = df['E_MLD_diffused'].apply(lambda x: ','.join(map(str, x)))

# Convert the datetime.datetime objects to numpy datetime64[ns] objects for parquet compatibility
df['G_dt'] = df['G_dt'].apply(lambda x: np.array(x, dtype='datetime64[ns]'))

# Convert valid combinations into a MultiIndex
index = pd.MultiIndex.from_tuples(valid_combinations, names=['Day', 'E_latitude', 'E_longitude'])

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

base_save_dir = '/home/chinahg/GCresearch/contrailuncertainty/Met_processing/parquet_files/'
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)
    
# Save the DataFrame to a parquet file
df.to_parquet(base_save_dir+'met_data_'+str(batch)+'.parquet')
print(f"Data saved to met_data_{batch}.parquet.")