# This script calculates the relative humidity with respect to ice (RHi) and water (RHw) from ERA5 data and appends them to the dataset.
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib
import pandas as pd
import netCDF4 as nc
import os
import numpy as np
import tqdm

# Load the CSV file and save each entry as an array entry
all_ERA5_files_to_process = pd.read_csv('/home/chinahg/GCresearch/contrailuncertainty/Met_processing/era5_files.csv', header=None).squeeze().tolist()
start_index = int(sys.argv[1])  # Start index from command line argument
end_index = int(sys.argv[2])    # End index from command line argument
ERA5_files_to_process = all_ERA5_files_to_process[start_index:end_index]
i = 0

print(f"Processing files from {start_index} to {end_index}...")
print(ERA5_files_to_process[0:5])

message = "Checking RHi and RHw in ERA5 files...\nChecking if keys for pressure and time are up to date ..."
for file in tqdm.tqdm(ERA5_files_to_process, desc=message):
    print(f"Processing file {file}...")

    i = i+1
    # Define the path to the ERA5 file
    year = file.split('_')[0]  # Extract the year from the filename
    path2file = f'/home/chinahg/GCresearch/ERA5_downloads/{year}/{file}'

    # Check if the file exists
    if not os.path.exists(path2file):
        print(f"File {path2file} does not exist.")
        continue
    # Check if the file has RH_i and RH_w variables
    try:
        # Open the ERA5 dataset in append mode
        ds_ERA5 = nc.Dataset(path2file, mode='r+')

    except Exception as e:
        # If an Exception occurs (e.g., file cannot be opened), print the error and skip the file
        print(f"Skipping file: {file} due to error: {e}")
        continue  # Skip this file and move to the next one
    
    # Rename the time variables to new nomenclature account for files downloaded before Fall 2024
    if 'time' in ds_ERA5.dimensions.keys():
        try:
            ds_ERA5.renameDimension('time', 'valid_time')
            refresh = True
            print(f"Renamed 'time' dimension in {file} to 'valid_time'.")
        except Exception as e:
            print(f"Error renaming 'time' dimension in {file}: {e}")
            ds_ERA5.close()
            continue
    else:
        print(f"File {file} already contains 'valid_time' dimension.")

    if 'time' and not 'valid_time' in ds_ERA5.variables.keys():
        try:
            ds_ERA5.renameVariable('time', 'valid_time')
            refresh = True
            print(f"Renamed 'time' variable in {file} to 'valid_time'.")
        except Exception as e:
            print(f"Error renaming 'time' variable in {file}: {e}")
            ds_ERA5.close()
            continue
    else:
        print(f"File {file} already contains 'valid_time' variable.")


    # Rename the pressure variables to new nomenclature account for files downloaded before Fall 2024
    if "isobaricInhPa" in ds_ERA5.dimensions.keys():
        try:
            ds_ERA5.renameDimension('isobaricInhPa', 'pressure_level')
            refresh = True
            print(f"Renamed 'isobaricInhPa' dimension in {file} to 'pressure_level'.")
        except Exception as e:
            print(f"Error renaming 'isobaricInhPa' dimension in {file}: {e}")
            ds_ERA5.close()
            continue
    else:
        print(f"File {file} already contains 'pressure_level' dimension.")
    
        # Rename the pressure variables to new nomenclature account for files downloaded before Fall 2024
    if "isobaricInhPa" in ds_ERA5.variables.keys():
        try:
            ds_ERA5.renameVariable('isobaricInhPa', 'pressure_level')
            refresh = True
            print(f"Renamed 'isobaricInhPa' variable in {file} to 'pressure_level'.")
        except Exception as e:
            print(f"Error renaming 'isobaricInhPa' variable in {file}: {e}")
            ds_ERA5.close()
            continue
    else:
        print(f"File {file} already contains 'pressure_level' variable.")


    # Check if the file already contains RH_i and RH_w variables
    if 'RH_i' in ds_ERA5.variables.keys() and 'RH_w' in ds_ERA5.variables.keys():
        print(f"File {path2file} already contains RH_i and RH_w variables. Continuing to next file.")
        ds_ERA5.close()
        continue
    else:
        print(f"File {path2file} does not contain RH_i and RH_w variables.")
    
    # if refresh == True:
    #     ds_ERA5.close()
    #     ds_ERA5 = nc.Dataset(path2file, mode='r+')

    print(f"Processing file: {file}, {i}/{end_index-start_index}")
    print(f"Variables stored in {file}: {ds_ERA5.variables.keys()}")

    # Extract necessary variables
    T = ds_ERA5.variables['t'][:]  # Temperature in Kelvin
    q = ds_ERA5.variables['q'][:]  # Specific humidity
    pres = ds_ERA5.variables['pressure_level'][:] * 100 # Pressure in Pa

    T0 = 273.15  # Reference temperature in Kelvin
    
    # Calculate saturation vapor pressure with respect to water
    P_sat_w = lib.compute_Psat_w(T)

    # Calculate RH_w using specific humidity
    RH_w = 0.263 * pres[:, None, None] * q * np.exp((17.67 * (T - T0)) / (T - 29.65)) ** (-1)

    # Calculate saturation vapor pressure with respect to ice
    P_sat_i = lib.compute_Psat_i(T)

    # Calculate RH_i
    RH_i = RH_w * P_sat_w / P_sat_i

    # Append RH_i and RH_w to the dataset
    RH_i_var = ds_ERA5.createVariable('RH_i', 'f4', ('valid_time', 'pressure_level', 'latitude', 'longitude'))
    RH_w_var = ds_ERA5.createVariable('RH_w', 'f4', ('valid_time', 'pressure_level', 'latitude', 'longitude'))

    RH_i_var[:] = RH_i
    RH_w_var[:] = RH_w

    # Save the modified dataset back to the file
    ds_ERA5.close()