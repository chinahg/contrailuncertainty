"""
This code retrieves and processes data from ERA5 and GRUAN sources. It creates a list of ERA5 file names, extracts the date and site information from GRUAN file names, and imports latitude and longitude locations of GRUAN sites. Then, it picks out the ERA5 files that match the GRUAN dates.

The code saves the list of files to download and convert to CSV files. It then processes the ERA5 data for each matching file, calculates relative humidity and the marine boundary layer depth, and saves the data to corresponding ERA5 indices. Finally, it saves the processed data to CSV files.

Functions:
- process_GRUAN_filename: Extracts the date and site information from GRUAN file names.
- match_files: Matches the GRUAN dates with the ERA5 files.
- press2alt: Converts pressure to altitude.
- compute_Psat_w: Computes the saturation vapor pressure over water.
- compute_Psat_i: Computes the saturation vapor pressure over ice.
- check_supersat: Checks if supersaturation occurs.

Parameters:
- ERA5_file_names (list): List of ERA5 file names.
- GRUAN_date_sites (list): List of GRUAN date and site information.
- matching_data (list): List of matching ERA5 file names, GRUAN site names, and GRUAN datetime objects.
- files2convert (list): List of files to convert.
- files2download (list): List of files to download.
- cruiseRH (list): List of cruise relative humidity values.
- MLD_array_alt (list): List of marine boundary layer depth values.
- MLD_top (float): Upper altitude limit of the marine boundary layer.
- MLD_bottom (float): Lower altitude limit of the marine boundary layer.
- press_upper (int): Upper pressure limit.
- press_lower (int): Lower pressure limit.
- alt_upper (float): Upper altitude limit.
- alt_lower (float): Lower altitude limit.
- num_files (int): Number of matching files to process.
- redownload (list): List of corrupted files to redownload.
- current_year (str): Current year being processed.
- path2file (str): Path to the current ERA5 file.
- ds_ERA5 (xarray.Dataset): Dataset containing ERA5 data.
- altitudes (numpy.ndarray): Array of altitudes.
- current_site_name (str): Current GRUAN site name.
- latitude (float): Latitude of the current site.
- longitude (float): Longitude of the current site.
- time (datetime.datetime): Datetime object of the current GRUAN date and time.
- RH_ERA5 (numpy.ndarray): Array of relative humidity values.
- T (numpy.ndarray): Array of temperature values.
- q (numpy.ndarray): Array of specific humidity values.
- p (numpy.ndarray): Array of pressure values.
- T0 (float): Reference temperature.
- RH_i (numpy.ndarray): Array of relative humidity values with respect to ice.
- RH_w (numpy.ndarray): Array of relative humidity values with respect to water.
- P_sat_w (numpy.ndarray): Array of saturation vapor pressure over water.
- P_sat_i (numpy.ndarray): Array of saturation vapor pressure over ice.
- RH_len (int): Length of the relative humidity array.
- supersat_bool (bool): Boolean indicating if supersaturation occurs.
- MLD_index (int): Index of the marine boundary layer depth.
- i (int): Index for iterating over the relative humidity array.
- j (int): Index for iterating over the list of relative humidity values.
- f (int): Index for iterating over the temperature array.
- i (int): Index for iterating over the relative humidity array.
- matching_data (list): List of matching ERA5 file names, GRUAN site names, GRUAN datetime objects, cruise relative humidity values, and marine boundary layer depth values.
- MLD_file_path (str): File path for the marine boundary layer depth CSV file.
- RH_file_path (str): File path for the cruise relative humidity CSV file.
- ERA5_processed_path (str): File path for the processed ERA5 CSV file.
- headerList (list): List of header names for the processed ERA5 CSV file.

Returns:
None
"""
# Libraries
import numpy as np
import os as os
import glob
import pandas as pd
import xarray as xr
import tqdm
import csv
import ERA5_process_fxnlib as lib
import pandas as pd

############################################################################################################

# This section retrieves and processes data from ERA5 and GRUAN sources. 
# It creates a list of ERA5 file names, extracts the date and site information from GRUAN file names, and imports latitude and longitude locations of GRUAN sites. ...
# Finally, it picks out the ERA5 files that match the GRUAN dates.

# Make array of .nc filenames (grib)
ERA5_file_names = []
GRUAN_date_sites = []
years = np.linspace(2005,2012,num=8,dtype=int)

# Make list of all ERA5 file names
# Need to loop as glob is not capturing all file names
for i in range(len(years)):
    ERA5_file_names = ERA5_file_names + glob.glob('/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'+str(years[i])+'/*.grib', recursive=True)
    ERA5_file_names = ERA5_file_names + glob.glob('/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'+str(years[i])+'/*.nc', recursive=True)
ERA5_file_names = list(map(os.path.basename, ERA5_file_names))
ERA5_file_names.sort()

# Stripping GRUAN datetime and site data from file names so we can match with ERA5 data
GRUAN_paths = glob.glob('/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/' + '/**/*.nc', recursive=True)
GRUAN_file_names = [os.path.basename(file) for file in GRUAN_paths]
GRUAN_num_files = len(GRUAN_file_names)

for j in range(GRUAN_num_files):
    GRUAN_date_sites.append(lib.process_GRUAN_filename(GRUAN_file_names[j], GRUAN_paths[j]))

# Create a list of ERA5 files that match the GRUAN dates
matching_data, matching_files, files2convert, files2download = lib.match_files(GRUAN_date_sites, ERA5_file_names) # Format for matching_data is [ERA5 file name, GRUAN site name, GRUAN datetime object]

if len(files2convert) == 0 and len(files2download) == 0:
    print("All files have been converted and downloaded. Continuing...")
else:
    print("Unconverted files and files to download should be 0 before continuing!")

############################################################################################################

# Save data to CSV file
files2download = list(set(files2download))
files2download.sort()
files2download = [str(date).replace("_", "").replace(".nc", "") for date in files2download]

file2download_path = 'files2download.csv'

# Open the file in write mode
with open(file2download_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    
    # Write the array to the CSV file
    writer.writerow(files2download)

# Save grib filenames to CSV file
files2convert = list(set(files2convert))
files2convert.sort()

files2convert_path = 'files2convert.csv'

# Open the file in write mode
with open(files2convert_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    
    # Write the array to the CSV file
    writer.writerow(files2convert)

############################################################################################################

# 10km is approx 265 hPa
print("Started!")

cruiseRH = []
MLD_array_pres = []
MLD_top = 0
MLD_bottom = 0
MLD_array_alt = []
press_upper = 250 #hPa
press_lower = 290 #hPa
alt_upper = lib.press2alt(press_upper) # Upper altitude limit [m]
alt_lower = lib.press2alt(press_lower) # Lower altitude limit [m]
num_files = len(matching_files)
redownload = []

## FOR TESTING
num_files = 10

for k in tqdm.tqdm(range(num_files)): # Look through all matching files
    print("file: ",k)
    current_year = matching_data[k][0][0:4]
    path2file = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/"+str(current_year)+"/"+matching_data[k][0]

    # Read file in the location of interest
    # View data for a single day
    try:
        ds_ERA5 = xr.open_dataset(path2file,engine='netcdf4')
    except:
        ds_ERA5.close()
        # Save the name of the corrupted file to redownload later
        redownload.append(path2file)
        continue
    
    altitudes = lib.press2alt(ds_ERA5.isobaricInhPa.to_numpy())

    # Get site name, coordinates, and time
    current_site_name = matching_data[k][1]
    latitude, longitude = lib.get_coordinates(current_site_name)
    
    if latitude == None or longitude == None:
        continue
    
    time = matching_data[k][2]

    # Assign ERA5 data to arrays
    RH_ERA5 = ds_ERA5.r.sel(time=time, latitude=latitude, longitude=longitude, method='nearest').to_numpy() # [%] Water relative humidity
    T = ds_ERA5.t.sel(time=time, latitude=latitude, longitude=longitude, method='nearest').to_numpy() # [K] Temperature
    print("temperature:",T)
    q = ds_ERA5.q.sel(time=time, latitude=latitude, longitude=longitude, method='nearest').to_numpy() # [kg/kg] Specific humidity
    p = (ds_ERA5.isobaricInhPa).to_numpy()*100 # [Pa] Pressure
    T0 = 273.15 # [K] Reference temperature
    RH_i = np.zeros(len(T))
    RH_w = np.zeros(len(T))

    # Calculate relative humidity
    for f in range(len(T)):
        P_sat_w = lib.compute_Psat_w(T[f]) # [Pa] Bolton 1980
        P_sat_i =  lib.compute_Psat_i(T[f]) # [Pa] Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
        RH_w[f] = 0.263*p[f]*q[f]*(np.exp((17.67*(T[f]-T0))/(T[f]-29.65))**(-1)) # [%] Relative humidity wrt water from specific humidity (WMO No.8 Guide to Instruments and Methods of Observation, Vol 1 Measurement of Meteorological Variables, ANNEX 4.B. FORMULAE FOR THE COMPUTATION OF MEASURES OF HUMIDITY)
        
    RH_i = RH_w*P_sat_w/P_sat_i # [%] Relative humidity wrt ice from relative humidity wrt water
        
    RH_len = len(RH_i)

    for i in range(RH_len): # look through all RH datapoints
        if altitudes[i] >= alt_lower and altitudes[i] <= alt_upper and RH_i[0:i] < 100: # if all values of RH_i are less than 100
            print("No supersaturation in region of interest")
            MLD_array_alt.append(0)
            cruiseRH.append(RH_i[i])
            break

        if altitudes[i] >= alt_lower and altitudes[i] <= alt_upper and RH_i[i] >= 100: # if pressure is approx 265 hPa and RH > 100% record RH
            print("Supersaturation in region of interest")
            # Save RH value where MLD starts
            cruiseRH.append(RH_i[i])

            # Save altitude where supersaturated RH starts
            MLD_top = altitudes[i]

            for j in range(i): # Look through list of RH under cruise alt and determine MLD
                
                if RH_i[i-j] < 100: # MLD ends when RHi < 100%
                    MLD_index = i-j
                    MLD_bottom = altitudes[MLD_index]
                    break
        
                elif j == i:
                    MLD_bottom = altitudes[0]
                    break
            
            # Now have an array of RH and MLD upper and lower bounds
        
            # Take difference of altitudes to get MLD in [m]
            MLD_array_alt.append(MLD_top-MLD_bottom)
            break
            

    # for i in range(RH_len): # look through all RH datapoints for this date and location
    #     # Save every RH value
    #     cruiseRH.append(RH_i[i])

    #     supersat_bool = lib.check_supersat(RH_i[i], altitudes[i], alt_lower, alt_upper) # Check if supersaturation occurs

    #     if supersat_bool == True:
    #         # Save altitude where supersaturated RH starts
    #         MLD_top = altitudes[i]

    #         for j in range(i): # Look through list of RH under cruise alt and determine MLD

    #             if RH_i[i-j] < 100: # MLD ends when RHi < 100%
    #                 MLD_index = i-j
    #                 MLD_bottom = altitudes[MLD_index]
    #                 break
    #             elif j == i:
    #                 MLD_bottom = altitudes[0]
    #                 break
    #         # Now have an array of RH and MLD upper and lower bounds
    #         # Take difference of altitudes to get MLD in [m]
    #         MLD_array_alt.append(MLD_top-MLD_bottom)
        
    #     else: # If no supersaturation, append 0 to MLD array
    #         MLD_array_alt.append(0)
    
    ds_ERA5.close()

# MLD_array_alt is appended to for each file, never overwritten
# cruiseRH is appended to for each file, never overwritten

############################################################################################################

# Save data to it's correpsonding ERA5 index
# matching_data is a list of lists, each list contains the ERA5 file name, GRUAN site name, and GRUAN datetime object
# We append the cruiseRH and MLD_array_alt to the matching_data list
# matching_data is now a list of lists, each list contains the ERA5 file name, GRUAN site name, GRUAN datetime object, cruiseRH [%], and MLD_array_alt [m]
for i in range(len(cruiseRH)):
    matching_data[i].append(cruiseRH[i])
    matching_data[i].append(MLD_array_alt[i])

# Sort matching data by the GRUAN site name (to match GRUAN data order)
matching_data.sort(key=lambda x: x[1])

############################################################################################################

# Save the data so we don't have to process it again
# Specify the file path
ERA5_processed_path = '/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_processed.csv'

# Delete the file if it already exists
if os.path.exists(ERA5_processed_path):
    os.remove(ERA5_processed_path)

headerList = ['ERA5_file_name', 'GRUAN_site_name', 'GRUAN_datetime', 'GRUAN_path', 'cruiseRH', 'MLD'] 
# Convert matching_data to a pandas DataFrame
df = pd.DataFrame(matching_data, columns=headerList)

# Save the DataFrame to the CSV file
df.to_csv(ERA5_processed_path, index=False)

print("Finished ERA5 Processing!")