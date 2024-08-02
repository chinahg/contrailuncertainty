"""
GRUAN_process.py

This script processes GRUAN (GCOS Reference Upper-Air Network) radiosonde data to calculate 
cruise relative humidity (RH) and mixed layer depth (MLD) for each file. The processed data 
is then saved to a CSV file.

Author: [China Hagström]

Date: [July 2024]

Requirements:
- numpy
- os
- netCDF4
- xarray
- matplotlib
- seaborn
- glob
- tqdm
- time
- pandas
- csv

Functions:
- press2alt(pressure): Converts pressure to altitude using the lapse rate and other constants.
- compute_Psat_w(T): Returns water liquid saturation pressure in Pascal.
- compute_Psat_i(T): Returns water solid saturation pressure in Pascal.

Usage:
1. Set the years of interest in the 'years' list.
2. Set the file path for the GRUAN site data Excel file in the 'xls' variable.
3. Set the upper and lower pressure limits and calculate the corresponding altitude limits.
4. Run the script.

Output:
- The processed data is saved to a CSV file named 'GRUAN_processed.csv' in the specified file path.

Note:
- Make sure to install the required packages before running the script.
- Ensure that the file paths are correctly set before running the script.
"""

# Imports
import numpy as np
import os as os
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import tqdm
import time
import pandas as pd
import csv

### FUNCTION LIBRARY ###
def press2alt(pressure):
    """
    Converts pressure to altitude using the lapse rate and other constants.

    Parameters
    ----------
    pressure : float
        Pressure in Pascal.

    Returns
    -------
    float
        Altitude in meters.
    """
    L = -6.5*10**-3 # lapse rate [K/m]
    P0 = 101325 # pressure at 0 alt [Pa]
    T0 = 288.15 # temp at 0 alt [K]
    R = 287.053 # gas constant for air [J/kgK]
    g = 9.81 #acceleration due to gravity [m/s^2]
    return (T0/L)*((pressure*100/P0)**(-R*L/g) -1)

def compute_Psat_w(T):
    """
    Returns water liquid saturation pressure in Pascal.
    Source: Sonntag (1994)

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        H2O Liquid saturation pressure in Pascal
    """
    return 100.0 * np.exp(
        -6096.9385 / T
        + 16.635794
        - 0.02711193 * T
        + 1.673952e-5 * T**2
        + 2.433502 * np.log(T)
    )

def compute_Psat_i(T):
    """
    Returns water solid saturation pressure in Pascal.
    Source: Sonntag (1990)

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        H2O solid saturation pressure in Pascal
    """
    return 100.0 * np.exp(
        -6024.5282 / T
        + 24.7219
        + 0.010613868 * T
        - 1.3198825e-5 * T**2
        - 0.49382577 * np.log(T)
    )

#######################

start_time = time.time()
years = ['2005','2006','2007']#,'2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] # Years of interest
matching_data = []

# Get all GRUAN radiosonde files
files = []
for year in years:
    files.extend(glob.glob('/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/'+'/**/'+year+'/*.nc', recursive=True))

files.sort()
num_files = len(files)

filenames = [os.path.basename(file) for file in files]
site_names = [os.path.basename(file).split('-')[0] for file in files]
datetimes = [os.path.basename(file).split('_')[4] for file in files]

for i in range(num_files):
    matching_data.append([filenames[i], site_names[i], datetimes[i], files[i]])

# Constants
# 10km is approx 265 hPa
cruiseRH = []
MLD_array_pres = []
MLD_array_alt = []
MLD_top = 0
MLD_bottom = 0

L = -6.5*10**-3 # lapse rate [K/m]
P0 = 101325 # pressure at 0 alt [Pa]
T0 = 288.15 # temp at 0 alt [K]
R = 287.053 # gas constant for air [J/kgK]
g = 9.81 #acceleration due to gravity [m/s^2]

press_upper = 250
press_lower = 290
alt_upper = (T0/L)*((press_upper*100/P0)**(-L*R/g) - 1) # Upper altitude limit
alt_lower = (T0/L)*((press_lower*100/P0)**(-R*L/g) -1) # Lower altitude limit

for k in tqdm.tqdm(range(num_files)): # Look through all GRUAN radiosonde files
    #print("--- %s minutes elapsed ---" % round(time.time()/60 - start_time/60,2))
    #print("Processing file %s out of %s" % (k,num_files))
    path2file = files[k]

    # Read file in the location of interest
    nc_GRUAN = nc.Dataset(path2file, 'r', format='NETCDF4_CLASSIC')
    altitudes = nc_GRUAN.variables['alt'][:] # [m]
    RH_w = nc_GRUAN.variables['rh'][:]*100 # [%]
    T = nc_GRUAN.variables['temp'][:] # [K]
    P_sat_w = 6.112*np.exp((17.67*(T-273.15))/((T-273.15)+243.5)) *100 # [Pa] Bolton 1980
    P_sat_i =  6.112*np.exp(22.46*(T-273.15)/(272.62 + (T-273.15))) *100 # [Pa] Guide to Meteorological Instruments and Methods of Observation (CIMO Guide) (WMO, 2008)
    RH_i = RH_w*P_sat_w/P_sat_i # [%]
    RH_len = len(RH_i)
    RH_i_placeholder = None

    for i in reversed(range(RH_len)): # Look through all RH values in the file, reverse order as [0] is the lowest altitude
        
        if altitudes[i] >= alt_lower and altitudes[i] <= alt_upper and isinstance(RH_i[i], (np.floating, float)) and RH_i_placeholder is None: # If RH_i_placeholder is None, save the first value of RH
            RH_i_placeholder = RH_i[i]

        if altitudes[i] >= alt_lower and altitudes[i] <= alt_upper and RH_i[i] >= 100: # if pressure is approx 265 hPa and RH > 100% record RH
            # Save RH value where MLD starts
            cruiseRH.append(RH_i[i])

            # Save altitude where supersaturated RH starts
            MLD_top = altitudes[i]

            for j in reversed(range(i)): # Look through list of RH under cruise alt and determine MLD
                
                if RH_i[j] < 100: # MLD ends when RHi < 100%
                    MLD_index = j
                    MLD_bottom = altitudes[MLD_index]
                    break
        
                elif j == 0:
                    MLD_bottom = altitudes[0]
                    break

            # Now have an array of RH and MLD upper and lower bounds
        
            # Take difference of altitudes to get MLD in [m]
            MLD_array_alt.append(MLD_top-MLD_bottom)
            break
        
        elif i == 0: # If no values above 100% RH, save the first value
            cruiseRH.append(RH_i_placeholder) # If no values above 100% RH, save the first value
            MLD_array_alt.append(0) # If no values above 100% RH, save 0 for MLD

    # MLD_array_alt is appended to for each file, never overwritten
    # cruiseRH is appended to for each file, never overwritten

########################################################################################
# Do some error checking, make sure all lists are the same length
if len(matching_data) != num_files:
    print("Error: matching_data and num_files are not the same length")
    print("matching_data length: %s" % len(matching_data))
    print("num_files length: %s" % num_files)
    exit()

if len(cruiseRH) != len(MLD_array_alt):
    print("Error: cruiseRH and MLD_array_alt are not the same length")
    print("cruiseRH length: %s" % len(cruiseRH))
    print("MLD_array_alt length: %s" % len(MLD_array_alt))
    exit()

if len(cruiseRH) != len(matching_data):
    print("Error: cruiseRH and matching_data are not the same length")
    print("cruiseRH length: %s" % len(cruiseRH))
    print("matching_data length: %s" % len(matching_data))
    exit()

if len(cruiseRH) != len(files):
    print("Error: cruiseRH and files are not the same length")
    print("cruiseRH length: %s" % len(cruiseRH))
    print("files length: %s" % len(files))
    exit()
########################################################################################

for m in range(num_files):
    matching_data[m].append(cruiseRH[m])
    matching_data[m].append(MLD_array_alt[m])

# Save the data so we don't have to process it again
# Specify the file path
GRUAN_processed_path = '/home/chinahg/GCresearch/contrailuncertainty/GRUAN_processing/GRUAN_processed.csv'

# Delete the file if it already exists
if os.path.exists(GRUAN_processed_path):
    os.remove(GRUAN_processed_path)

headerList = ['GRUAN_file_name', 'GRUAN_site_name', 'GRUAN_datetime', 'GRUAN_path', 'cruiseRH', 'MLD'] 
# Convert matching_data to a pandas DataFrame
df = pd.DataFrame(matching_data, columns=headerList)

# Save the DataFrame to the CSV file
df.to_csv(GRUAN_processed_path, index=False)

print("Finished GRUAN Processing!")