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

# Get all GRUAN radiosonde files
files = glob.glob('/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/' + '/**/*.nc', recursive=True)
num_files = len(files)

xls = pd.ExcelFile('GRUAN_site_data.xlsx')
df_GRUAN_sites = pd.read_excel(xls, '30-60lat')

#get the values for a given column
values = df_GRUAN_sites['Code'].values

### FUNCTION LIBRARY ###
def press2alt(pressure):
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
    T : Union[float, np.ndarray]
        Temperature in Kelvin

    Returns
    -------
    Union[float, np.ndarray]
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
    T : Union[float, np.ndarray]
        Temperature in Kelvin

    Returns
    -------
    Union[float, np.ndarray]
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

# Constants
# 10km is approx 265 hPa
start_time = time.time()

cruiseRH = []
MLD_array_pres = []
MLD_top = 0
MLD_bottom = 0
MLD_array_alt = []

L = -6.5*10**-3 # lapse rate [K/m]
P0 = 101325 # pressure at 0 alt [Pa]
T0 = 288.15 # temp at 0 alt [K]
R = 287.053 # gas constant for air [J/kgK]
g = 9.81 #acceleration due to gravity [m/s^2]

T_t = 273.16 # Ice triple point temp [K]
P_ip = 6.12 # Ice triple point pressure [hPa]

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

    for i in range(RH_len): # look through all RH datapoints for all dates and locations

        if altitudes[i] >= alt_lower and altitudes[i] <= alt_upper and RH_i[i] >= 100: # if pressure is approx 265 hPa and RH > 100% record RH
            # Save supersaturated RH value
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

    

    # MLD_array_alt is appended to for each file, never overwritten
    # cruiseRH is appended to for each file, never overwritten

# Save the data so we don't have to process it again
# Specify the file path
file_path = 'GRUAN_data_processed.csv'

# Open the file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    
    # Write the array to the CSV file
    writer.writerow(MLD_array_alt)
    writer.writerow(cruiseRH)