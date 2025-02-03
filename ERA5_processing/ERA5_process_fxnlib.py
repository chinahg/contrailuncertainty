### FUNCTION LIBRARY for ERA5_process.ipynb ###
import numpy as np
import os
import pandas as pd
import tqdm as tqdm

def convert_to_datetime(year, month, day, time):
    """
    Convert year, month, day, and time to a numpy datetime64 object.

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    time : str
        Time in the format 'HH:MM:SS'.

    Returns
    -------
    np.datetime64
        Numpy datetime64 object.
    """
    date_str = f"{year}-{month}-{day}T{time}"
    return np.datetime64(date_str)

def process_GRUAN_filename(filepath):
    """
    Process GRUAN filename and extract site name and time information.

    Parameters
    ----------
    string : str
        GRUAN filename.

    Returns
    -------
    list
        List containing site name and time information.
    """

    # Stripping GRUAN datetime and site data from file names so we can match with ERA5 data
    GRUAN_file_names = os.path.basename(filepath)

    split_string = GRUAN_file_names.split("_")
    site_name = split_string[0][:3]

    date_time_raw = split_string[4]
    year = date_time_raw[:4]
    month = date_time_raw[4:6]
    day = date_time_raw[6:8]

    time = date_time_raw[-6:]
    time = time[::-1]
    time = ":".join(time[i:i+2] for i in range(0, len(time), 2))
    time = time[::-1]

    time = convert_to_datetime(year, month, day, time)
    GRUAN_site_info = [site_name, time, filepath]
    return GRUAN_site_info

def match_files(GRUAN_date_sites):
    """
    Match GRUAN dates with ERA5 file names.

    This function takes a list of GRUAN dates and sites, and a list of ERA5 file paths.
    It matches the GRUAN dates with the corresponding ERA5 file names and returns a list of matching ERA5 file names.

    Parameters
    ----------
    GRUAN_date_sites : list
        List of GRUAN dates and sites.
    ERA5_file_paths : list
        List of ERA5 file paths.

    Returns
    -------
    tuple
        A tuple containing three lists:
        - List of matching ERA5 file names.
        - List of GRIB file names to convert to NetCDF format.
        - List of GRIB file names to download.
    """
    GRUAN_date_formatted = []
    ERA5_data_matching = []
    ERA5_name_only = []
    files2convert = []
    files2download = []
    print("Matching GRUAN dates with ERA5 files...")

    ERA5_directory = '/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads'
    ERA5_file_names = []
    for root, dirs, files in os.walk(ERA5_directory):
        for file in files:
            ERA5_file_names.append(os.path.basename(os.path.join(root, file)))
    print(ERA5_file_names[0])

    for i in tqdm.tqdm(range(len(GRUAN_date_sites))):
        GRUAN_date_formatted = str(GRUAN_date_sites[i][1]).replace("-", "_")[:10]
        GRUAN_date_nc = GRUAN_date_formatted + '.nc'
        GRUAN_date_grib = GRUAN_date_formatted + '.grib'

        if GRUAN_date_nc in ERA5_file_names:
            filename = [ERA5_file_names[ERA5_file_names.index(GRUAN_date_nc)]]
            site_info = GRUAN_date_sites[i]
            ERA5_data_matching.append((filename + site_info))  # Save all relevant site data for the date
            ERA5_name_only.append(filename[0])  # Only save name of file for later use

        elif GRUAN_date_grib in ERA5_file_names:  # If a GRIB file exists but not a NetCDF file for the date of interest
            files2convert.append(GRUAN_date_grib)  # Save filename to reference later when converting files to nc
        else:
            files2download.append(GRUAN_date_formatted)  # Save filename to reference later when downloading files
    
    # Get rid of duplicates and sort
    ERA5_file_names_matching = list(set(ERA5_name_only)) 
    ERA5_file_names_matching.sort()

    files2convert = list(set(files2convert))
    files2download = list(set(files2download))

    print("Number of unconverted GRIB files: ", len(files2convert))
    print("Number of files to download: ", len(files2download))
    print("Number of matching files: ", len(ERA5_file_names_matching))
    
    return ERA5_data_matching, ERA5_file_names_matching, files2convert, files2download

def get_coordinates(site_code):
    """
    Retrieves the latitude and longitude coordinates for a given site code.

    Parameters:
    - site_code (str): The code of the site for which coordinates are to be retrieved.

    Returns:
    - tuple: A tuple containing the latitude and longitude coordinates of the site.
             If the site code is not found, returns None.
    """
    df = pd.read_excel('/home/chinahg/GCresearch/contrailuncertainty/GRUAN_processing/GRUAN_site_data.xlsx', sheet_name='30-60lat')
    row = df[df['Code'] == site_code]
    if len(row) > 0:
        latitude = row['Latitude'].values[0]
        longitude = row['Longitude'].values[0]
        return latitude, longitude
    else:
        return None, None

def press2alt(pressure):
    """
    Convert pressure to altitude.

    Parameters
    ----------
    pressure : Union[int, np.ndarray]
        Pressure in Pascal.

    Returns
    -------
    Union[float, np.ndarray]
        Altitude in meters.
    """
    L = -6.5*10**-3
    P0 = 101325
    T0 = 288.15
    R = 287.053
    g = 9.81

    altitudes = np.zeros_like(pressure)

    if type(pressure)==int:
        return (T0/L)*((pressure*100/P0)**(-R*L/g) -1)
    else:
        for i in range(len(pressure)):
            altitudes[i] = (T0/L)*((pressure[i]*100/P0)**(-R*L/g) -1)

        return altitudes

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

def check_supersat(RH_i, altitudes, alt_lower, alt_upper):
    """
    Check if the given altitudes and relative humidity (RH) values indicate supersaturation.

    Parameters:
    altitudes (float): The altitude value to check.
    RH_i (float): The relative humidity value to check.
    alt_lower (float): The lower limit of the altitude range for supersaturation check.
    alt_upper (float): The upper limit of the altitude range for supersaturation check.

    Returns:
    bool: True if the altitudes and RH indicate supersaturation within the specified altitude range, False otherwise.
    """
    if altitudes >= alt_lower and altitudes <= alt_upper and RH_i >= 100:
        return True
    else:
        return False