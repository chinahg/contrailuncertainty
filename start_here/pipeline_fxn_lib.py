##############################################################################################################################################
# Imports
import os
import csv
import glob
import numpy as np
import pandas as pd
import xarray as xr
import tqdm
import netCDF4 as nc
import shutil
import yaml
from numpy.polynomial.hermite_e import HermiteE
import scipy as sc

##############################################################################################################################################

# Function library for pipeline
############################################################################################################################################################
# Part 1: Downloading and formatting GRUAN and ERA5 data

def check_files(GRUAN_base_dir, years):
    # Get all GRUAN radiosonde files
    GRUAN_file_paths = []
    for year in years:
        GRUAN_file_paths.extend(glob.glob(GRUAN_base_dir +'/**/'+year+'/*.nc', recursive=True))

    # Make array of .nc filenames (grib)
    GRUAN_date_sites = []

    # Create a list of objects containing the site name, time, and filepath from each GRUAN file
    GRUAN_num_files = len(GRUAN_file_paths)
    for j in range(GRUAN_num_files):
        GRUAN_date_sites.append(process_GRUAN_filename(GRUAN_file_paths[j]))

    # Create a list of ERA5 files that match the GRUAN dates
    matching_data, matching_files, files2convert, files2download = match_files(GRUAN_date_sites) # Format for matching_data is [ERA5 file name, GRUAN site name, GRUAN datetime object]

    if len(files2convert) == 0 and len(files2download) == 0:
        print("All files have been converted and downloaded. Continuing...")
    else:
        print("Unconverted files and files to download should be 0 before continuing!")
        # Load the first file in ERA5_file_names as a pandas dataframe

    # Save the names of the files to convert to .nc or download to CSV files
    # We will use the CSV list of names to download the missing data and/or convert the file type
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
    
    return files2convert, GRUAN_date_sites

def convert_files(data_dir, files2convert):
    # Convert unconverted GRIB files to netCDF

    for j in range(len(files2convert)):
        current_year = files2convert[j][:4]
        path_grib = data_dir + f"{current_year}/{files2convert[j]}"

        filename = os.path.basename(path_grib)
        year = filename.split('_')[0]
        save_path = data_dir + year +"/"+ filename.replace(".grib", ".nc")
        if os.path.exists(save_path):
            if os.path.exists(path_grib):
                #Delete old grib file
                os.remove(path_grib)
            print(f'Found {save_path}, skipping conversion to NetCDF')

        else: 
            ds = xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath":""})
            print(save_path)
            ds.to_netcdf(save_path)
            ds.close()
            #Delete old grib file
            os.remove(path_grib)
            print(f"File '{path_grib}' deleted successfully.")

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

    ERA5_directory = '/home/chinahg/GCresearch/ERA5_downloads'
    
    ERA5_file_names = []
    for root, dirs, files in os.walk(ERA5_directory):
        for file in files:
            ERA5_file_names.append(os.path.basename(os.path.join(root, file)))

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
    

def add_RHi_RHw(matching_files):
    """
    Calculate RHi and RHw for all the ERA5 files and add as a new variable.
    Parameters
    ----------
    matching_files : list
        List of matching ERA5 file names.
    """ 

    for file in tqdm.tqdm(matching_files, desc="Processing files"):
        # Construct the full path to the file
        current_year = file[:4]
        path2file = f"/home/chinahg/GCresearch/ERA5_downloads/{current_year}/{file}"

        # Open the ERA5 dataset in read mode
        ds_ERA5 = nc.Dataset(path2file, mode='r')

        # Check if RH_i and RH_w already exist in the netcdf file
        if 'RH_i' in ds_ERA5.variables and 'RH_w' in ds_ERA5.variables:
            print(f"Skipping {file} as RH_i and RH_w already exist.")
            ds_ERA5.close()
            continue
        
        # Extract necessary variables
        T = ds_ERA5.variables['t'][:]  # Temperature in Kelvin
        q = ds_ERA5.variables['q'][:]  # Specific humidity
        pres = ds_ERA5.variables['isobaricInhPa'][:] * 100  # Pressure in Pa
        T0 = 273.15  # Reference temperature in Kelvin
        ds_ERA5.close()

        # Calculate saturation vapor pressure with respect to water
        P_sat_w = compute_Psat_w(T)

        # Calculate RH_w using specific humidity
        RH_w = 0.263 * pres[:, None, None] * q * np.exp((17.67 * (T - T0)) / (T - 29.65)) ** (-1)

        # Calculate saturation vapor pressure with respect to ice
        P_sat_i = compute_Psat_i(T)

        # Calculate RH_i
        RH_i = RH_w * P_sat_w / P_sat_i

        # Open the ERA5 dataset in append mode
        ds_ERA5 = nc.Dataset(path2file, mode='a')

        # Append RH_i and RH_w to the dataset
        RH_i_var = ds_ERA5.createVariable('RH_i', 'f4', ('time', 'isobaricInhPa', 'latitude', 'longitude'))
        RH_w_var = ds_ERA5.createVariable('RH_w', 'f4', ('time', 'isobaricInhPa', 'latitude', 'longitude'))
        RH_i_var[:] = RH_i
        RH_w_var[:] = RH_w

        # Save the modified dataset back to the file
        ds_ERA5.close()

# Function to construct ERA5 file path from GRUAN file path
def construct_era5_path(era5_base_dir, gruan_file_path):
    """
    Construct the file path to an ERA5 data file.

    Parameters
    ----------
    era5_base_dir : str
        The base directory where ERA5 data files are stored.
    gruan_file_path : str
        The file path to a GRUAN file, which contains a date string in its name.

    Returns
    -------
    str
        The constructed file path to the corresponding ERA5 data file.

    Example
    -------
    If `era5_base_dir` is "/data/era5" and `gruan_file_path` is "/data/gruan/file_20230101_data.txt",
    the function will return "/data/era5/2023/2023_01_01.nc".
    """

    date_str = os.path.basename(gruan_file_path).split('_')[4][:8]
    era5_file_path = os.path.join(era5_base_dir, date_str[:4], f'{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}.nc')
    return era5_file_path

def fill_nan_with_next(arr):
    """Fill NaN values in the array with the next non-NaN value, if available."""
    for i in range(len(arr) - 1):
        if np.isnan(arr[i]):
            next_valid = next((x for x in arr[i + 1:] if not np.isnan(x)), np.nan)
            arr[i] = next_valid
    return arr

############################################################################################################################################################
# Part 2: Meteorological pre-processing

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

def alt2press(altitude):
    """
    Convert altitude to pressure.

    Parameters
    ----------
    altitude : Union[float, np.ndarray]
        Altitude in meters.

    Returns
    -------
    Union[int, np.ndarray]
        Pressure in Pascal.
    """
    L = -6.5*10**-3
    P0 = 101325
    T0 = 288.15
    R = 287.053
    g = 9.81

    if isinstance(altitude, (float, np.float32)):
        return P0*(1 + L*altitude/T0)**(-g/(R*L))
    else:
        pressures = np.zeros_like(altitude)
        for i in range(len(altitude)):
            pressures[i] = P0*(1 + L*altitude[i]/T0)**(-g/(R*L))

        return pressures

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

def calculate_evaporation_depth(regime_array, altitudes, RHw, T, pressures, latitude, longitude, met_type):
    """
    Calculate the evaporation depth of contrails based on atmospheric conditions.
    This function determines the evaporation depth of contrails by analyzing the 
    supersaturation and subsaturation regimes in the atmosphere. It uses meteorological 
    data to compute the volume of air and the amount of water molecules present, 
    adjusting the regime array to reflect the evaporation depth.
    Parameters:
        regime_array (numpy.ndarray): Binary array indicating supersaturation (1) 
            and subsaturation (0) regimes at different altitudes.
        altitudes (numpy.ndarray): Array of altitudes (in meters) corresponding to 
            the regime array.
        RHw (numpy.ndarray): Relative humidity with respect to water (unitless).
        T (numpy.ndarray): Temperature array (in Kelvin).
        pressures (numpy.ndarray): Atmospheric pressure array (in Pascals).
        latitude (float or numpy.ndarray): Latitude(s) of the location. For "ERA5", 
            this is a single float. For "GRUAN", this is an array of floats.
        longitude (float or numpy.ndarray): Longitude(s) of the location. For "ERA5", 
            this is a single float. For "GRUAN", this is an array of floats.
        met_type (str): Meteorological data type, either "ERA5" or "GRUAN". Determines 
            the format of latitude and longitude inputs.
    Returns:
        numpy.ndarray: Updated binary regime array indicating the evaporation depth 
        of contrails. Supersaturated regions are marked as 1, and subsaturated regions 
        are marked as 0.
    Raises:
        ValueError: If `latitude` or `longitude` types do not match the expected 
            format for the specified `met_type`.
        ValueError: If `met_type` is not "ERA5" or "GRUAN".
    Notes:
        - For "ERA5", latitude and longitude are single floats representing the 
          location of the GRUAN launch site.
        - For "GRUAN", latitude and longitude are arrays of floats describing the 
          radiosonde's path.
        - The function assumes that the input arrays (e.g., `regime_array`, `altitudes`, 
          `RHw`, `T`, `pressures`) are aligned and correspond to the same vertical 
          levels.
    Example:
        >>> regime_array = np.array([1, 0, 1, 0])
        >>> altitudes = np.array([1000, 2000, 3000, 4000])
        >>> RHw = np.array([0.8, 0.6, 0.9, 0.5])
        >>> T = np.array([273, 268, 263, 258])
        >>> pressures = np.array([90000, 80000, 70000, 60000])
        >>> latitude = 50.0
        >>> longitude = 10.0
        >>> met_type = "ERA5"
        >>> calculate_evaporation_depth(regime_array, altitudes, RHw, T, pressures, latitude, longitude, met_type)
        array([1, 1, 1, 0])
    """
    # P_sat: Saturation vapor pressure (Pa)
    # RH: Relative humidity wrt ice (unitless)
    # P_atm: Atmospheric pressure (Pa)
    
    # print("Initial Regime Binary:", regime_array)
    regime_array_ED = regime_array.copy()
    R = 6371*10**3 # Radius of the Earth (m)
    
    if met_type == "ERA5":
        # latitude and longitude must be floats describing the location of the GRUAN launch site
        if isinstance(latitude, float) != True:
            raise ValueError("Invalid type of", type(latitude) ,". Latitude and Longitude must be floats describing the location of the GRUAN launch site. Did you mean to pass in the GRUAN type parameter?")
        elif isinstance(longitude, float) != True:
            raise ValueError("Invalid type of", type(longitude) ,". Latitude and Longitude must be floats describing the location of the GRUAN launch site. Did you mean to pass in the GRUAN type parameter?")
        
        lat_len = 111320*0.25 # Latitude length (m). The size of a degree of latitude remains fairly constant across the Earth.
        lon_len = np.abs((2*np.pi*R*np.cos(latitude)/360)*0.25) # Longitude length (m), Haverside function
        heights = np.diff(altitudes) # Height of the gridcell (m)
        V = heights*lat_len*lon_len # volume of air in m^3
        
    elif met_type == "GRUAN":
        # latitude and longitude are altitude/pressure dependent arrays of floats describing the path of the radiosonde
        if isinstance(latitude, float) == True:
            raise ValueError("Invalid type of", type(latitude) ,". Latitude and Longitude must be arrays of floats describing the path of the GRUAN radiosonde. Did you mean to pass in the ERA5 type parameter?")
        elif isinstance(longitude, float) == True:
            raise ValueError("Invalid type of", type(longitude),". Latitude and Longitude must be arrays of floats describing the path of the GRUAN radiosonde. Did you mean to pass in the ERA5 type parameter?")
        
        lat_len = np.abs(111320*np.diff(latitude)) # Latitude length (m)
        lon_len = np.abs((2*np.pi*R*np.cos(averages_between_elements(latitude))/360)*(np.diff(longitude))) # Longitude length (m), Haverside function
        height = np.diff(altitudes) # Height of the gridcell (m)
        V = height*lat_len*lon_len # volume of air in m^3
    
    else:
        raise ValueError("Invalid type. Must be 'ERA5' or 'GRUAN'.")
    
    contrail_molecules_water = 0 # Initialize the amount of water molecules in the volume of air
    for i in range (len(regime_array)-1):
        
        # If regime_array[i] == 1, the altitude is supersaturated
        if regime_array[i] == 1:
            # print("Initially Supersaturated: regime_array is 1")

            # Calculate water picked up in supersaturated zone
            P_sat = compute_Psat_w(T[i]) # [Pa]
            P_atm = pressures[i] # [Pa]
            ppmv = (P_sat/P_atm)*(RHw[i])*10**6 # ppmv
            contrail_molecules_water = contrail_molecules_water + ppmv*V[i] # molecules of water in the volume of air

        else:
            # print("Initially Subsaturated: regime_array is 0")

            # Calculate water deposited in subsaturated zone
            P_sat = compute_Psat_w(T[i]) # [Pa]
            P_atm = pressures[i] # [Pa]

            # contrail_molecules_water_initial = contrail_molecules_water

            sat_molecules_water = (P_sat/P_atm)*10**6*V[i] # For RH = 1, molecules of water in the volume of air required for saturation
            background_molecules_water = (P_sat/P_atm)*(RHw[i])*10**6*V[i] # For RH < 1, molecules of water actually in the volume of air
            contrail_molecules_water = contrail_molecules_water - (sat_molecules_water - background_molecules_water) # Amount of water molecules deposited in the volume of air by the contrail
            
            # print("Available water:", f"{contrail_molecules_water_initial :.2e}", 
            #       "Saturation Req:", f"{sat_molecules_water - background_molecules_water:.2e}", 
            #       "Remaining ppm:", f"{contrail_molecules_water:.2e}")

        # print("contrail_molecules_water:", f"{contrail_molecules_water:.2e}")

        # When contrail_molecules_water = 0, the contrail has evaporated
        if contrail_molecules_water > 0:
            regime_array_ED[i] = 1 # Mark subsaturated binary as supersaturated (evaporation depth)
        else:
            contrail_molecules_water = 0 # Reset the amount of water molecules in the volume of air
            # print("Contrail Death, ppm reset: ", contrail_molecules_water, " ppm")
        
    #     print("Molecules of contrail water at altitude ", altitudes[i], "are:", f"{contrail_molecules_water:.2e}\n")
    # print("Evaporation Depth Regime Binary:", regime_array_ED)
    return regime_array_ED

def calculate_MLD(altitudes, pressures, humidities, temperatures, latitude, longitude, met_type):
    """
    Classifies atmospheric altitudes into regimes based on relative humidity (RH) 
    and calculates evaporation depth for subsaturated altitudes below the first 
    supersaturated altitude.
    Parameters:
        altitudes (list or numpy.ndarray): Array of altitudes (in meters).
        pressures (list or numpy.ndarray): Array of atmospheric pressures (in hPa or Pa).
        humidities (list or numpy.ndarray): Array of relative humidities (RH, unitless).
        temperatures (list or numpy.ndarray): Array of temperatures (in Kelvin).
        latitude (float): Latitude of the location (in degrees).
        longitude (float): Longitude of the location (in degrees).
        met_type (str): Type of meteorological data (e.g., "ERA5", "GFS").
    Returns:
        tuple:
            - regime_array (list): Binary array indicating the regime classification 
              for each altitude (0 for subsaturated, 1 for supersaturated).
            - regime_array_ED (list): Modified regime array after calculating 
              evaporation depth for subsaturated altitudes below the first 
              supersaturated altitude.
    Notes:
        - Supersaturation is defined as RH >= 1.
        - If no supersaturated altitudes are found, the evaporation depth calculation 
          is skipped, and the original regime array is returned as `regime_array_ED`.
        - The function assumes that the input arrays are aligned and of the same length.
    """

    # Convert arrays to read as floats
    humidities = np.array(humidities, dtype=float)
    altitudes = np.array(altitudes, dtype=float)
    
    # Initialize the binary array to store the regime classification
    regime_array = [0] * len(altitudes)  # Start by assuming all are subsaturated (0)

    # Step 1: Find the first altitude with RH >= 1 (supersaturated)
    found_supersaturated = False
    first_supersaturated_index = -1
    
    for i in range(len(humidities)):
        if humidities[i] >= 1:
            found_supersaturated = True
            first_supersaturated_index = i
            regime_array[i] = 1  # Mark this altitude as supersaturated
            break  # Stop once we find the first supersaturated altitude

    if not found_supersaturated:
        print("No supersaturated altitudes found.")
        # If no supersaturated altitudes, return the original regime array
        regime_array_ED = regime_array.copy()
    else:
        # Step 2: Now classify the rest of the altitudes
        for i in range(first_supersaturated_index + 1, len(humidities)):
            if humidities[i] >= 1:
                regime_array[i] = 1  # Mark as supersaturated (since RH >= 1)

        # Step 3: For altitudes below the first supersaturated, calculate evaporation depth
        regime_array_ED = calculate_evaporation_depth(regime_array, altitudes, humidities, temperatures, pressures, latitude, longitude, met_type)

    return regime_array, regime_array_ED

def averages_between_elements(arr):
    return [(arr[i] + arr[i + 1]) / 2 for i in range(len(arr) - 1)]

def linear_diffusion(arr):
    # Find indices where the value changes
    change_points = np.where(np.diff(arr) != 0)[0] + 1
    result = arr.astype(float).copy()

    # Interpolate between change points
    start = 0
    for end in change_points:
        # Interpolate only between start and end (exclusive of endpoints)
        if end - start > 1:
            result[start:end] = np.linspace(arr[start], arr[end-1], end - start)
        start = end
    # Handle the last segment after the final change point
    if start < len(arr) - 1:
        result[start:] = np.linspace(arr[start], arr[-1], len(arr) - start)

    return result

def construct_complementary_paths(gruan_base_dir, era5_base_dir):
    # Arrays to store the paths
    gruan_file_paths = []
    era5_file_paths = []
    
    # Recursively find all GRUAN files and construct corresponding ERA5 file paths
    for root, dirs, files in os.walk(gruan_base_dir):
        for file in files:
            if file.endswith('.nc'):
                gruan_file_path = os.path.join(root, file)
                era5_file_path = construct_era5_path(era5_base_dir, gruan_file_path)
                if os.path.exists(era5_file_path):  # Check if the ERA5 file path exists
                    gruan_file_paths.append(gruan_file_path)
                    era5_file_paths.append(era5_file_path)

    # Sort the paths
    gruan_file_paths.sort()
    era5_file_paths.sort()
    return gruan_file_paths, era5_file_paths

############################################################################################################################################################
# Part 3a: Generate APCEMM input files

class APCEMMConfig:
    def __init__(self):
        # Specify the test
        self.aircraft_engine = "B737-800_CFM56"  # Which aircraft and engine are we using?
        self.test_id = "test_10"  # Test number the met and YAML files are associated with
        self.training_runs = 20  # Number of APCEMM runs to process for the test
        self.validation_runs = 20  # Number of APCEMM runs to process for the test
        self.polynomial_degree = 1  # Polynomial degree for the PCE
        self.maximum_degree = 2  # Maximum degree for the PCE

        # Define the meteorological APCEMM file dimensions
        self.APCEMM_altitudes = 125  # Number of altitudes recorded in meteorological file
        self.APCEMM_timesteps = 24  # Hours recorded in meteorological file

        # RHi initial condition distribution details
        self.IC_std_rhi = 10
        self.IC_mean_amplitude_rhi = 117
        # RHi temporal distribution details
        self.time_std_rhi = 0
        self.time_mean_amplitude_rhi = 0

        # MLD initial condition distribution details
        self.IC_std_mld = 20
        self.IC_mean_amplitude_mld = 100
        # MLD temporal distribution details
        self.time_std_mld = 0
        self.time_mean_amplitude_mld = 0

def combine_parquet_files(input_dir, output_file):
    """
    Combine all Parquet files in the input directory into a single Parquet file.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the Parquet files.
    output_file : str
        Path to save the combined Parquet file.
    """
    # Get a list of all Parquet files in the input directory
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]
    
    # Read and concatenate all Parquet files into a single DataFrame
    combined_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    
    # Save the combined DataFrame to a single Parquet file
    combined_df.to_parquet(output_file, index=False)

    return combined_df

def generate_apcemm_input_files(base_met_dir, num_met_files, set_type, test_specifications):
    # Set up output path
    if set_type == "validation":
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/inputs/validation/APCEMM_met_validation_{}.nc'
    else:
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/inputs/training/APCEMM_met_{}.nc'
    output_file_template = out_path

    for run_num in range(1,num_met_files+1):

        # Sample initial condition, throwing out entries under threshold
        low_RHi_IC = True
        while low_RHi_IC == True:
            # Sample stochastic RHi initial condition
            RHi_IC = np.round(np.random.normal(0, test_specifications.IC_std_rhi, size=1) + test_specifications.IC_mean_amplitude_rhi, 2)
            if RHi_IC[0] >= 117:
                low_RHi_IC = False

        # Sample RHi values for the rest of the timesteps, checking for negative values
        flag_negative = True
        while flag_negative == True:
            # Sample fluctuations in RH values, centered around IC mean from ERA5 ensembles
            RHi_time_samples = np.round(np.random.normal(0, test_specifications.time_std_rhi, size=test_specifications.APCEMM_timesteps-1) + test_specifications.time_mean_amplitude_rhi, 2)

            # Check if RH_sampled has any zero or negative values
            if np.any(RHi_time_samples <= 0):
                flag_negative = True
                print("RHi_sampled contains zero or negative values.")
            else:
                flag_negative = False

        # Consolidate the sampled RHi values into a matrix
        RHi_time = np.append(RHi_IC, RHi_time_samples)

        RHi_sampled_matrix = np.tile(RHi_time, (16, 1)) # Replacing only 16 altitude layers with samples. The rest are predefined as subsaturated. #MUST BE UPDATED WITH MLD ADDITION

        # Open the input NetCDF file containing the base metoeorological data
        ds = xr.open_dataset(base_met_dir)

        # Replace the 250 hPa row of ds with the sampled RH timeseries
        ds['relative_humidity_ice'][90:106, :] = RHi_sampled_matrix

        # Save the changes to new output files
        output_file_path = output_file_template.format(test_specifications.test_id, run_num)
        ds.to_netcdf(output_file_path)
        ds.close()
        # Generate associated YAML file
        generate_yaml_file(test_specifications.test_id, test_specifications.aircraft_engine, set_type, run_num)

# Now we can update the YAML base file to call the new met files
def generate_yaml_file(test_id, aircraft_engine, set_type, run_num):
    test_num = test_id.split("_")[1]  # Extract the test number from the test ID
    # Define source and destination paths
    source_path = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/YAML_files/{}_APCEMM_input.yaml".format(aircraft_engine)  # Source YAML file
    destination_dir = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/test_{}/inputs/{}".format(test_num, set_type)  # Target directory
    destination_path = os.path.join(destination_dir, "APCEMM_input_run_{}.yaml".format(run_num))

    # Copy the YAML file to the new directory
    shutil.copy(source_path, destination_path)

    # Read and modify the copied YAML file
    with open(destination_path, "r") as file:
        data = yaml.safe_load(file)  # Load YAML into a Python dictionary

    # Modify the YAML content
    data["SIMULATION MENU"]["OUTPUT SUBMENU"]["Output folder (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/test_{}/outputs/{}/test_{}_run_{}".format(test_num, set_type, test_num, run_num)

    if set_type == "training":
        data["METEOROLOGY MENU"]["METEOROLOGICAL INPUT SUBMENU"]["Met input file path (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/test_{}/inputs/{}/APCEMM_met_{}.nc".format(test_num, set_type, run_num)
    else:
        data["METEOROLOGY MENU"]["METEOROLOGICAL INPUT SUBMENU"]["Met input file path (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/test_{}/inputs/{}/APCEMM_met_validation_{}.nc".format(test_num, set_type, run_num)

    # Write the modified YAML back to the file
    with open(destination_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, indent=4)

############################################################################################################################################################
# Part 3b:  Save APCEMM input and output variables to a PCE readable format

class apcemm_data_struct:
    def __init__(self, t, ds_t, int_OD, RHi):
        self.t = t
        self.ds_t = ds_t
        self.int_OD = int_OD
        self.RHi = RHi
    
def read_apcemm_data(directory):
    t_mins = []
    ds_t = []
    int_OD = []
    RHi = []

    for file in sorted(os.listdir(directory)):
        if(file.startswith('ts_aerosol') and file.endswith('.nc')):
            file_path = os.path.join(directory,file)
            ds = xr.open_dataset(file_path, engine = "netcdf4", decode_times = False)
            ds_t.append(ds)
            tokens = file_path.split('.')
            mins = int(tokens[-2][-2:])
            hrs = int(tokens[-2][-4:-2])
            t_mins.append(hrs*60 + mins)
            int_OD.append(ds["intOD"])
            RHi.append(ds["RHi"])

    return apcemm_data_struct(t_mins, ds_t, int_OD, RHi)

def create_sample_matrix(test_specifications, set_type): # NEED TO UPDATE WITH MLD AND TIME DEPENDENT SCALING
    # Define constants
    test_id = test_specifications.test_id
    num_runs = test_specifications.training_runs if set_type == "training" else test_specifications.validation_runs
    IC_mean_amplitude_rhi = test_specifications.IC_mean_amplitude_rhi
    sample_arrays = []

    for i in range(1, num_runs+1): # For test runs num_runs
        if set_type == "validation": # If you are processing a validation set
            apcemm_data = read_apcemm_data(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/outputs/validation/{test_id}_run_{i}')
            input_RHi_ds = xr.open_dataset(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/inputs/validation/APCEMM_met_validation_{i}.nc')
        else: # If you are processing a training set
            apcemm_data = read_apcemm_data(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/outputs/training/{test_id}_run_{i}')
            input_RHi_ds = xr.open_dataset(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/inputs/training/APCEMM_met_{i}.nc')

        int_OD = apcemm_data.int_OD
        recorded_timesteps = len(int_OD)

        # Normalizing outputs to a 12 hour timeframe
        if recorded_timesteps == 0: # Ignore any runs where a contrail does not form
            continue
        elif recorded_timesteps > 73: # Truncate to 73 timesteps if contrail persists longer than 12 hours
            current_sample_output = np.array(int_OD)[:73].reshape(1, 73)[0] # 1 row, 73 columns
        elif recorded_timesteps < 73: # Pad with zeros to reach 73 timesteps
            current_sample_output = np.pad(np.array(int_OD).reshape(1, recorded_timesteps)[0], (0, 73 - recorded_timesteps), 'constant', constant_values = 0)

        # Define your input_RHi array (length 24 --> 24 hours in original met file)
        input_RHi = input_RHi_ds['relative_humidity_ice'][98].values # RHi values for times 0-24 hours at pressure level index 98. 

        # Expand input_RHi to match the timestamps output by APCEMM. APCEMM is run for 12 hours at 10-minute intervals. The met file is for 24 hours at 1-hour intervals.
        # 1 hour is 6 10-minute intervals, so we repeat each RHi value 6 times to match met input to the APCEMM output.
        expanded_input_RHi = []
        expanded_input_RHi.append(input_RHi[0]) # Count the zeroth timestep as a 7th repeat
        repeated_input_RHi = np.repeat(input_RHi[0:12], 6) # Look only at the first 12 hours of the met file. Repeat each element 6 times: [100 110 105] --> [100 100 100 100 100 100 110 110 110 110 110 110 105 105 105 105 105 105]
        expanded_input_RHi.extend(repeated_input_RHi)
        normalized_input_RHi = (np.array(expanded_input_RHi) - IC_mean_amplitude_rhi) # Shift the mean back to 0 for PCE to parse
        current_sample_input = normalized_input_RHi.reshape(1,73)[0] # Transform into a row vector for stacking

        # Stacking the input and output arrays
        stacked = np.column_stack((current_sample_input, current_sample_output)) # Shape: (73, 2) --> (column, depth) --> (timesteps, 1 input 1 output) INPUTS: (:,0), OUTPUTS: (:,1)
        sample_arrays.append(stacked)
        input_RHi_ds.close()

    # This matrix will be used for training the machine learning model, it contains input RHi and output int_OD. In The future there will be multiple input variables.
    sample_matrix = np.array(sample_arrays) # Shape: (num_runs, 73, 2) --> (depth, row, column) --> (number of datasets to train PCE, timesteps, 1 input 1 output) INPUTS: (:,:,0), OUTPUTS: (:,:,1)
    # Transpose the matrix to match the expected input shape of the machine learning model
    sample_matrix = np.transpose(sample_matrix, (2, 0, 1)) # Shape: (2, num_runs, 73) --> (depth, row, column) --> (1 input 1 output, number of datasets to train PCE, timesteps) INPUTS: (0,:,:), OUTPUTS: (1,:,:)

    # Save the training_sample_matrix to a .npy file
    np.save('/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/outputs/{}/{}_sample_matrix.npy'.format(test_id, set_type, set_type), sample_matrix)

    print(f"{set_type} sample matrix created and saved successfully.")

#############################################################################################################################################################
# Part 4/5/6: Create and validate the PCE model
from numpy.matlib import repmat

def totalOrderMultiIndices(m, p):
    """
    Returns a matrix containing all multi-indices of size m of total order p
    (implementation due to prior course member)

    Parameters
    ----------
    m : int 
        Number of uncertain variables
    p : int
        Maximum total degree (sum of degrees across a multiindex)

    Returns
    -------
    np.array
        Matrix containing multi-indices on rows.

    """
    Mk = np.zeros((1, m))
    M = Mk
    for k in range(p):
        Mk = repmat(Mk, m, 1) + np.kron(np.eye(m), np.ones((Mk.shape[0], 1)))
        Mk = np.unique(Mk, axis=0)
        M = np.vstack((M, Mk))
    return M.astype(int)

"""
psi = hermitePoly(x, degree)
Evaluates hermite polynomials up to degree `degree` on all points x

Arguments:
    x: points to evaluate at
    degree: highest degree to evaluate to

Returns
    psi: (degree+1,length(x)) Evaluate degree+1 hermite polynomials on x
"""
def hermitePoly(x, degree):
    c = np.identity(degree+1)
    psi = np.polynomial.hermite_e.hermeval(x, c)
    return psi


### Create and Validate PCE ###
def create_PCE(test_specifications, sample_type:str):
    training_runs = test_specifications.training_runs if sample_type == "training" else test_specifications.validation_runs

    print(f"Creating PCE for {sample_type} samples with {training_runs} training runs.")
    c, alpha_set = compute_c(sample_type, test_specifications)
    return c, alpha_set

def predicted_novel_solutions(c, alpha_set, novel_runs): # Novel inputs to PCE from online samples
    sample_type = "novel"
    multiindices = alpha_set.shape[0]
    He_array = np.zeros((multiindices, novel_runs))
    samples_matrix = get_samples_matrix_online(novel_runs, sample_type)

    for i in tqdm.tqdm(range(novel_runs)): # for each MC run
        for j in range(multiindices): # for each coefficient
            current_alpha = alpha_set[j, :] # look at one row at a time for all alpha describing a single coefficient
            samples_matrix_He = samples_matrix[i, :]
            He_array[j, i] = compute_He(samples_matrix_He, current_alpha)

    # Solve for the Least Squares solution
    predicted_output = c.T @ He_array
    return predicted_output.T

def true_training_solutions(sample_type:str, test_id):
    true_output = func_eval_offline(sample_type, test_id).T # TRUE APCEMM SOLUTIONS [Integrated VOD] with shape: (number of datasets to train PCE, timesteps)
    return true_output

### NEED TO UPDATE OFFLINE FUNCTIONS TO DISTINGUISH VALIDATION AND TRAINING PATHS
def predicted_validation_solutions(c, alpha_set, test_specifications): # Validation inputs to PCE from offline pre-computed validation set
    validation_runs = test_specifications.validation_runs
    test_id = test_specifications.test_id
    sample_type = "validation"
    multiindices = alpha_set.shape[0]
    He_array = np.zeros((multiindices, validation_runs))
    samples_matrix = get_samples_matrix_offline(sample_type, test_id)

    for i in tqdm.tqdm(range(validation_runs)): # for each MC run
        
        for j in range(multiindices): # for each coefficient
            current_alpha = alpha_set[j, :] # look at one row at a time for all alpha describing a single coefficient
            samples_matrix_He = samples_matrix[i, :]
            He_array[j, i] = compute_He(samples_matrix_He, current_alpha)

    # Solve for the Least Squares solution
    predicted_output = (c.T @ He_array).T

    return predicted_output

def true_validation_solutions(sample_type:str, test_id):
    true_output = func_eval_offline(sample_type, test_id).T # TRUE APCEMM SOLUTIONS [Integrated VOD] with shape: (number of datasets to train PCE, timesteps)
    return true_output

### Internal Functions ###
def compute_c(sample_type:str, test_specifications):
    test_id = test_specifications.test_id
    training_runs = test_specifications.training_runs if sample_type == "training" else test_specifications.validation_runs

    c_samples_matrix = get_samples_matrix_offline(sample_type, test_id) # (timesteps, number of datasets to train PCE)
    alpha_set = get_alpha_set(test_specifications.polynomial_degree, test_specifications.maximum_degree) # (number of datasets to train PCE, timesteps)
    c_multiindices = alpha_set.shape[0]

    V = np.zeros((training_runs, c_multiindices)) # Vandermonde Matrix: Training Runs x Degree of Polynomial

    for i in tqdm.tqdm(range(training_runs)):
        for j in range(c_multiindices):
            current_alpha = alpha_set[j, :]
            c_samples_matrix_He = c_samples_matrix[:, i]
            V[i, j] = compute_He(c_samples_matrix_He, current_alpha) / np.sqrt(np.product(sc.special.factorial(current_alpha)))

    f = solve_u(sample_type, test_id).T # (number of datasets to train PCE, timesteps)
    
    c, residuals, rank, singular_values = np.linalg.lstsq(V.T @ V, V.T @ f)
    
    return c, alpha_set

def func_eval_offline(sample_type:str, test_id):

    if sample_type == "training":
        training_samples_matrix_offline =np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/PCE_results/APCEMM_PCE_results/{}/training_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM training data

        training_samples_matrix_offline_OUTPUTS = training_samples_matrix_offline[1,:,:] # (depth, row, column) --> (0=input 1=output, number of datasets to train PCE, timesteps)
        # Transpose the rows and columns to accomodate later calculations
        training_samples_matrix_offline_OUTPUTS = training_samples_matrix_offline_OUTPUTS.T # (timesteps, number of datasets to train PCE)
        return training_samples_matrix_offline_OUTPUTS
    
    elif sample_type == "validation":
        validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/PCE_results/APCEMM_PCE_results/{}/validation_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM validation data

        validation_samples_matrix_offline_OUTPUTS = validation_samples_matrix_offline[1,:,:]
        validation_samples_matrix_offline_OUTPUTS = validation_samples_matrix_offline_OUTPUTS.T
        return validation_samples_matrix_offline_OUTPUTS
    
    else: 
        print("Invalid sample type. Please enter 'training' or 'validation'.")
        return None


def get_samples_matrix_offline(sample_type:str, test_id):

    if sample_type == "training":
        training_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/PCE_results/APCEMM_PCE_results/{}/training_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM training data

        training_samples_matrix_offline_INPUTS = training_samples_matrix_offline[0,:,:] # (depth, row, column) --> (0=input 1=output, number of datasets to train PCE, timesteps)
        training_samples_matrix_offline_INPUTS = training_samples_matrix_offline_INPUTS.T # (timesteps, number of datasets to train PCE)
        print(training_samples_matrix_offline_INPUTS.shape)
        return training_samples_matrix_offline_INPUTS
    
    elif sample_type == "validation":
        validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/PCE_results/APCEMM_PCE_results/{}/validation_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM validation data

        validation_samples_matrix_offline_INPUTS = validation_samples_matrix_offline[0,:,:] # (depth, row, column) --> (0=input 1=output, number of datasets, timesteps)
        validation_samples_matrix_offline_INPUTS = validation_samples_matrix_offline_INPUTS.T # (timesteps, number of datasets)
        return validation_samples_matrix_offline_INPUTS

    else: 
        print("Invalid sample type. Please enter 'training' or 'validation'.")
        return None
    
def get_samples_matrix_online(runs):
    return np.hstack((np.random.normal(mean_IC_norm, std_IC, size=(runs, 1)), np.random.normal(mean_norm, std_norm, size=(runs, timesteps - 1))))

def get_alpha_set(poly_dim, max_deg):
    return totalOrderMultiIndices(poly_dim, max_deg)

def compute_He(Z, alpha):
    res = 1
    for alpha_i, z_i in zip(alpha, Z):
        res = res * HermiteE.basis(deg = alpha_i)(z_i)
    return res

def solve_u(sample_type:str, test_id):
    u_sol = func_eval_offline(sample_type, test_id) # (timesteps, number of datasets to train PCE)
    return u_sol

def smooth_artifacts(offline_samples_matrix):

    y = offline_samples_matrix # Extract the output for the i-th training run
    # Step 1: Find large jumps (spikes)
    diff_y = np.abs(np.diff(y))
    threshold = 500  # You can adjust this
    spike_indices = np.where(diff_y > threshold)[0]
    # Step 2: Mask the spike region
    mask = np.ones_like(y, dtype=bool)
    for idx in spike_indices:
        mask[max(0, idx-1):min(len(y), idx+5)] = False  # mask +- some points
    # Step 3: Interpolate to fill spike
    x = np.arange(len(y))
    y_cleaned = np.copy(y)
    y_cleaned[~mask] = np.interp(x[~mask], x[mask], y[mask])
    
    return y_cleaned

#############################################################################################################################################################
# Part 7: Compute sensitivity indices

def compute_total_effect_sensitivity_indices(coefficients, num_variables):
    """
    Compute total effect sensitivity indices for each uncertain variable.

    Parameters:
    coefficients (numpy.ndarray): Simulated coefficients, rows represent different basis terms, columns represent timesteps.
    num_variables (int): Number of uncertain variables.

    Returns:
    dict: Dictionary containing total effect sensitivity indices for each variable.
    """
    # Number of basis terms
    num_basis = coefficients.shape[0]

    # Compute total variance for each timestep (sum of squared coefficients excluding the constant term)
    total_variance = np.sum(coefficients[1:, :] ** 2, axis=0)  # Sum across basis terms, excluding the first row

    # Dictionary to store total effect sensitivity indices
    S_T = {}

    # Assuming that each variable's basis functions are in consecutive blocks, 
    # we will group basis terms by the uncertain variables they correspond to.
    for var_idx in range(num_variables):
        # For each uncertain variable, sum over the basis terms that correspond to that variable.
        # Assuming here that each variable's terms are grouped consecutively in the matrix:
        start_idx = var_idx * (num_basis // num_variables)  # Starting index for this variable's terms
        end_idx = start_idx + (num_basis // num_variables)  # Ending index for this variable's terms

        # Sum over all terms that involve this variable (main effect + interactions)
        variance_contribution = np.sum(coefficients[start_idx:end_idx, :] ** 2, axis=0)  # Sum over terms for this variable
        S_T[f"S_T_{var_idx}"] = variance_contribution / total_variance

    return S_T

