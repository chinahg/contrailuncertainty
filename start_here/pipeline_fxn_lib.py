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
    # Recursively find all GRUAN files and construct corresponding ERA5 file paths
    for root, dirs, files in os.walk(gruan_base_dir):
        for file in files:
            if file.endswith('.nc'):
                gruan_file_path = os.path.join(root, file)
                era5_file_path = fxn.construct_era5_path(era5_base_dir, gruan_file_path)
                if os.path.exists(era5_file_path):  # Check if the ERA5 file path exists
                    gruan_file_paths.append(gruan_file_path)
                    era5_file_paths.append(era5_file_path)

    # Sort the paths
    gruan_file_paths.sort()
    era5_file_paths.sort()
    return gruan_file_paths, era5_file_paths

############################################################################################################################################################
# Part 3: Generate APCEMM input files

def generate_apcemm_input_files(base_met_dir, num_met_files, test_num, set_type):
    # Set up output path
    if set_type == "validation":
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/inputs/{}/APCEMM_met_validation_{}.nc'
    else:
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/inputs/{}/APCEMM_met_{}.nc'
    output_file_template = out_path

    for run_num in range(1,num_met_files+1):
        # Sample initial condition, throwing out entries under threshold
        low_RHi_IC = True
        while low_RHi_IC == True:
            # Sample stochastic RHi initial condition
            RHi_IC = np.round(np.random.normal(mean_norm, std_norm, size=1) + IC_scaled_mean, 2)
            if RHi_IC[0] >= 117:
                low_RHi_IC = False

        # Sample RHi values for the rest of the timesteps, checking for negative values
        flag_negative = True
        while flag_negative == True:
            # Sample fluctuations in RH values, centered around IC mean from ERA5 ensembles
            RHi_time_samples = np.round(np.random.normal(mean_norm_time, std_norm_time, size=timesteps-1) + RHi_IC[0], 2)

            # Check if RH_sampled has any zero or negative values
            if np.any(RHi_time_samples <= 0):
                flag_negative = True
                print("RHi_sampled contains zero or negative values.")
            else:
                flag_negative = False

        # Consolidate the sampled RHi values into a matrix
        RHi_time = np.append(RHi_IC, RHi_time_samples)

        RHi_sampled_matrix = np.tile(RHi_time, (16, 1)) # Replacing only 16 altitude layers with samples. The rest are predefined as subsaturated.

        # Open the input NetCDF file containing the base metoeorological data
        ds = xr.open_dataset(base_met_dir)

        # Replace the 250 hPa row of ds with the sampled RH timeseries
        ds['relative_humidity_ice'][90:106, :] = RHi_sampled_matrix

        # Save the changes to new output files
        output_file_path = output_file_template.format(test_num, set_type, run_num)
        ds.to_netcdf(output_file_path)
        ds.close()
        # Generate associated YAML file
        generate_yaml_file(test_num, set_type, run_num)

# Now we can update the YAML base file to call the new met files
def generate_yaml_file(test_num, set_type, run_num):
    # Define source and destination paths
    source_path = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/BASE_APCEMM_input.yaml"  # Source YAML file
    destination_dir = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/inputs/{}".format(test_num, set_type)  # Target directory
    destination_path = os.path.join(destination_dir, "APCEMM_input_run_{}.yaml".format(run_num))

    # Copy the YAML file to the new directory
    shutil.copy(source_path, destination_path)

    # Read and modify the copied YAML file
    with open(destination_path, "r") as file:
        data = yaml.safe_load(file)  # Load YAML into a Python dictionary

    # Modify the YAML content
    data["SIMULATION MENU"]["OUTPUT SUBMENU"]["Output folder (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/outputs/{}/test_{}_run_{}".format(test_num, set_type, test_num, run_num)

    if set_type == "training":
        data["METEOROLOGY MENU"]["METEOROLOGICAL INPUT SUBMENU"]["Met input file path (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/inputs/{}/APCEMM_met_{}.nc".format(test_num, set_type, run_num)
    else:
        data["METEOROLOGY MENU"]["METEOROLOGICAL INPUT SUBMENU"]["Met input file path (string)"] = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{}/inputs/{}/APCEMM_met_validation_{}.nc".format(test_num, set_type, run_num)

    # Write the modified YAML back to the file
    with open(destination_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, indent=4)