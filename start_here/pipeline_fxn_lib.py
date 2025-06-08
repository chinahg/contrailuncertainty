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
import subprocess
import time
from dataclasses import dataclass
from scipy.stats import gaussian_kde
import json
from numpy.matlib import repmat

##############################################################################################################################################
# Function library for pipeline
################################################################################################################################################
# General Functions: Not Step Specific

def submit_job_and_get_id(script_path, job_type, export_args):
    # Submit the job using sbatch and capture the output
    if job_type == "no_args":
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True
        )
    elif job_type == "has_args":
        result = subprocess.run(
        ["sbatch", "--export=" + export_args, script_path],
        capture_output=True,
        text=True
    )

    # Extract the job ID from the output
    output = result.stdout.strip()
    job_id = output.split()[-1]  # Get the last word from the output
    print(f"Submitted job with ID: {job_id}")
    return job_id

def wait_for_specific_jobs(job_ids):
    job_ids_str = ",".join(job_ids)
    while True:
        result = subprocess.run(
            ["squeue", "-j", job_ids_str],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error from squeue: {result.stderr}")
            break
        # Check if only header line is present
        if len(result.stdout.strip().splitlines()) <= 1:
            print("All specified jobs are completed!")
            break
        else:
            print("Waiting for jobs to complete...")
            time.sleep(60) #  # Wait for 1 minute before checking again

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
    

################################################################################################################################################
# Part 1: Downloading and formatting GRUAN and ERA5 data

def download_files(entire_file, files2download_path, files_per_batch, files2download):
    ## Download ERA5 data correpsonding to the available GRUAN data
    # Download the missing ERA5 files using slurm
    job_ids = []  # Initialize an array to store job IDs
    download_details_path = "/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/download_details.yaml"

    if entire_file == True:
        # Serialise and save the download job details to a yaml file
        download_details_yaml = download_details(files2download_path, 0, len(files2download))
        with open(download_details_path, "w") as f:
            yaml.safe_dump(download_details_yaml.__dict__, f)

        # Pass the download_details filepath to the bash script
        export_args = f"ARG1={download_details_path}"

        # Submit the job and get the job ID
        job_ids.append(submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/era5_download.sh", "has_args", export_args))
    else:
        groups = len(files2download) // files_per_batch + (1 if len(files2download) % files_per_batch != 0 else 0) # Number of groups to split the files into for slurm
        for i in range(groups):
            start_index = i * files_per_batch
            end_index = start_index + files_per_batch
            print(f"Submitting batch {i+1} with files from {start_index} to {end_index}")

            if end_index > len(files2download):
                end_index = len(files2download)

            # Serialise and save the download job details to a yaml file
            download_details_yaml = download_details(files2download_path, start_index, end_index)
            with open(download_details_path, "w") as f:
                yaml.safe_dump(download_details_yaml.__dict__, f)

            # Pass the download_details filepath to the bash script
            export_args = f"ARG1={download_details_path}"

            # Submit the job and get the job ID
            job_ids.append(submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/era5_download.sh", "has_args", export_args))
            time.sleep(15)  # Optional: wait a bit before submitting the next job

    # Wait for the job to complete
    wait_for_specific_jobs(job_ids)

    complete_message = "Job(s) completed, continuing with the script!"
    ### WAIT FOR THE JOB TO FINISH THEN CONTINUE
    return complete_message

@dataclass
class download_details:
    files2download_path: str
    start_index: int
    end_index: int

def check_files(GRUAN_base_dir, years, generated_files_dir):
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

    file2download_path = f'{generated_files_dir}/files2download.csv'

    # Open the file in write mode
    with open(file2download_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the array to the CSV file
        writer.writerow(files2download)

    # Save grib filenames to CSV file
    files2convert = list(set(files2convert))
    files2convert.sort()

    files2convert_path = f'{generated_files_dir}/files2convert.csv'

    # Open the file in write mode
    with open(files2convert_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the array to the CSV file
        writer.writerow(files2convert)
    
    return files2convert, files2download, GRUAN_date_sites

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

    ERA5_directory = '/home/chinahg/GCresearch/ERA5_downloads'
    
    ERA5_file_names = []
    for root, dirs, files in os.walk(ERA5_directory):
        for file in files:
            ERA5_file_names.append(os.path.basename(os.path.join(root, file)))

    for i in range(len(GRUAN_date_sites)):
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
    ERA5_file_names_matching = list(ERA5_name_only)
    ERA5_file_names_matching.sort()

    files2convert = list(set(files2convert))
    files2download = list(set(files2download))

    # print("Number of unconverted ERA5 GRIB files: ", len(files2convert))
    # print("Number of ERA5 files to download: ", len(files2download))
    # print("Number of GRUAN files: ", len(GRUAN_date_sites))
    # print("Number of matching meteorological files: ", len(ERA5_file_names_matching))
    
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
    # Convert final array to a numpy array
    arr = np.array(arr)
    return arr

############################################################################################################################################################
# Step 2: Calculate and add RHi and RHw keys to ERA5 nc file

def add_RHi_RHw(matching_files):
    """
    Calculate RHi and RHw for all the ERA5 files and add as a new variable.
    Also check if pressure and time keys are standardized to new 2024 format.
    Parameters
    ----------
    matching_files : list
        List of matching ERA5 file names.
    """ 

    # Make a file with all ERA5 files, as we need to check the contents and formatting
    directory = '/home/chinahg/GCresearch/ERA5_downloads'
    csv_path = '/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/era5_files.csv'

    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.nc'):
                files.append(f)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file in files:
            writer.writerow([file])

    # Slurm setup
    files_per_batch = 500
    groups = len(files) // files_per_batch + (len(files) % files_per_batch > 0)
    job_ids = []

    for i in range(groups):
        start_index = i * files_per_batch
        end_index = start_index + files_per_batch
        print(f"Submitting batch {i+1} with files from {start_index} to {end_index}")

        if end_index > len(files):
            end_index = len(files)

        # Run slurm jobs to update the files with RHi and RHw
        export_args = f"ARG1={start_index},ARG2={end_index}"

        # Submit the job and get the job ID
        job_ids.append(submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/Met_processing/add_rh_era5.sh", "has_args", export_args))
        time.sleep(10)  # Optional: wait a bit before submitting the next job
    
    # Wait for the job to complete
    wait_for_specific_jobs(job_ids)

    complete_message = "Job(s) completed, continuing with the script!"
    
    return complete_message

############################################################################################################################################################
# Step 3: Calculate MLD and consolidate ERA5 data and GRUAN data into a single file

def process_met_data(GRUAN_base_dir, ERA5_base_dir, generated_files_dir):
    gruan_file_paths, era5_file_paths = construct_complementary_paths(GRUAN_base_dir, ERA5_base_dir)

    era5_file_paths = [path for path in era5_file_paths]
    gruan_file_paths = [path for path in gruan_file_paths]

    print(f"First GRUAN file path: {gruan_file_paths[0]}")
    print(f"First ERA5 file path: {era5_file_paths[0]}")

    # Define the start and end file indices for processing, as ideally we will split the files into batches for slurm
    files_per_batch = 100  # Number of files to process per slurm batch
    batches = len(gruan_file_paths) // files_per_batch + (1 if len(gruan_file_paths) % files_per_batch != 0 else 0)
    print(f"Number of Slurm jobs to be submitted: {batches}")

    # Fill NaN values in the start and end indices arrays to avoid errors
    start_indices = np.full(batches, np.NaN)
    end_indices = np.full(batches, np.NaN)

    # Calculate the start and end indices for each batch
    for i in range(batches):
        start_indices[i] = i * files_per_batch
        end_indices[i] = start_indices[i] + files_per_batch
        if end_indices[i] > len(gruan_file_paths):
            end_indices[i] = len(gruan_file_paths)

    job_ids = []  # Initialize an array to store job IDs
    for i in range(batches):
        start_index = int(start_indices[i])
        end_index = int(end_indices[i])
        batch = i + 1  # Batch number for slurm job, starting from 1

        # Save the required arguments to pass to met_processing.py into a csv file that can be easily read in
        args = {
        'start_index': start_index,
        'end_index': end_index,
        'files_per_batch': files_per_batch,
        'era5_file_paths': era5_file_paths,
        'gruan_file_paths': gruan_file_paths,
        'batch': batch
        }

        json_path = f'{generated_files_dir}/met_processing_args.json'

        with open(json_path, "w") as f:
            json.dump(args, f)

        # Submit the job and get the job ID
        print(f"Submitting batch {batch} with files from {start_index} to {end_index}")
        export_args = f"ARG1={json_path}"
        job_ids.append(submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.sh", "has_args", export_args))
        time.sleep(5)  # Optional: wait a bit before submitting the next job

    # Wait for the job to complete
    wait_for_specific_jobs(job_ids)

    complete_message = "Job(s) completed, continuing with the script!"
    ### WAIT FOR THE JOB TO FINISH THEN CONTINUE

    return complete_message

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
                else:
                    print(f"ERA5 file not found for {gruan_file_path}. Expected at {era5_file_path}.")

    return gruan_file_paths, era5_file_paths

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

############################################################################################################################################################
# Step 4: Create APCEMM meteorological input files, run APCEMM, and save the output

class APCEMMConfig:
    def __init__(self, **kwargs):
        #DEFAULTS
        # Specify the test
        self.aircraft_engine = "B737-800_CFM56"  # Which aircraft and engine are we using?
        self.test_id = "test_10"  # Test number the met and YAML files are associated with

        self.training_runs = 0  # Number of APCEMM runs to process for the test
        self.validation_runs = 0  # Number of APCEMM runs to process for the test
        self.novel_runs = 0 # Number of novel runs (no APCEMM)

        self.polynomial_degree = 1  # Polynomial degree for the PCE
        self.maximum_degree = 2  # Maximum degree for the PCE

        # Define the meteorological APCEMM file dimensions
        self.APCEMM_altitudes = 125  # Number of altitudes recorded in meteorological file
        self.APCEMM_timesteps = 24  # Hours recorded in meteorological file
        self.APCEMM_output_timesteps = 72 # Number of timesteps in the APCEMM output file (72*10 minutes = 12 hours)

        # RHi initial condition distribution details
        self.IC_std_rhi = 0
        self.IC_mean_amplitude_rhi = 0
        # RHi temporal distribution details
        self.time_std_rhi = 0
        self.time_mean_amplitude_rhi = 0

        # MLD initial condition distribution details
        self.IC_std_mld = 20
        self.IC_mean_amplitude_mld = 0
        # MLD temporal distribution details
        self.time_std_mld = 0
        self.time_mean_amplitude_mld = 0

        for key, val in kwargs.items():
            setattr(self, key, val)

def generate_apcemm_input_files(RHi_sampled, base_met_dir, num_met_files, set_type, test_specifications):
    # Set up output path
    if set_type == "validation":
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/inputs/validation/APCEMM_met_validation_{}.nc'
    else:
        out_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/inputs/training/APCEMM_met_{}.nc'
    output_file_template = out_path

    for run_num in range(1,num_met_files+1):
        # Consolidate the sampled RHi values into a matrix
        RHi_array = RHi_sampled[run_num-1,:]  # Get the sampled RHi for the current run [timesteps]

        RHi_sampled_matrix = np.tile(RHi_array, (16, 1)) # Replacing only 16 altitude layers with samples. The rest are predefined as subsaturated. #MUST BE UPDATED WITH MLD ADDITION

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
    source_path = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/YAML_base_files/{}_APCEMM_input.yaml".format(aircraft_engine)  # Source YAML file
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

def combine_parquet_files():
    """
    Combine all Parquet files in the input directory into a single Parquet file.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the Parquet files.
    output_file : str
        Path to save the combined Parquet file.
    """
    # This is where the Parquet files are stored before combining
    input_dir = "/home/chinahg/GCresearch/contrailuncertainty/Met_processing/parquet_files"
    # This is where the combined Parquet file will be saved or is already saved
    output_file = f'/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/combined_data.parquet'

    if os.path.exists(output_file):
        print(f"Output file already exists. Skipping combination.")
        combined_df = pd.read_parquet(output_file)
        return combined_df
    
    # Get a list of all Parquet files in the input directory
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files into a single DataFrame
    combined_df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    # Save the combined DataFrame to a single Parquet file
    combined_df.to_parquet(output_file, index=False)

    return combined_df

############################################################################################################################################################
# Step 4: Create APCEMM meteorological input files, run APCEMM, and save the output

def create_and_validate_surrogate(test_specifications):
    # Load in APCEMM inputs and outputs
    # APCEMM inputs that were trained on, and we also have the True solution for
    raw_training_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/outputs/training/training_sample_matrix.npy'.format(test_specifications.test_id)) # Import APCEMM training data
    # APCEMM inputs that were not trained on, and we also have the True solution for
    raw_validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/outputs/validation/validation_sample_matrix.npy'.format(test_specifications.test_id)) # Import APCEMM validation data

    # APCEMM has issues with artifacts, smooth those out as pre-processing
    clean_training_samples_matrix_offline = np.copy(raw_training_samples_matrix_offline)
    clean_validation_samples_matrix_offline = np.copy(raw_validation_samples_matrix_offline)

    for i in range(test_specifications.training_runs):
        clean_training_samples_matrix_offline[1, i, :] = smooth_artifacts(raw_training_samples_matrix_offline[1,i,:])
    for i in range(test_specifications.validation_runs):
        clean_validation_samples_matrix_offline[1, i, :] = smooth_artifacts(raw_validation_samples_matrix_offline[1,i,:])

    # Save the smoothed training and validation sample matrices
    # Specify the directory to save the results
    results_dir = f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_specifications.test_id}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=False)

    np.save(f"{results_dir}/training_samples_matrix_offline_cleaned.npy", clean_training_samples_matrix_offline)
    np.save(f"{results_dir}/validation_samples_matrix_offline_cleaned.npy", clean_validation_samples_matrix_offline)

    # Create the PCE with the training data
    coefficients, alpha = create_PCE(test_specifications, "training")
    
    # Save the PCE coefficients and alpha set
    np.save(f"{results_dir}/PCE_coefficients.npy", coefficients)
    np.save(f"{results_dir}/PCE_alpha_set.npy", alpha)

    print("PCE model created successfully!")

    # Validate the PCE model with the validation data
    # Predict the validation solutions based on the trained PCE
    predicted_validation_solutions = predict_validation_solutions(coefficients, alpha, test_specifications)

    print("PCE validation tests ran successfully!")

    # Save the predicted and true validation solutions
    np.save(f"{results_dir}/predicted_validation_solutions.npy", predicted_validation_solutions)
    np.save(f"{results_dir}/true_validation_solutions.npy", true_validation_solutions("validation", test_specifications.test_id))

    # Save the training solutions
    np.save(f"{results_dir}/true_training_solutions.npy", true_training_solutions("training", test_specifications.test_id))

    complete_message = f"Saved PCE training and validation results here: {results_dir}"

    return complete_message

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
    sample_arrays = []

    for i in range(1, num_runs+1): # For test runs num_runs
        if set_type == "validation": # If you are processing a validation set
            apcemm_data = read_apcemm_data(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/outputs/validation/{test_id}_run_{i}')
            input_RHi_ds = xr.open_dataset(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/inputs/validation/APCEMM_met_validation_{i}.nc')
        elif set_type == "training": # If you are processing a training set
            apcemm_data = read_apcemm_data(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/outputs/training/{test_id}_run_{i}')
            input_RHi_ds = xr.open_dataset(f'/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_id}/inputs/training/APCEMM_met_{i}.nc')
        elif set_type != "novel":
            raise ValueError("Invalid set_type. Must be 'training', 'validation', or 'novel'.")
        
        # Import the ERA5 and GRUAN data
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
        IC_mean_amplitude_rhi = input_RHi[0] # The first value is the mean amplitude of the RHi initial condition, used to normalize the RHi values

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

# Downselect the DataFrame to include only rows where 8000 <= E_alt <= 13000 and 8000 <= G_alt <= 13000
def filter_row_by_alt(row):
    # Create a boolean mask for G_alt between 8000 and 13000
    mask = (row['G_alt'] >= 9000) & (row['G_alt'] <= 12000)

    # Apply the mask to all relevant columns
    row['G_alt'] = row['G_alt'][mask]
    row['G_RHi'] = row['G_RHi'][mask]  # do this for each array-like column
    row['E_RHi_diffused'] = row['E_RHi_diffused'][mask]
    return row
    
# def get_initial_dist(already_combined):
#     # Create a probability density function (PDF) for the RHi and MLD distributions
#     # This function will load the GRUAN and ERA5 data and create a PDF for each variable

#     # Combine all parquet files in the base directory into a single DataFrame
#     # combined_parquet_name = 'combined_data.parquet'
#     if already_combined == False:
#         combined_df = combine_parquet_files() # Check the first few rows of the combined DataFrame
#     elif already_combined == True:
#         # Load the combined DataFrame from the parquet file
#         print("Loading combined DataFrame from parquet file...")
#         combined_df = pd.read_parquet("/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/combined_data.parquet")

#     # Extract the RHi and MLD columns from the DataFrame, keeping the GRUAN and ERA5 data separate
#     print("Extracting GRUAN and ERA5 data...")
#     print(combined_df.dtypes)
#     # Print the rows in the combined DataFrame
#     print("Rows in the combined DataFrame:")
#     print(combined_df)
#     print("Number of rows in combined DataFrame:", len(combined_df))

#     # Apply to the DataFrame
#     combined_df = combined_df.apply(filter_row_by_alt, axis=1)
    
#     gruan_RHi = np.array(np.concatenate(combined_df['G_RHi'].values),dtype=float)
#     era5_RHi = np.array(np.concatenate(combined_df['E_RHi_diffused'].values),dtype=float)
#     # gruan_MLD = np.array(combined_df['G_MLD'].values,dtype=float) # FIX PARQUET UNPACKING
#     # era5_MLD = np.array(combined_df['E_MLD'].values,dtype=float)

#     # Remove NaN values from the data
#     gruan_RHi = gruan_RHi[~np.isnan(gruan_RHi)]
#     era5_RHi = era5_RHi[~np.isnan(era5_RHi)]
#     # gruan_MLD = gruan_MLD[~np.isnan(gruan_MLD)]
#     # era5_MLD = era5_MLD[~np.isnan(era5_MLD)]

#     # Create a PDF for each variable using the GRUAN and ERA5 data using KDE (Kernel Density Estimation)
#     print("Creating PDFs for GRUAN and ERA5 data...")
#     gruan_RHi_pdf = gaussian_kde(gruan_RHi)
#     era5_RHi_pdf = gaussian_kde(era5_RHi)
#     # gruan_MLD_pdf = gaussian_kde(gruan_MLD.explode().astype(float))
#     # era5_MLD_pdf = gaussian_kde(era5_MLD.explode().astype(float))

#     return gruan_RHi, era5_RHi, gruan_RHi_pdf, era5_RHi_pdf#, gruan_MLD_pdf, era5_MLD_pdf

#############################################################################################################################################################
# Step 5: Create and validate the PCE model

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
def predict_validation_solutions(c, alpha_set, test_specifications): # Validation inputs to PCE from offline pre-computed validation set
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
        training_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{}/training_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM training data

        training_samples_matrix_offline_OUTPUTS = training_samples_matrix_offline[1,:,:] # (depth, row, column) --> (0=input 1=output, number of datasets to train PCE, timesteps)
        # Transpose the rows and columns to accomodate later calculations
        training_samples_matrix_offline_OUTPUTS = training_samples_matrix_offline_OUTPUTS.T # (timesteps, number of datasets to train PCE)
        return training_samples_matrix_offline_OUTPUTS
    
    elif sample_type == "validation":
        validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{}/validation_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM validation data

        validation_samples_matrix_offline_OUTPUTS = validation_samples_matrix_offline[1,:,:]
        validation_samples_matrix_offline_OUTPUTS = validation_samples_matrix_offline_OUTPUTS.T
        return validation_samples_matrix_offline_OUTPUTS
    
    else: 
        print("Invalid sample type. Please enter 'training' or 'validation'.")
        return None


def get_samples_matrix_offline(sample_type:str, test_id):

    if sample_type == "training":
        training_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{}/training_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM training data

        training_samples_matrix_offline_INPUTS = training_samples_matrix_offline[0,:,:] # (depth, row, column) --> (0=input 1=output, number of datasets to train PCE, timesteps)
        training_samples_matrix_offline_INPUTS = training_samples_matrix_offline_INPUTS.T # (timesteps, number of datasets to train PCE)
        print(training_samples_matrix_offline_INPUTS.shape)
        return training_samples_matrix_offline_INPUTS
    
    elif sample_type == "validation":
        validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{}/validation_samples_matrix_offline_cleaned.npy'.format(test_id)) # Import APCEMM validation data

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
# Step: Compute sensitivity indices

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

#############################################################################################################################################################
# Step: Make novel input matrices for the surrogate model

def post_process_met_data():
    df_combined = combine_parquet_files()
    
    # Extract RHi values for GRUAN and ERA5
    # Each cell in 'G_RHi' and 'E_RHi_diffused' is a list; flatten them for all rows
    gruan_rhi = [item for sublist in df_combined['G_RHi'] for item in sublist]
    era5_rhi = [item for sublist in df_combined['E_RHi_diffused'] for item in sublist]

    e_indices_to_remove = []
    g_indices_to_remove = []

    cleaned_gruan_rhi = gruan_rhi[:]
    cleaned_era5_rhi = era5_rhi[:]

    # Check if any values are NaN
    for i in range(len(gruan_rhi)):
        if pd.isna(gruan_rhi[i]):
            g_indices_to_remove.append(i)

    for i in range(len(era5_rhi)):
        if pd.isna(era5_rhi[i]):
            e_indices_to_remove.append(i)

    combined_indices_to_remove = sorted(g_indices_to_remove + e_indices_to_remove, reverse=True)

    for index in combined_indices_to_remove:
        if 0 <= index < len(cleaned_era5_rhi):
            cleaned_era5_rhi.pop(index)
            cleaned_gruan_rhi.pop(index)

    # Convert era5_rhi and gruan_rhi to numpy arrays for easier manipulation
    cleaned_era5_rhi = np.array(cleaned_era5_rhi)
    cleaned_gruan_rhi = np.array(cleaned_gruan_rhi)

    # Check if any values are NaN after the removal
    if np.isnan(cleaned_era5_rhi).any() or np.isnan(cleaned_gruan_rhi).any():
        print("NaN values still present after removal!")

    # Filter pairs where ERA5 RHi is less than or equal to 200
    mask = cleaned_era5_rhi <= 150
    filtered_gruan_rhi = cleaned_gruan_rhi[mask]
    filtered_era5_rhi = cleaned_era5_rhi[mask]

    # Fit the ERA5 data to a Gaussian distribution, but only for RHi values between 90 and 140
    era5_rhi_input = filtered_era5_rhi[(filtered_era5_rhi >= 110) & (filtered_era5_rhi <= 140)]
    mirror_data = 2*3 - era5_rhi_input + 214

    # Combine the original and mirrored data
    combined_data = np.concatenate((era5_rhi_input, mirror_data))
    # Fit a Gaussian distribution to the combined data
    mean = np.mean(combined_data)
    std_dev = np.std(combined_data)

    return mean, std_dev, filtered_gruan_rhi, filtered_era5_rhi

def make_surrogate_input(test_specifications, run_type):
    test_id = test_specifications.test_id
    
    if run_type == 'training':
        timesteps = test_specifications.APCEMM_timesteps # 24 hours
        num_runs = test_specifications.training_runs
    elif run_type == 'validation':
        timesteps = test_specifications.APCEMM_timesteps # 24 hours
        num_runs = test_specifications.validation_runs
    elif run_type == "novel":
        timesteps = test_specifications.APCEMM_output_timesteps # 72 * 10 minutes = 720 minutes = 12 hours
        num_runs = test_specifications.novel_runs
    else:
        raise ValueError("Invalid run_type. Must be 'novel', 'training', or 'validation'.")
    
    mean_rhi, std_rhi, gruan_rhi, era5_rhi = post_process_met_data()
    initial_rhi = np.zeros(num_runs) # Initialize the initial RHi values array
    
    # Only keep values where a contrail will form, i.e. RHi >= 117
    i = 0
    while i < num_runs:
        sampled_value = np.random.normal(mean_rhi, std_rhi, 1)
        if sampled_value >= 117 and sampled_value <= 140: # Ensure the sampled value is within the range of RHi values
            initial_rhi[i] =  sampled_value # This is how we will sample the initial conditions
            i = i+1

    # Sample the timesteps after the initial condition
    # Input: Initial RHi conditions
    # Steps: Map each IC to error distribution, sample error distribution for timesteps
    # Output: Matrix of size (num_runs, 72) with sampled RHi values
    # Compute the difference between ERA5 and GRUAN RHi values
    
    # Compute the difference between ERA5 and GRUAN RHi values
    rhi_diff = np.abs(np.array(era5_rhi) - np.array(gruan_rhi))
    
    rhi_time = np.zeros((num_runs, timesteps-1)) # Initialize the time-dependent RHi values matrix (num_runs, timesteps-1)
    for i in range(num_runs):
        rhi = initial_rhi[i]
        idx_nearest = np.abs(np.array(era5_rhi) - rhi).argmin() # Find the index of the closest RHi value in the ERA5 data

        print(f"Sampling RHi for run {i+1}/{num_runs} with initial RHi: {rhi:.2f}")

        # Get the corresponding error from the RHi difference
        IC_error = rhi_diff[idx_nearest]
        print(f"IC error for run {i+1}: {IC_error:.2f}")

        # Define the variance of a gaussian by the error: 99.7% of the data lies within +- IC/4
        std_time = IC_error / 4

        # Sample the initial RHi values from a Gaussian distribution
        rhi_time[i,:] = np.random.normal(rhi, std_time, timesteps-1) # runs x timesteps-1
    
    # Make initial_rhi shape (num_runs, timesteps) with the first column being the initial RHi values
    initial_rhi = initial_rhi.reshape(num_runs, 1) # Reshape to (num_runs, 1)
    input_rhi = np.hstack((initial_rhi, rhi_time)) # Concatenate the initial RHi values with the sampled time-dependent RHi values
    
    if run_type == "novel":
        # Save the input RHi matrix for novel runs
        if not os.path.exists(f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}"):
            os.makedirs(f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}", exist_ok=False)
        
        print(f"Saving novel samples matrix to: /home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}/novel_samples_matrix.npy")
        np.save(f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}/novel_samples_matrix.npy", input_rhi)

    # # Get temperature fluxuations from the ERA5 data and apply to the initial temperature values
    # # Find the temperature associated with the closest RHi value to the sampled initial RHi
    # df_combined = combine_parquet_files(input_dir, output_file)
    # closest_indices = []
    # for rhi_value in initial_rhi:
    #     closest_index = (np.abs(df_combined['E_RHi_diffused'] - rhi_value)).idxmin()
    #     closest_indices.append(closest_index)
    # initial_temperatures = df_combined['G_T'].iloc[closest_indices].values
    # initial_q = df_combined['G_q'].iloc[closest_indices].values

    # # We have the initial temperature associated with the initial RHi values, now we need to sample the temperature fluctuations
    # # Get the temperature values from the met file

    # # Read in the temperature array at pressure level 240 hPa (or the closest pressure level to that)

    # met_file_temperatures = ds_met['temperature'][idx_240, :].values  # shape: (timesteps,)
    # met_file_q = ds_met['specific_humidity'][idx_240, :].values  # shape: (timesteps,)
    # ds_met.close()

    return input_rhi#, initial_temperatures

def novel_solutions(c, alpha_set, test_specifications): # Validation inputs to PCE from offline pre-computed validation set
    novel_runs = test_specifications.novel_runs
    test_id = test_specifications.test_id
    multiindices = alpha_set.shape[0]
    He_array = np.zeros((multiindices, novel_runs))
    results_dir = f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}"
    samples_matrix = np.load(f"{results_dir}/novel_samples_matrix.npy")

    for i in tqdm.tqdm(range(novel_runs)): # for each run
        
        for j in range(multiindices): # for each coefficient
            current_alpha = alpha_set[j, :] # look at one row at a time for all alpha describing a single coefficient
            samples_matrix_He = samples_matrix[i, :]
            He_array[j, i] = compute_He(samples_matrix_He, current_alpha)

    # Solve for the Least Squares solution
    novel_solutions = (c.T @ He_array).T

    print("PCE novel tests ran successfully!")

    # Save the predicted and true validation solutions
    np.save(f"{results_dir}/novel_solutions.npy", novel_solutions)

    complete_message = f"Saved PCE novel sample results here: {results_dir}/novel_solutions.npy"

    return complete_message

#############################################################################################################################################################################
# Step : Compute fluxes for each timestep

def updateInput(filepath, attributes, contrail, type):
    """
    Update the input file with the given attributes.

    Parameters:
    - filepath (str): The path of the input file to be updated.
    - attributes (object): An object containing the attributes to be written to the input file.
    - contrail (bool): A flag indicating whether the input file is for contrail simulation or not.

    Returns:
    None
    """
    file = open(filepath,"w")
    if contrail == True and type == "water":
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n",
                            "wc_file 1D cloud.in\n",
                            "quiet"])
        file.close()
    elif contrail == True and type == "ice":
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n", 
                            "ic_habit "+str(attributes.ic_habit[0])+"\n",
                            "ic_properties "+str(attributes.ic_properties[0])+"\n",
                            "ic_file 1D "+attributes.ic_file[0]+"\n",
                            "ic_modify tau set " +attributes.ic_modify[0]+"\n",
                            "quiet"])
        file.close()
    else: 
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n",
                            "quiet\n"])
        file.close()
    
# def clearskyRF(attributes):
#     """
#     Run the clear sky radiative forcing simulation.

#     Parameters:
#     - attributes (object): An object containing the attributes for the simulation.

#     Returns:
#     - LRToutput (list): A list containing the output of the simulation.
#     """
#     updateInput("/home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-clear.in", attributes, False, "None")
#     LRToutput = !LD_LIBRARY_PATH=/data/home/chinahg/.conda/envs/afca-test/lib:$LD_LIBRARY_PATH /home/iross/misc-code/libRadtran/bin/uvspec < /home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-clear.in # [X,X,X,net TOA flux]
#     return LRToutput

# def contrailRF(attributes, type):
#     """
#     Run the contrail radiative forcing simulation.

#     Parameters:
#     - attributes (object): An object containing the attributes for the simulation.

#     Returns:
#     - LRToutput (list): A list containing the output of the simulation.
#     """
#     updateInput("/home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-cloud.in", attributes, True, type)
#     LRToutput = !LD_LIBRARY_PATH=/data/home/chinahg/.conda/envs/afca-test/lib:$LD_LIBRARY_PATH /home/iross/misc-code/libRadtran/bin/uvspec < /home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-cloud.in # [X,X,X,X,X,net TOA flux]
#     return LRToutput

# def reformatResults(resultsRaw):
#     """
#     Reformat the raw results.

#     Parameters:
#     - resultsRaw (list): A list containing the raw results.

#     Returns:
#     - li (list): A list containing the reformatted results.
#     """
#     string = str(resultsRaw[0].strip().replace("  ", " "))
#     li = list(string.split(" ")) 
#     return li

