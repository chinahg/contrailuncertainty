import xarray as xr
import datetime as dt
import numpy as np
import sys
import csv
import cdsapi
from zarr.errors import GroupNotFoundError
import yaml

# Import functions from the pipeline_fxn_lib.py script
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib

# Downloading ERA5 data
### FUNCTION LIBRARY ###
def read_csv_file(file_path):
    """
    Reads a CSV file and returns the values as a list.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.

    Returns
    -------
    values : list
        The values read from the CSV file.
    """
    values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            values.extend(row)
    return values


ERA5_PRESSURE_LEVELS = ['200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
        ]

ERA5_VARIABLES = ['relative_humidity', 
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'specific_humidity',
            'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']


EXTENT = [-180, 180, 30, 60] # Sub-region extraction [min_lon, max_lon, min_lat, max_lat]

PRODUCT_TYPES = ['ensemble_mean', 'ensemble_members', 'ensemble_spread',
            'reanalysis']

def get_CDSAPI_settings(time, variables=ERA5_VARIABLES,
                                    pressure_levels=ERA5_PRESSURE_LEVELS,
                                    extent=EXTENT,
                                    product_type="reanalysis"):
    """
    Parses the CDS (Copernicus Data Store) API settings for the given day,
    variables and pressure levels.
    
    Parameters
    ----------
    time : dt.datetime
        Day for which to get ERA5 data
    variables : List[string] (optional)
        Variables to download
    pressure_levels : List[string] or List[int] (optional)
        Pressure levels in hPa to download
    extent : List[float] (optional)
        Geodetic extent to use in format [min_lon, max_lon, min_lat, max_lat]
    product_type : str (optional)
        The product to download

    Returns
    -------
    CDSAPI_settings: dict
        Dictionary holding the CDSAPI settings
    """

    if product_type not in PRODUCT_TYPES:
        raise ValueError("Product type should be one of " ",".join(PRODUCT_TYPES))
    
    # Remove hour and minute from datetime
    time = time.replace(hour=0)
    time = time.replace(minute=0)


    # Time list
    times = [(time+dt.timedelta(hours=i)).strftime('%H:%M') for i in range(24)]

    # Re-order extent to comply with CDSAPI convention
    extent = [extent[3], extent[0], extent[2], extent[1]]

    # Convert pressure levels to strings
    pressure_levels = [str(p) for p in pressure_levels]

    CDSAPI_settings = {"variable": variables,
                        "pressure_level": pressure_levels,
                        "product_type": product_type,
                        "year" : f"{time.year}",
                        "month" : f"{str(time.month).rjust(2,'0')}",
                        "day" : f"{str(time.day).rjust(2,'0')}",
                        "time": times, 
                        "area" : extent,
                        "format" : "netcdf"
                       }
    return CDSAPI_settings


def get_ERA5_data(date):
    """
    Returns ERA5 file for requested date

    Parameters
    ----------
    date: dt.date 
        Requested date
    
    Returns
    -------
    ds: xr.Dataset
        xarray dataset holding ERA5 data
    """

    path = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/" + date.strftime("%Y/%Y_%m_%d.nc")

    return xr.open_dataset(path)

def download_ERA5_data(time, save_path, variables=ERA5_VARIABLES,
                            pressure_levels=ERA5_PRESSURE_LEVELS,
                            extent=EXTENT):
    """
    Downloads ERA5 pressure level data using the CDSAPI.

    Parameters
    ----------
    time : dt.datetime
        Day for which to get ERA5 data
    variables : List[string], optional
        Variables to download, by default ERA5_VARIABLES
    pressure_levels : List[string] or List[int], optional
        Pressure levels in hPa to download, by default ERA5_PRESSURE_LEVELS
    extent : List[float], optional
        Geodetic extent to use in format [min_lon, max_lon, min_lat, max_lat], by default EXTENT
    """

    # Get the API settings
    CDSAPI_settings = get_CDSAPI_settings(time, variables=variables,
                                                pressure_levels=pressure_levels,
                                                extent=extent)

    c = cdsapi.Client()
    
    c.retrieve("reanalysis-era5-pressure-levels", CDSAPI_settings, save_path)

def request_ERA5_data(csv_path, start_index, end_index):

    # For a custom time period, download ERA5 data
    # Read the CSV file containing dates to download
    files2download = read_csv_file(csv_path)

    for i in range(start_index, end_index):
        date_str = files2download[i]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        date = dt.datetime(year, month, day)
        save_path = "/home/chinahg/GCresearch/ERA5_downloads/" + str(year) + "/" + date.strftime("%Y_%m_%d.nc")
        download_ERA5_data(date, save_path)

#####################################################################################################################

# Import the job download details
download_details_path = sys.argv[1]
# Read in the variable associated with the download details
with open(download_details_path, "r") as f:
    details_dict = yaml.safe_load(f)

download_details = lib.download_details(**details_dict)
files2download_path = download_details.files2download_path
start_index = download_details.start_index
end_index = download_details.end_index

request_ERA5_data(files2download_path, start_index, end_index)