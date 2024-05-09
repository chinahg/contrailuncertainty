import xarray as xr
import datetime as dt
import sys
sys.path.append('/home/chinahg/GCresearch/contrails/contrails/meteorology')
import cdsapi

from zarr.errors import GroupNotFoundError


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
            'temperature']


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
                        "format" : "grib"
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

    path = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/Data/" + date.strftime("%Y/%Y_%m_%d.nc")

    return xr.open_dataset(path)


def download_ERA5_data(path, time, variables=ERA5_VARIABLES,
                            pressure_levels=ERA5_PRESSURE_LEVELS,
                            extent=EXTENT):
    """
    Downloads ERA5 pressure level data using the CDSAPI

    Parameters
    ----------
    path : str
        Location to store downloaded data
    time : dt.datetime
        Day for which to get ERA5 data
    variables : List[string] (optional)
        Variables to download
    pressure_levels : List[string] or List[int] (optional)
        Pressure levels in hPa to download
    extent : List[float] (optional)
        Geodetic extent to use in format [min_lon, max_lon, min_lat, max_lat]
    """

    # Get the API settings
    CDSAPI_settings = get_CDSAPI_settings(time, variables=variables,
                                                pressure_levels=pressure_levels,
                                                extent=extent)

    c = cdsapi.Client()
    
    c.retrieve("reanalysis-era5-pressure-levels", CDSAPI_settings)

# Call function of choice
sdate = dt.datetime(2012, 1, 1) # Start date
edate = dt.datetime(2012, 12, 31) # End date
delta = dt.timedelta(days=1) # Timestep

save_path = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/Data/"
day = sdate

while day <= edate:
    download_ERA5_data(save_path, day)
    # increment start date by timedelta
    day += delta

