# Import functions from the pipeline_fxn_lib.py script
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib
import run_CoCiP_fxn_lib as cocip_lib

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails import MetDataset
from pycontrails.models.cocip import contrail_properties as cp

test_id = "110T218L25"
temperature_test = float(test_id[4:7])  # Extract temperature from test ID
altitude_test = 10700.00  # Altitude in meters for this test
post_vortex_altitude = 10654.12 # Function of aircraft properties and BV stratification level, same for all tests with initial altitude 10700 m
altitude_initial = altitude_test + (altitude_test - post_vortex_altitude)  # Initial altitude before wake vortex effects

EInvpm_string = sys.argv[1] # Emissions index being tested
EInvpm_test = float(EInvpm_string)
number_type = "EInvpm"  # Type of number being used: "number per meter" for ice particles per meter, "EInvpm" for emissions index
number = EInvpm_test
r_ice_string = sys.argv[2] # String version of ice crystal radius for file naming, in micrometers
r_ice = float(r_ice_string)  # m, ice crystal radius for this test
f_surv = 1.0  # Fraction of ice crystals that survive wake vortex sinking for this test

u = 237  # Set true airspeed to 237 m/s
efficiency = 0.3
mdot_f = 1.36986 # Estimated using cruise velocity and fuel flow rate from Lewellen et al. (2014) [kg/s]
epsilon = 0.622 # Ratio of molecular weights of dry air and water vapor
q_fuel = 43e6 # Specific heat of combustion [J/kg]
ei_h2o = 1.24 # emission index of water vapor [kg/kg]
air_pressure = lib.alt2press(altitude_initial) # [Pa]
c_pd = 1005 # Specific heat of dry air
c_pv = 1850 # Specific heat of water vapor
RH = float(test_id[0:3])/100 # Fraction of RHi
q = cocip_lib.estimate_specific_humidity(temperature_test, air_pressure, RH) # Specific humidity
c_pm = c_pd * (1 - q) + c_pv * q # Specific heat capacity of moist air
G = (ei_h2o * c_pm * air_pressure) / (epsilon * q_fuel * (1 - efficiency)) # Slope of SAC
T_LM = cocip_lib.T_sat_liquid(G) # [K]

activation_rate = cocip_lib.ice_particle_activation_rate(temperature_test, T_LM)
fuel_per_m = mdot_f / u  # [kg/m]
N = EInvpm_test * activation_rate * fuel_per_m * f_surv

save_dir = "micro_sweeps_110_218/" # Where the met and rad data will be saved, used for all tests
base_flight_csv_path = "/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/110T218L25-flight-cocip-aligned.csv"
new_flight_csv_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/EInvpm/{EInvpm_string}/{test_id}-flight-cocip-aligned.csv" # Where to save new flight data csv or to read existing data from

make_flight_csv = True if sys.argv[3] == "True" else False  # Set to True to create a new CSV file for the flight. A new CSV is needed if you change the flight properties (ex. EInvpm or wingspan)
save_data = True
check_if_cloudless = False # Whether to check if the ERA5 data is cloudless at the flight points

print(f"Running CoCiP for EInvpm = {EInvpm_test} and r_ice = {r_ice} m.")
print(f"Number of ice crystals per meter: {N}.")
print(f"Save results: {save_data}.")
print(f"Flight CSV path: {new_flight_csv_path}.")
print(f"Make new flight CSV: {make_flight_csv}.")

# Monkeypatch of iwc_post_wake_vortex function
def my_iwc_post_wake_vortex(iwc, iwc_ad):
    # Calculate the post-vortex iwc such that it follows a specified crystal number per meter and crystal radius
    rho_ice = 917  # kg/m3
    R_dry = 287.05  # J/(kg·K), from CoCiP physics.constants
    p = lib.alt2press(altitude_test) # Pa
    T = temperature_test  # K
    area_eff = 3094.290202803097  # For B767, wingspan = 47.25 m
    rho_air = p / (R_dry * T)
    
    iwc1 = np.ones_like(iwc) * r_ice**3 * 4*np.pi/3 * rho_ice * N * (area_eff * rho_air)**(-1)
    return iwc1

# Monkeypatch the ice_particle_survival_fraction and iwc_post_wake_vortex functions in contrail_properties.py
cp.ice_particle_survival_fraction = cocip_lib.my_ice_particle_survival_fraction
cp.iwc_post_wake_vortex = my_iwc_post_wake_vortex

if make_flight_csv:
    cocip_lib.make_flight_csv(altitude_initial, base_flight_csv_path, number, new_flight_csv_path, T_LM, mdot_f, u, efficiency, f_surv, number_type, temperature_test)

df_flight, attrs = cocip_lib.format_flight_csv(new_flight_csv_path)

fl = Flight(data=df_flight, attrs=attrs)

# Get radiative data for clear sky at the nearest lat/lon/time from flight data
# We checked that for time=2025-06-29 00:00 UTC, lat=45.00, lon=-46.50, the fraction_of_cloud_cover is 0 at all levels
time_of_day = "midnight"
time = pd.to_datetime(fl["time"][0]).floor("h")

# Download radiation data from ERA5
era5_single_rad = ERA5(time=time,variables=Cocip.rad_variables).open_metdataset()
xr_era5_single = era5_single_rad.data

# Broadcast the clearsky data to have the same time dimension as the flight data
# Define your desired coordinate grids
longitudes = np.linspace(-180, 179.75, int((180+179.75)/(0.25) + 1))
latitudes = np.linspace(30, 70, int((70-30)/(0.25) + 1))
times = pd.date_range(time, periods=40, freq="1H")

# Select data to broadcast to new xarray dataset
desired_time = xr_era5_single["time"].values[0]
desired_latitude = 45.0
desired_longitude = -46.5

xr_era5_rad = xr_era5_single.assign(
    {var: xr_era5_single[var].sel(time=desired_time, latitude=desired_latitude, longitude=desired_longitude).broadcast_like(
        xr.DataArray(np.empty((len(times), len(latitudes), len(longitudes))),
                     dims=("time", "latitude", "longitude"),
                     coords={"time": times, "latitude": latitudes, "longitude": longitudes})
    ) for var in xr_era5_single.data_vars}
)

# Save the modified rad data to a new NetCDF file
xr_era5_rad.to_netcdf(f"{save_dir}/{test_id}-{time_of_day}-rad.nc", mode="w", format="NETCDF4")

# create rad `MetDataset` to feed to CoCiP
rad = MetDataset(xr_era5_rad)

met_from_scratch = False  # Set to True to create the met dataset from scratch

if met_from_scratch:
    # Make the met file

    # Import APCEMM met data
    xr_apcemm_met = xr.open_dataset(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/base_inputs/{test_id}/{test_id}.nc")

    # Define dimensions
    longitudes = np.linspace(-180, 179.75, int((180+179.75)/(0.25) + 1))

    latitudes = np.linspace(30, 70, int((70-30)/(0.25) + 1))
    levels = np.flip(xr_apcemm_met['pressure'].values)

    UTC = pd.date_range(start="2025-06-29 00:00:00", periods=60, freq="1h").astype(np.int64) // 10**9
    time = pd.to_datetime(UTC, origin="unix", unit="s") # Convert the UTC time to datetimes and assign to the "time" column

    # Define the additional coordinates to add
    altitudes = np.flip((xr_apcemm_met['altitude'].values).astype(np.float32)) * 1000  # Convert altitude from km to m
    pressures = levels * 100  # Convert pressure from hPa to Pa

    # Define the shpae of the dataset
    new_shape = (len(longitudes), len(latitudes), len(levels), len(time))

    # Create the data to be saved in the new dataset (variables)
    air_temperature = np.broadcast_to(
        np.flip(xr_apcemm_met['temperature'][:,0].values)[np.newaxis, np.newaxis, :, np.newaxis],
        new_shape
    ).astype(np.float32)

    relative_humidity = np.broadcast_to(
        np.flip(xr_apcemm_met['relative_humidity_ice'][:,0].values)[np.newaxis, np.newaxis, :, np.newaxis],
        new_shape
    ).astype(np.float32)

    northward_wind = np.zeros_like(relative_humidity, dtype=np.float32) # Stays 0, only horizontal shearing
    lagrangian_tendency_of_air_pressure = np.zeros_like(relative_humidity, dtype=np.float32) # Stays 0, no downward motion
    specific_cloud_ice_water_content = np.zeros_like(relative_humidity, dtype=np.float32) # Stays 0, no cloud cover

    # Calculate the specific humidity based on RHi and temperature
    T0 = 273.15  # Reference temperature in Kelvin
    T_base = air_temperature[0,0,:,0] # Temperature in Kelvin
    RHi_base = relative_humidity[0,0,:,0]
    pres_base = levels * 100  # Pressure in Pa
    P_sat_w = lib.compute_Psat_w(T_base)
    P_sat_i = lib.compute_Psat_i(T_base)
    q_base = (RHi_base * P_sat_i / P_sat_w) * (1 / (0.263 * pres_base)) * np.exp((17.67 * (T_base - T0)) / (T_base - 29.65))
    specific_humidity = np.broadcast_to(
            q_base[np.newaxis, np.newaxis, :, np.newaxis],  # lon, lat, level, time
            new_shape
        ).astype(np.float32)

    # Calculate the wind feild for constant shear of 0.004 m/s/m
    slope = 0.004  # m/s/m
    u = slope * levels
    eastward_wind = np.broadcast_to(
            u[np.newaxis, np.newaxis, :, np.newaxis],  # lon, lat, level, time
            new_shape
        ).astype(np.float32)

    # Create dataset with the variables and coordinates defined above
    ds_met = xr.Dataset(
        {
            "air_temperature": (['longitude', 'latitude', 'level', 'time'], air_temperature),
            "specific_humidity": (['longitude', 'latitude', 'level', 'time'], specific_humidity),
            "eastward_wind": (['longitude', 'latitude', 'level', 'time'], eastward_wind),
            "northward_wind": (['longitude', 'latitude', 'level', 'time'], northward_wind),
            "lagrangian_tendency_of_air_pressure": (['longitude', 'latitude', 'level', 'time'], lagrangian_tendency_of_air_pressure),
            "specific_cloud_ice_water_content": (['longitude', 'latitude', 'level', 'time'], specific_cloud_ice_water_content),
            "relative_humidity": (['longitude', 'latitude', 'level', 'time'], relative_humidity),

        },
        coords={
            'longitude': longitudes,
            'latitude': latitudes,
            'level': levels,
            'time': time,
            'altitude': ('level', altitudes),
            'air_pressure': ('level', pressures)
        },
        attrs=None
    )

    # Save the modified xr_met to a new NetCDF file
    ds_met.to_netcdf(f"{save_dir}/{test_id}-met.nc", mode="w", format="NETCDF4")

else:
    # Load in the existing met file
    ds_met = xr.open_dataset(f"{save_dir}/{test_id}-met.nc", )

# Rename the dimensions to match the expected format
new_met = MetDataset(ds_met)

params = {
    "process_emissions": False,
    "verbose_outputs": True,
    "humidity_scaling": ConstantHumidityScaling(rhi_adj=0.98),
    "max_age": np.timedelta64(60, "h"),
    "dt_integration": np.timedelta64(1, "m"),
    "min_ice_particle_number_nvpm_ei_n": 10e11,
}
cocip = Cocip(met=new_met, rad=rad, params=params)

fl_out = cocip.eval(source=fl)
waypoint = cocip.contrail[cocip.contrail.waypoint == 0]

# Convert the waypoint DataFrame to a DataSet
waypoint_ds = waypoint.to_xarray()

if save_data == True:
    # Save the waypoint DataFrame to a .nc file
    df_save_dir = f"{save_dir}/EInvpm/{EInvpm_string}/{EInvpm_string}_{r_ice_string}.nc"
    waypoint_ds.to_netcdf(df_save_dir)
    print("Results saved to {}.".format(df_save_dir))
else:
    print("Results not saved. Set save_data = True to save the data.")