import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib
from pycontrails.models.cocip import contrail_properties as cp
from pycontrails.physics import units
import numpy as np
import pandas as pd

def format_flight_csv(path):
    df_flight = pd.read_csv(path)

    # constant properties along the length of the flight
    attrs = {
        "flight_id": "fid",
        "aircraft_type": df_flight["ICAO Aircraft Type"].values[0],
        "wingspan": df_flight["Wingspan (m)"].values[0]
    }

    # rename a few columns for compatibility with `Flight` requirements
    df_flight = df_flight.rename(
        columns={
            "Longitude (degrees)": "longitude",
            "Latitude (degrees)": "latitude",
            "True airspeed (m s-1)": "true_airspeed",
            "Mach Number": "mach_number",
            "Aircraft mass (kg)": "aircraft_mass",
            "Fuel mass flow rate (kg s-1)": "fuel_flow",
            "Overall propulsion efficiency": "engine_efficiency",
            "nvPM number emissions index (kg-1)": "nvpm_ei_n",
            "Altitude (m)": "altitude",
        }
    )

    # clean up a few columns before building Flight class
    df_flight = df_flight.drop(
        columns=["ICAO Aircraft Type", "Wingspan (m)", "UTC time"]
    )
    return df_flight, attrs

def ice_particle_activation_rate(air_temperature, T_crit_sac):
    d_temp = air_temperature - T_crit_sac
    if d_temp > 0.0:
        d_temp = 0.0
    if d_temp < -5.0:
        d_temp = float('-inf')
    return -0.661 * np.exp(d_temp) + 1.0

def T_sat_liquid(G):
    G_log = np.log(G - 0.053)
    T_LM = (-46.46 + 9.43 * G_log + 0.72 * G_log**2) + 273.15 # Convert to K
    return T_LM

def estimate_specific_humidity(air_temperature, air_pressure, rh):
    air_temperature_celsius = units.kelvin_to_celsius(air_temperature)
    exponent = (7.5 * air_temperature_celsius) / (237.3 + air_temperature_celsius)
    P_sat = 6.107 * 10 ** exponent  # Saturation vapor pressure, hPa
    air_pressure_hpa = air_pressure / 100
    numer = 0.62197058 * rh * P_sat
    denom = air_pressure_hpa - rh * P_sat
    return numer / denom

# Monkeypatch the ice_particle_survival_fraction and iwc_post_wake_vortex functions in contrail_properties.py
def my_ice_particle_survival_fraction(iwc, iwc_1):
    # Make survival fraction = 1
    f_surv = np.ones_like(iwc)
    return f_surv

def make_flight_csv(altitude_initial, longitude, latitude, time, base_flight_csv_path, number, new_flight_csv_path, T_LM, mdot_f, u, efficiency, f_surv, number_type, T):
    # Update the CSV file with B767 characteristics
    df_flight = pd.read_csv(base_flight_csv_path)

    df_flight["Longitude (degrees)"] = np.linspace(longitude, longitude + (len(df_flight) - 1)*0.25, len(df_flight)) # Set longitude
    df_flight["Latitude (degrees)"] = latitude # Set latitude
    df_flight["Altitude (m)"] = altitude_initial # Set altitude to pre-vortex altitude, to account for sinking later

    df_flight["True airspeed (m s-1)"] = u

    # Calculate the mach number
    gamma = 1.4  # Specific heat ratio for air at 10.7 km
    R = 287.05  # Specific gas constant for dry air in J/(kg·K), from CoCiP physics.constants
    a = np.sqrt(gamma*R*T)  # Speed of sound at 10.7 km
    M = df_flight["True airspeed (m s-1)"] / a  # Speed of sound at 10.7 km
    df_flight["Mach Number"] = M  # Set Mach number
    
    activation_rate = ice_particle_activation_rate(T, T_LM)
    print(f"Activation rate: {activation_rate}")
    fuel_per_m = mdot_f / u  # [kg/m]

    # Calculate the initial EInvpm to match the LLES values after vortex sinking
    if number_type == "number per meter":
        EI_nvpm = number * (activation_rate * fuel_per_m * f_surv)**(-1)
        print(f"Calculated EInvpm to match {number:.2e} nvpm/m after vortex sinking: {EI_nvpm:.2e} #/kg fuel")
    elif number_type == "EInvpm":
        EI_nvpm = number

    df_flight["Aircraft mass (kg)"] = 152281.72  # At beginning of cruise (35k ft) for B767-300 (uses Trent 4000, same as all B767 models)
    df_flight["Fuel mass flow rate (kg s-1)"] = mdot_f
    df_flight["Overall propulsion efficiency"] = efficiency  # Set overall propulsion efficiency to 30%
    df_flight["nvPM number emissions index (kg-1)"] = EI_nvpm # Number per kg fuel
    df_flight["ICAO Aircraft Type"] = "B767"
    df_flight["Wingspan (m)"] = 47.25 # Set wingspan to 47.25 m

    # Calculate the UTC time in seconds for Noon at 45N, 45W on June 29, 2025 at 83 second intervals (1/4 degree latitude travelled)
    df_flight["UTC time"] = pd.date_range(start=time, periods=len(df_flight), freq="83s").astype(np.int64) // 10**9
    df_flight["time"] = pd.to_datetime(df_flight["UTC time"], origin="unix", unit="s") # Convert the UTC time to datetimes and assign to the "time" column

    # Save the updated DataFrame to a new CSV file
    print(f"Saving new flight CSV to {new_flight_csv_path}")
    df_flight.to_csv(new_flight_csv_path, index=False, mode="w")