"""run_CoCiP_batches.py

Run CoCiP for a single synthetic (idealized-met) flight and compare the
resulting contrail against a matching APCEMM/LES test case.

The flight, radiation field, and meteorology are all constructed to be as
close to spatially/temporally uniform as possible, so that the only thing
varying along the contrail is what CoCiP's own physics does to it. Several
pieces of CoCiP's native wake-vortex / ice-formation physics are overridden
below so the *initial* contrail state (ice crystal number and ice mass) is
forced to match the corresponding LES run, letting us isolate and compare
CoCiP's post-formation evolution (spreading, sublimation, radiative
forcing) against APCEMM and LES on equal footing.

Usage
-----
    python run_CoCiP_batches.py <test_id> <time_of_day> <met_from_scratch>

    test_id          e.g. "130T205L25" -- encodes RH (first 3 digits),
                     temperature (next 3 digits) and other test parameters.
    time_of_day      "midnight" or "noon"
    met_from_scratch "True" to build a new synthetic met NetCDF file,
                     "False" to reuse a previously-saved one.
"""

import sys
import os

sys.path.append("/home/chinahg/GCresearch/contrailuncertainty/start_here/")
import pipeline_fxn_lib as lib
import run_CoCiP_fxn_lib as cocip_lib

import numpy as np
import pandas as pd
import xarray as xr

from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.cocip import contrail_properties as cp
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.physics import constants

print("Starting run_CoCiP_batches.py")

# ============================================================================
# Command-line arguments
# ============================================================================
test_id = sys.argv[1]
time_of_day = sys.argv[2]
met_from_scratch = sys.argv[3] == "True"

print(f"Test ID for CoCiP run: {test_id}")
print(f"Time of day for CoCiP run: {time_of_day}")
print(f"Make new met dataset: {met_from_scratch}")

# ============================================================================
# Aircraft geometry / wake-vortex constants
#
# These are fixed for every test case: they describe the (idealized) initial
# contrail cross-section CoCiP is forced to start from, independent of the
# aircraft's actual wingspan or CoCiP's native Holzapfel wake-vortex physics.
# See `CocipCustomDzMax._simulate_wake_vortex_downwash` below for where these
# get applied.
# ============================================================================
wingspan = 47.25          # m, B767 -- informational only; not currently fed
                           # into the model since width/depth are prescribed
                           # directly rather than derived from wingspan.
custom_dz_max = 200        # m, prescribed max wake-vortex downward displacement
CD0 = 0.5                  # Wake-vortex depth scaling factor, Schumann (2012) eq. 14

initial_depth = custom_dz_max * CD0
initial_width = 10          # m, prescribed initial contrail width (same for all runs)
initial_eff_area = np.pi / 4 * initial_depth * initial_width  # m^2, sanity-check value

temperature_test = float(test_id[4:7])   # K, parsed from test_id
altitude_test = 10700.00                 # m, cruise altitude for this test family
post_vortex_altitude = 10654.12          # m, expected post-wake-vortex altitude
                                          # (derived externally from aircraft
                                          # properties + BV stratification;
                                          # same for all tests at 10700 m)
altitude_initial = altitude_test + (altitude_test - post_vortex_altitude)

# ============================================================================
# Engine / fuel constants
# ============================================================================
flight_u = 237       # m/s, true airspeed
efficiency = 0.3     # overall propulsion efficiency
mdot_f = 1.36986     # kg/s, fuel mass flow rate (Lewellen et al. 2014, cruise)
epsilon = 0.622      # ratio of molecular weights, dry air / water vapor
q_fuel = 43e6        # J/kg, specific heat of combustion
ei_h2o = 1.24        # kg/kg, water vapor emissions index
c_pd = 1005          # J/(kg K), specific heat of dry air
c_pv = 1850          # J/(kg K), specific heat of water vapor

fuel_per_m = mdot_f / flight_u   # kg fuel per meter of flight track

# ============================================================================
# Schmidt-Appleman criterion (SAC) and ice activation rate
#
# Used to back out an initial nvPM emissions index (EI_nvpm) for the flight
# CSV that roughly reproduces the LES ice crystal number after activation.
# The *actual* ice crystal number used by CoCiP is force-set later in
# CocipCustomDzMax, so this EI_nvpm is only a physically-motivated starting
# point, not the final word on ice number.
# ============================================================================
air_pressure = lib.alt2press(altitude_initial)  # Pa
RH = float(test_id[0:3]) / 100                  # fraction of RHi, parsed from test_id
q = cocip_lib.estimate_specific_humidity(temperature_test, air_pressure, RH)
c_pm = c_pd * (1 - q) + c_pv * q
G = (ei_h2o * c_pm * air_pressure) / (epsilon * q_fuel * (1 - efficiency))
T_LM = cocip_lib.T_sat_liquid(G)
activation_rate = cocip_lib.ice_particle_activation_rate(temperature_test, T_LM)

# ============================================================================
# LES target values for this test case
#
# n_ice_per_m (N) and kg_ice_per_m are the two values CoCiP's initial contrail
# state is forced to match exactly (see CocipCustomDzMax._create_downwash_contrail).
# r_ice_les is kept only as a point of comparison against whatever effective
# ice-crystal radius falls out of matching N and mass -- it is NOT enforced.
# See the discussion in the accompanying chat: forcing all three of
# {number, mass, radius} independently over-constrains the system, since
# CoCiP only carries n_ice_per_m and iwc as persisted state and derives
# r_ice_vol from them every timestep.
# ============================================================================
LES_df = pd.read_csv(
    f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/processed_data/all_data/{test_id}.csv"
)

number_type = "number per meter"  # "number per meter" or "EInvpm"
f_surv = 1.0                       # fraction of ice crystals surviving wake-vortex
                                   # sinking (monkeypatched to 1.0 everywhere below)

ice_test = LES_df["Ice_number"].iloc[0]
if number_type == "EInvpm":
    N = ice_test * activation_rate * fuel_per_m * f_surv
else:
    N = ice_test  # ice crystals per meter

r_ice_les = LES_df["Effective_radius_um"].iloc[0] * 1e-6  # m, diagnostic only

# Total ice mass per meter of flight track (kg/m)
kg_ice_per_m = LES_df["Ice_mass"].iloc[0]

# ============================================================================
# Paths
# ============================================================================
save_dir = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}"
base_flight_csv_path = "/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/flight-cocip-aligned.csv"
new_flight_csv_path = f"{save_dir}/{test_id}-flight-cocip-aligned.csv"

os.makedirs(save_dir, exist_ok=True)

make_flight_csv = True  # set False to reuse an existing flight CSV
save_data = True

print(f"Running CoCiP for {number_type} = {ice_test} and r_ice (LES, diagnostic) = {r_ice_les} m.")
print(f"Save results: {save_data}.")
print(f"Flight CSV path: {new_flight_csv_path}.")

# ============================================================================
# Flight setup
# ============================================================================
if time_of_day == "midnight":
    desired_latitude, desired_longitude = 45.0, -46.5
    desired_datetime = "2025-06-29 00:00:00"
else:
    desired_latitude, desired_longitude = 45.0, -45.75
    desired_datetime = "2025-06-29 12:00:00"

if make_flight_csv:
    cocip_lib.make_flight_csv(
        altitude_initial, desired_longitude, desired_latitude, desired_datetime,
        base_flight_csv_path, ice_test, new_flight_csv_path, T_LM, mdot_f,
        flight_u, efficiency, f_surv, number_type, temperature_test,
    )

df_flight, attrs = cocip_lib.format_flight_csv(new_flight_csv_path)
fl = Flight(data=df_flight, attrs=attrs)

# time of first waypoint, floored to the hour -- used to fetch ERA5 radiation data
flight_time = pd.to_datetime(fl["time"][0]).floor("h")


# ============================================================================
# Radiation field: uniform clear-sky ERA5 data broadcast over the whole domain
#
# We verified separately that fraction_of_cloud_cover == 0 at all levels for
# both (lat=45.00, lon=-46.50, 2025-06-29 00:00 UTC) and
# (lat=45.00, lon=-45.75, 2025-06-29 12:00 UTC), so a single clear-sky column
# can be safely broadcast to the full lat/lon/time grid CoCiP expects.
# ============================================================================
def build_broadcast_rad_dataset(flight_time, lat, lon, n_periods=40):
    """Fetch ERA5 radiation data at a single point in time/space and
    broadcast it to a uniform lat/lon/time grid for use with CoCiP."""
    era5_single_rad = ERA5(time=flight_time, variables=Cocip.rad_variables).open_metdataset()
    xr_single = era5_single_rad.data
    desired_time = xr_single["time"].values[0]

    longitudes = np.linspace(-180, 179.75, int((180 + 179.75) / 0.25 + 1))
    latitudes = np.linspace(30, 70, int((70 - 30) / 0.25 + 1))
    times = pd.date_range(flight_time, periods=n_periods, freq="1H")

    target_da = xr.DataArray(
        np.empty((len(times), len(latitudes), len(longitudes))),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    xr_rad = xr.Dataset(coords=target_da.coords)

    for var in xr_single.data_vars:
        scalar = xr_single[var].sel(time=desired_time, latitude=lat, longitude=lon)
        xr_rad[var] = scalar.broadcast_like(target_da)

    xr_rad.attrs.update(dataset="ERA5", provider="ECMWF", product="reanalysis")
    return xr_rad


xr_era5_rad = build_broadcast_rad_dataset(flight_time, desired_latitude, desired_longitude)
xr_era5_rad.to_netcdf(f"{save_dir}/{test_id}-rad-{time_of_day}.nc", mode="w", format="NETCDF4")
rad = MetDataset(xr_era5_rad)


# ============================================================================
# Meteorology: synthetic idealized field built from the matching APCEMM
# vertical profile (temperature, RHi), with a prescribed constant-shear wind.
# ============================================================================
def build_synthetic_met_dataset(test_id, desired_datetime, wind_shear=-0.004, ref_wind=17.8):
    """Build an idealized met NetCDF from the APCEMM base-input profile for
    this test case: temperature and RHi come from APCEMM, vertical motion
    and cloud ice are zero everywhere, and a single wind component carries a
    constant vertical shear (the only source of contrail-spreading shear;
    since the flight track heads due east, wind shear must be in the
    cross-track / northward component for CoCiP's spreading physics to see it --
    the eastward / along-track component is left uniform with altitude)."""
    xr_apcemm_met = xr.open_dataset(
        f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/base_inputs/{test_id}/{test_id}.nc"
    )

    longitudes = np.linspace(-180, 179.75, int((180 + 179.75) / 0.25 + 1))
    latitudes = np.linspace(30, 70, int((70 - 30) / 0.25 + 1))
    levels = np.flip(xr_apcemm_met["pressure"].values)
    altitudes = np.flip(xr_apcemm_met["altitude"].values.astype(np.float32)) * 1000  # km -> m
    pressures = levels * 100  # hPa -> Pa

    utc_seconds = pd.date_range(start=desired_datetime, periods=60, freq="1h").astype(np.int64) // 10**9
    times = pd.to_datetime(utc_seconds, origin="unix", unit="s")

    shape = (len(longitudes), len(latitudes), len(levels), len(times))

    air_temperature = np.broadcast_to(
        np.flip(xr_apcemm_met["temperature"][:, 0].values)[np.newaxis, np.newaxis, :, np.newaxis], shape
    ).astype(np.float32)
    relative_humidity = np.broadcast_to(
        np.flip(xr_apcemm_met["relative_humidity_ice"][:, 0].values)[np.newaxis, np.newaxis, :, np.newaxis], shape
    ).astype(np.float32)

    # Uniform along-track (eastward) wind -- no vertical shear, so it does not
    # contribute to cross-sectional plume spreading in CoCiP's shear physics.
    eastward_wind = np.ones_like(relative_humidity, dtype=np.float32) * 25

    # No vertical motion or background cloud ice
    lagrangian_tendency_of_air_pressure = np.zeros_like(relative_humidity, dtype=np.float32)
    specific_cloud_ice_water_content = np.zeros_like(relative_humidity, dtype=np.float32)

    # Specific humidity from RHi + temperature (Sonntag-based saturation vapor pressures)
    T0 = 273.15
    T_base = air_temperature[0, 0, :, 0]
    RHi_base = relative_humidity[0, 0, :, 0]
    pres_base = pressures
    P_sat_w = lib.compute_Psat_w(T_base)
    P_sat_i = lib.compute_Psat_i(T_base)
    q_base = (RHi_base * P_sat_i / P_sat_w) * (1 / (0.263 * pres_base)) * np.exp(
        (17.67 * (T_base - T0)) / (T_base - 29.65)
    )
    specific_humidity = np.broadcast_to(q_base[np.newaxis, np.newaxis, :, np.newaxis], shape).astype(np.float32)

    # Cross-track (northward) wind carries the constant vertical shear that
    # drives contrail width growth. Magnitude reaches 25 m/s at 10.7 km,
    # matching the eastward component there.
    print(f"Using wind shear of {wind_shear} m/s/m for the northward (cross-track) wind component.")
    wind_u = wind_shear * altitudes + ref_wind
    northward_wind = np.broadcast_to(wind_u[np.newaxis, np.newaxis, :, np.newaxis], shape).astype(np.float32)

    ds_met = xr.Dataset(
        {
            "air_temperature": (["longitude", "latitude", "level", "time"], air_temperature),
            "specific_humidity": (["longitude", "latitude", "level", "time"], specific_humidity),
            "eastward_wind": (["longitude", "latitude", "level", "time"], eastward_wind),
            "northward_wind": (["longitude", "latitude", "level", "time"], northward_wind),
            "lagrangian_tendency_of_air_pressure": (["longitude", "latitude", "level", "time"], lagrangian_tendency_of_air_pressure),
            "specific_cloud_ice_water_content": (["longitude", "latitude", "level", "time"], specific_cloud_ice_water_content),
            "relative_humidity": (["longitude", "latitude", "level", "time"], relative_humidity),
        },
        coords={
            "longitude": longitudes,
            "latitude": latitudes,
            "level": levels,
            "time": times,
            "altitude": ("level", altitudes),
            "air_pressure": ("level", pressures),
        },
    )
    return ds_met


met_path = f"{save_dir}/{test_id}-met-{time_of_day}.nc"
if met_from_scratch:
    print("Creating new met dataset from scratch...")
    ds_met = build_synthetic_met_dataset(test_id, desired_datetime)
    ds_met.to_netcdf(met_path, mode="w", format="NETCDF4")
    print(f"Saved new met file to {met_path}.")
else:
    ds_met = xr.open_dataset(met_path)

new_met = MetDataset(ds_met)

# ============================================================================
# Sanity-check printout of initial conditions before running CoCiP
# ============================================================================
print("Initial conditions for CoCiP run:")
print(f"Initial altitude: {altitude_initial} m")
print(f"Initial ice number per meter: {ice_test} {number_type}")
print(f"Initial ice crystal radius (LES, diagnostic only): {r_ice_les} m")
print(f"Target ice mass per meter: {kg_ice_per_m:.3e} kg/m")
print(f"Initial effective area: {initial_eff_area:.2f} m²")
print(f"EI ice: {float(df_flight['nvpm_ei_n'].values[0]):.2e} #/kg")


# ============================================================================
# CoCiP subclass: force-set initial wake-vortex geometry and ice properties
# ============================================================================
class CocipCustomDzMax(Cocip):
    """CoCiP with two of its physics stages overridden so that the initial
    contrail state matches a prescribed geometry and ice content, rather than
    the values CoCiP's own wake-vortex model (Holzapfel descent, Schumann
    activation/survival) would otherwise compute from aircraft/atmosphere
    inputs. This lets CoCiP's *evolution* physics be compared against
    APCEMM/LES starting from an identical initial state.
    """

    def __init__(self, *args, custom_dz_max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_dz_max = custom_dz_max
        print(f"Initialized with custom_dz_max = {self.custom_dz_max}")

    def _simulate_wake_vortex_downwash(self, met):
        """Run CoCiP's native wake-vortex downwash, then overwrite dz_max,
        width, and depth with prescribed values (see module-level constants
        `custom_dz_max`, `initial_width`, `initial_depth`)."""
        print("\n!!! CUSTOM _simulate_wake_vortex_downwash CALLED !!!")
        super()._simulate_wake_vortex_downwash(met)

        if self.custom_dz_max is None:
            return

        print("=== OVERRIDING dz_max / width / depth ===")
        print(f"Before: dz_max={self._sac_flight['dz_max'].mean():.2f}, depth={self._sac_flight['depth'].mean():.2f}")

        n = self._sac_flight.size
        self._sac_flight["dz_max"] = np.full(n, self.custom_dz_max)
        self._sac_flight["width"] = np.full(n, initial_width)
        self._sac_flight["depth"] = np.full(n, initial_depth)

        print(f"After:  dz_max={self._sac_flight['dz_max'].mean():.2f}, depth={self._sac_flight['depth'].mean():.2f}, "
              f"width={self._sac_flight['width'].mean():.2f}")
        print(f"Altitude after downwash: {self._sac_flight['altitude'].mean():.2f}")
        print(f"Fuel flow: {self._sac_flight['fuel_flow'].mean():.2e} kg/s")
        print(f"EInvpm after downwash: {self._sac_flight['nvpm_ei_n'].mean():.2e} #/kg")
        print(f"Implied ice crystals/m after downwash (pre-override): "
              f"{self._sac_flight['nvpm_ei_n'].mean() * activation_rate * fuel_per_m * f_surv:.2e}")

    def _create_downwash_contrail(self):
        """Run CoCiP's native contrail initialization, then force n_ice_per_m
        and iwc to match the LES targets (N, kg_ice_per_m) exactly.

        n_ice_per_m and iwc are the only two ice-related quantities CoCiP
        actually persists as state; r_ice_vol is recomputed from them at
        every timestep (`contrail_properties.ice_particle_volume_mean_radius`),
        so it is left as a free/diagnostic quantity here rather than being
        force-set directly -- overwriting it would just get discarded on the
        next integration step anyway.
        """
        contrail = super()._create_downwash_contrail()

        print("\n!!! CUSTOM _create_downwash_contrail CALLED !!!")
        print(f"Pre-override n_ice_per_m: {contrail['n_ice_per_m'][0]:.3e}")
        print(f"Pre-override iwc: {contrail['iwc'][0]:.3e}")

        print(f"Initial width = {contrail["width"]}")
        print(f"Initial depth = {contrail["depth"]}")

        contrail["width"] = initial_width * np.ones_like(contrail["width"])
        contrail["depth"] = initial_depth * np.ones_like(contrail["depth"])
        area_eff = np.pi/4 * contrail["width"][0] * contrail["depth"][0] # There are multiple waypoints, all have the same initializations, choose first element
        rho_air = contrail["rho_air"][0] # There are multiple waypoints, all have the same initializations, choose first element

        iwc_target = kg_ice_per_m / (area_eff * rho_air)

        contrail["n_ice_per_m"] = N * np.ones_like(contrail["n_ice_per_m"])
        contrail["iwc"] = iwc_target * np.ones_like(contrail["iwc"])
        

        # Diagnostic-only self-check: what effective radius does this
        # (N, mass) pair imply, compared to the LES-reported radius?
        n_per_vol = cp.ice_particle_number_per_volume_of_plume(contrail["n_ice_per_m"], area_eff)
        n_per_kg_air = cp.ice_particle_number_per_mass_of_air(n_per_vol, rho_air)
        r_implied = cp.ice_particle_volume_mean_radius(contrail["iwc"], n_per_kg_air)
        print(f"Target kg_ice_per_m={kg_ice_per_m:.3e} -> iwc set to {iwc_target:.3e}")
        print(f"Implied r_ice_vol from (N, mass): {r_implied[0]:.3e} m "
              f"(LES effective radius for comparison: {r_ice_les:.3e} m)")
        
        n_co = float(contrail["n_ice_per_m"][0])
        m_co = float((4.0 / 3.0) * np.pi * 917 * (r_implied[0] ** 3) * n_co)
        print(f"Target ice mass per m={kg_ice_per_m:.3e}, Calculated ice mass per m={m_co:.3e}")
        print(f"iwc immediately before return: {contrail['iwc'][0]:.3e}")

        return contrail


# Force ice-particle survival fraction to 1.0 everywhere (no loss during wake-
# vortex sinking), consistent with f_surv=1.0 used above when back-calculating
# EI_nvpm. `contrail_properties` is imported as a module (not a bare function),
# so this module-level monkeypatch is picked up correctly wherever CoCiP calls
# `contrail_properties.ice_particle_survival_fraction(...)`.
cp.ice_particle_survival_fraction = cocip_lib.my_ice_particle_survival_fraction

# Per-test humidity scaling adjustment.
# NOTE: "110T225L25" is a special case tuned separately from every other test;
# if you add more tests that need bespoke tuning, consider moving this to a
# small {test_id: rhi_scale} lookup table instead of if/else branches.
rhi_scale = 0.94 if test_id == "110T225L25" else 0.98

params = {
    "process_emissions": False,
    "verbose_outputs": True,
    "humidity_scaling": ConstantHumidityScaling(rhi_adj=rhi_scale),
    "max_age": np.timedelta64(40, "h"),
    "dt_integration": np.timedelta64(1, "m"),
}

# ============================================================================
# Run CoCiP
# ============================================================================
model = CocipCustomDzMax(met=new_met, rad=rad, custom_dz_max=custom_dz_max, params=params)
fl_out = model.eval(source=fl)

waypoint_0 = model.contrail[model.contrail.waypoint == 0]
waypoint_0_formation = waypoint_0.iloc[0]

print(waypoint_0.head())

print("At exact formation (age_hours=0):")
print(f"  width: {waypoint_0_formation['width']}")
print(f"  depth: {waypoint_0_formation['depth']}")
print(f"  area_eff: {waypoint_0_formation['area_eff']}")
print(f"  number of ice crystals per meter: {waypoint_0_formation['n_ice_per_m']:.3e}")
print(f"  r_ice_vol (implied, not enforced): {waypoint_0_formation['r_ice_vol']:.3e}")
print(f"  iwc: {waypoint_0_formation['iwc']:.3e}")

n_co = float(waypoint_0_formation["n_ice_per_m"])
m_co = float((4.0 / 3.0) * np.pi * 917 * (waypoint_0_formation['r_ice_vol'] ** 3) * n_co)
print(f"  Calculated ice mass per m={m_co:.3e}")

# ============================================================================
# Add "true initial values" to dataset
# ============================================================================
# Copy the first row
new_row = waypoint_0.iloc[[0]].copy()

# Edit the new row
new_row["age_hours"] = 0.0
new_row["width"] = initial_width
new_row["depth"] = initial_depth
new_row["area_eff"] = initial_eff_area
new_row["n_ice_per_m"] = N
new_row["iwc"] = kg_ice_per_m / (initial_eff_area * new_row['rho_air'][0])

rho_ice = 917 # kg/m3
r_implied = (kg_ice_per_m / (N * 4/3 * np.pi * rho_ice))**(1/3)
new_row["r_ice_vol"] = r_implied
print(f"Implied r_ice_vol from (N, mass): {r_implied:.3e} m "
      f"(LES effective radius for comparison: {r_ice_les:.3e} m)")


# Insert at the beginning
waypoint_0 = pd.concat([new_row, waypoint_0], ignore_index=True)
waypoint_0["timestep"] = waypoint_0.index

# Recompute age_hours for every row
dt_hours = 1 / 60  # 1 minute
waypoint_0["age_hours"] = waypoint_0.index * dt_hours

print(waypoint_0.head())

# ============================================================================
# Save output
# ============================================================================
if save_data:
    waypoint_ds = waypoint_0.to_xarray()
    df_save_dir = f"{save_dir}/{test_id}-bypass_{time_of_day}.nc"
    waypoint_ds.to_netcdf(df_save_dir)
    print(f"Results saved to {df_save_dir}.")
else:
    print("Results not saved. Set save_data = True to save the data.")