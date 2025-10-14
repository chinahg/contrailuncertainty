#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# For thesis stuff (not necessary for LRT calculations)
import xarray as xr
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as pipelinelib
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/LRT/')
import LRT_fxnlib as LRTlib


### Function Library ###
def make_LW_options(habit, hour):
    LW_contrail_options_BASE = [
        ["data_files_path", "data", "Location of libRadtran data files"],
        ["source", "thermal", "Calculate the longwave radiation"],
        ["latitude", "N 45", "Latitude of the location"],
        ["longitude", "W 45", "Longitude of the location"],
        ["time", f"2025 6 29 {hour} 0 0", "Local time YYYY MM DD hh mm ss"],  # Example times
        ["albedo", "0.2", "Surface albedo"],
        ["rte_solver", "disort", "Radiative transfer equation solver"],
        ["mol_abs_param", "reptran", "Fine structure parameter"],
        ["number_of_streams", "6", "Number of streams"],
        ["wavelength", "2500 80000", "Wavelength range [nm]"],
        ["zout", "TOA", "Sum at the top of atmosphere"],
        ["ic_file", "1D ice.in", "Ice properties input file"],
        ["ic_properties", "yang", "Ice properties"],
        ["ic_habit", "OVERWRITE", "Ice habit"],
        ["output_process", "integrate", "Spectrally integrate the output"],
        ["output_user", "edir eglo edn eup enet esum",
         "Return direct/global/downward/upward irradiance. Net = global - upward."],
        ["quiet", "", ""]
    ]

    if habit == "ghm":
        LW_contrail_options = LW_contrail_options_BASE.copy()
        for i, option in enumerate(LW_contrail_options):
            if option[0] == "ic_habit":
                LW_contrail_options[i][1] = habit
            if option[0] == "ic_properties":
                LW_contrail_options[i][0] = "ic_properties"
                LW_contrail_options[i][1] = "baum_v36 interpolate"
                LW_contrail_options[i][2] = "Ice properties"

    elif habit == "solid-column":
        LW_contrail_options = LW_contrail_options_BASE.copy()
        for i, option in enumerate(LW_contrail_options):
            if option[0] == "ic_habit":
                LW_contrail_options[i][1] = habit

    else:
        raise ValueError("Habit must be either 'ghm' or 'solid-column'.")

    LW_contrail_options = pd.DataFrame(LW_contrail_options, columns=["Name", "Value", "Description"])
    LW_clearsky_options = LW_contrail_options[~LW_contrail_options["Name"].str.startswith("ic_")].reset_index(drop=True)

    return LW_contrail_options, LW_clearsky_options


def make_SW_options(LW_contrail_options):
    # SW is the same as LW except for the source and wavelength
    SW_contrail_options = LW_contrail_options.copy()
    SW_contrail_options.loc[SW_contrail_options["Name"] == "source", ["Value", "Description"]] = [
        "solar data/solar_flux/atlas_plus_modtran",
        "Calculate the shortwave radiation, location of the extraterrestrial spectrum"
    ]
    SW_contrail_options.loc[SW_contrail_options["Name"] == "wavelength", ["Value", "Description"]] = [
        "299 341", "Wavelength range [nm]"
    ]

    SW_clearsky_options = SW_contrail_options[~SW_contrail_options["Name"].str.startswith("ic_")].reset_index(drop=True)
    return SW_contrail_options, SW_clearsky_options


def calculate_LLES_tau(S, W):
    """Calculate vertically integrated optical depth."""
    tau = 0.5 * S / W
    return tau


def calculate_sauter_mean(particle_sizes, frequencies):
    """
    Calculates the Sauter Mean Diameter (D[3,2]) for a particle size distribution.
    """
    particle_sizes = np.array(particle_sizes)
    frequencies = np.array(frequencies)
    third_moment = np.sum(particle_sizes ** 3 * frequencies)
    second_moment = np.sum(particle_sizes ** 2 * frequencies)
    sauter_mean = third_moment / second_moment
    return sauter_mean


def calculate_RF(times, IWC, depth, radii, habit, hour):
    print(f"Calculating LW and SW RF for dataset with habit: {habit}...")

    LW_RF = np.zeros(len(times))
    SW_RF = np.zeros(len(times))

    if habit == "solid-column":
        min_radius = 5.96  # Minimum radius for solid-column [um]
    elif habit == "ghm":
        min_radius = 5.1  # Minimum radius for ghm [um]

    for i in range(len(times)):
        # Define LW and SW input configurations
        LW_contrail_options, LW_clearsky_options = make_LW_options(habit, hour)
        SW_contrail_options, SW_clearsky_options = make_SW_options(LW_contrail_options)

        # Update the effective radius, IWC, and contrail depth in ice.in
        ice_in_file = "/home/chinahg/GCresearch/contrailuncertainty/LRT/ice.in"
        with open(ice_in_file, "r") as f:
            ice_in_content = f.readlines()

        radius = max(radii[i], min_radius)
        contrail_depth = 10.7 - depth[i] / 1000  # Convert m → km
        contrail_IWC = IWC[i]  # [g/m^3]

        ice_in_content[3] = f"{contrail_depth:.3f} {contrail_IWC:.3f} {radius:.3f}\n"
        with open(ice_in_file, "w") as f:
            f.writelines(ice_in_content)

        print(f"Updated ice.in: depth={contrail_depth:.3f} km, radius={radius:.3f} µm, IWC={contrail_IWC:.3e} g/m³")

        LW_RF[i] = np.abs(LRTlib.calculate_LW_Flux(LW_contrail_options, True)
                          - LRTlib.calculate_LW_Flux(LW_clearsky_options, False))
        SW_RF[i] = np.abs(LRTlib.calculate_SW_Flux(SW_contrail_options, True)
                          - LRTlib.calculate_SW_Flux(SW_clearsky_options, False))

    LW_RF_df = pd.DataFrame({"Contrail_Age_hours": times, "LW_Radiative_Forcing_W_m2": LW_RF})
    SW_RF_df = pd.DataFrame({"Contrail_Age_hours": times, "SW_Radiative_Forcing_W_m2": SW_RF})
    return LW_RF_df, SW_RF_df


#######################################################################################################################
# Main script
#######################################################################################################################
test_id = '110T218L25'
model_type = ['CoCiP']  # 'CoCiP', 'APCEMM', 'LLES'
print(f"Running LRT calculations for: {test_id}")

### GHM ###
habit_type = "ghm"
hours = ["0", "12"]


if "CoCiP" in model_type:
    print("Processing CoCiP data...")
    ### COCIP DATASET ###
    cocip_ds = xr.open_dataset(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/{test_id}-bypass.nc")
    cocip_tau = cocip_ds['tau_contrail'].values
    cocip_times = cocip_ds['age_hours'].values
    cocip_radii = cocip_ds['r_ice_vol'].values * 1e6  # [µm]
    cocip_depth = cocip_ds['depth'].values  # [m]
    cocip_IWC = cocip_ds['iwc'].values * cocip_ds['rho_air'] *1e3 # [kg ice/ kg air] * [kg air/m3] * 1e3 # Convert to [g/m^3]

    # LW and SW for CoCiP
    for hour in hours:
        LW_RF_cocip, SW_RF_cocip = calculate_RF(cocip_times, cocip_IWC, cocip_depth, cocip_radii, habit_type, hour)
        SW_RF_cocip.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/SW_RF_cocip_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"SW RF results saved to .../SW_RF_cocip_{habit_type}_{hour}h_IWC.csv")

        LW_RF_cocip.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/LW_RF_cocip_{habit_type}_{hour}h_IWC.csv",
            index=False
        )
        print(f"LW RF results saved to .../LW_RF_cocip_{habit_type}_{hour}h_IWC.csv")

if "APCEMM" in model_type:
    ### APCEMM DATASET ###
    APCEMM_data = pipelinelib.read_apcemm_data(
        f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/outputs'
    )
    APCEMM_ds = APCEMM_data.ds_t
    n_bins = 38

    APCEMM_intOD = np.zeros(len(APCEMM_ds))
    APCEMM_widths = np.zeros(len(APCEMM_ds))
    APCEMM_depth = np.zeros(len(APCEMM_ds))
    APCEMM_radii_bins = APCEMM_ds[0]['r'].values * 1e6  # [µm]
    APCEMM_radii_counts = np.zeros((n_bins, len(APCEMM_ds)))
    APCEMM_IWC_avg = np.zeros(len(APCEMM_ds))

    for i in range(len(APCEMM_ds)):
        width = APCEMM_ds[i]['width'].values.item()
        if i > 0 and width == 0:
            width = APCEMM_widths[i - 1]
        APCEMM_widths[i] = width

        APCEMM_radii_counts[:, i] = APCEMM_ds[i]["Overall size distribution"].values
        APCEMM_depth[i] = APCEMM_ds[i]['depth'].values.item()

        val = APCEMM_ds[i]["intOD"].values.item()
        if val > 1200 and 0 < i < len(APCEMM_ds) - 1:
            prev_val = APCEMM_ds[i - 1]["intOD"].values.item()
            next_val = APCEMM_ds[i + 1]["intOD"].values.item()
            val = (prev_val + next_val) / 2
        APCEMM_intOD[i] = val

        APCEMM_IWC = APCEMM_ds[i]['IWC'].values * 1e3  # [kg/m³] → [g/m³]
        APCEMM_IWC_masked = np.where(APCEMM_IWC > 0.3e-5, APCEMM_IWC, np.nan)
        APCEMM_IWC_avg[i] = np.nanmean(APCEMM_IWC_masked)

    APCEMM_tau = np.array([intOD / width for intOD, width in zip(APCEMM_intOD, APCEMM_widths)])
    APCEMM_times = np.arange(len(APCEMM_tau)) * 10 / 60  # hours (10 min/step)
    # LW and SW for APCEMM
    APCEMM_radii = np.zeros(len(APCEMM_times))
    for p in range(len(APCEMM_times)):
        APCEMM_radii[p] = calculate_sauter_mean(APCEMM_radii_bins, APCEMM_radii_counts[:, p])

    for hour in hours:
        LW_RF_APCEMM, SW_RF_APCEMM = calculate_RF(APCEMM_times, APCEMM_IWC_avg, APCEMM_depth, APCEMM_radii, habit_type, hour)
        SW_RF_APCEMM.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/SW_RF_APCEMM_{habit_type}_{hour}h_IWC.csv",
            index=False
        )
        print(f"SW RF results saved to .../SW_RF_APCEMM_{habit_type}_{hour}h_IWC.csv")

        LW_RF_APCEMM.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/LW_RF_APCEMM_{habit_type}_{hour}h_IWC.csv",
            index=False
        )
        print(f"LW RF results saved to .../LW_RF_APCEMM_{habit_type}_{hour}h_IWC.csv")
