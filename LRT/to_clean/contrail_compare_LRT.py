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
        ["time", f"2025 6 29 {hour} 0 0", "Local time YYYY MM DD hh mm ss"], # Noon at 45N, 45W on June 29, 2025 : "2025 6 29 12 0 0", # 3 PM at 45N, 45W on June 29, 2025 : "2025 6 29 15 0 0", Midnight at 45N, 45W on June 29, 2025 : "2025 6 29 0 0 0"
        ["albedo", "0.2", "Surface albedo"],
        ["rte_solver", "disort", "Radiative transfer equation solver"],
        ["mol_abs_param", "reptran", "Fine structure parameter"],
        ["number_of_streams", "6", "Number of streams"],
        ["wavelength", "2500 80000", "Wavelength range [nm]"],
        ["zout", "TOA", "Sum at the top of atmosphere"],
        ["ic_file", "1D ice.in", "Ice properties input file"],
        ["ic_properties", "yang", "Ice properties"],
        ["ic_habit", "OVERWRITE", "Ice habit"],
        ["ic_modify", "tau set OVERWRITE", "Set optical depth"],
        ["output_process", "integrate", "Spectrally integrate the output"],
        ["output_user", "edir eglo edn eup enet esum", "Return the direct, global, diffuse downward, and diffuse upward irradiance. Net is global - upward, sum is global + upward."],
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
    # Define input options: [Name, Value, Description]
    # SW is the same as LW except for the source and wavelength
    SW_contrail_options = LW_contrail_options.copy()
    SW_contrail_options.loc[SW_contrail_options["Name"] == "source", ["Value", "Description"]] = ["solar data/solar_flux/atlas_plus_modtran", "Calculate the shortwave radiation, Location of the extraterrestrial spectrum"]
    SW_contrail_options.loc[SW_contrail_options["Name"] == "wavelength", ["Value", "Description"]] = ["299 341", "Wavelength range [nm]"]

    # Clearsky is the same as contrail except for the ice cloud properties
    SW_clearsky_options = SW_contrail_options[~SW_contrail_options["Name"].str.startswith("ic_")].reset_index(drop=True)

    return SW_contrail_options, SW_clearsky_options

# Calculate the vertical integrated optical depth for each time step with an associated contrail width
def calculate_LLES_tau(S, W):
    tau = 0.5*S/W
    return tau

def get_weights(r):
    # Convert to micrometers from meters
    r = r * 1e6

    if r < 5:
        weights = habit_weights[0]
    elif 5 <= r < 9.5:
        weights = habit_weights[1]
    elif 9.5 <= r < 23:
        weights = habit_weights[2]
    elif 23 <= r < 190:
        weights = habit_weights[3]
    else: # r >= 190
        weights = habit_weights[4]
    return np.array(weights)

def calculate_sauter_mean(particle_sizes, frequencies):
    """
    Calculates the Sauter Mean Diameter (D[3,2]) for a particle size distribution.

    Args:
        particle_sizes (list or numpy array): A list or array of particle sizes.
        frequencies (list or numpy array): A list or array of corresponding frequencies
                                           or probabilities for each particle size.

    Returns:
        float: The Sauter Mean Diameter (D[3,2]).
    """
    # Ensure inputs are numpy arrays for easier element-wise operations
    particle_sizes = np.array(particle_sizes)
    frequencies = np.array(frequencies)

    # Calculate the third moment (sum of (size^3 * frequency))
    third_moment = np.sum(particle_sizes**3 * frequencies)

    # Calculate the second moment (sum of (size^2 * frequency))
    second_moment = np.sum(particle_sizes**2 * frequencies)

    # Calculate the Sauter Mean Diameter (ratio of third moment to second moment)
    sauter_mean = third_moment / second_moment

    return sauter_mean

def calculate_RF(times, tau, depth, radii, habit, hour):
    print(f"Calculating LW and SW RF for CoCiP dataset with habit: {habit}...")

    LW_RF = np.zeros(len(times))
    SW_RF = np.zeros(len(times))

    if habit == "solid-column":
        min_radius = 5.96 # Minimum radius for solid-column to comply with mechanism minimum [um]
    elif habit == "ghm":
        min_radius = 5.1 # Minimum radius for ghm to comply with mechanism minimum [um]

    for i in range(len(times)):
        # Define LW inputs
        LW_contrail_options_cocip, LW_clearsky_options_cocip = make_LW_options(habit, hour)
        # Modify the tau in the LW options
        LW_contrail_options_cocip.loc[LW_contrail_options_cocip["Name"] == "ic_modify", "Value"] = f"tau set {tau[i]}"
    
        # Define SW inputs
        SW_contrail_options_cocip, SW_clearsky_options_cocip = make_SW_options(LW_contrail_options_cocip)
        # Modify the tau in the SW options
        SW_contrail_options_cocip.loc[SW_contrail_options_cocip["Name"] == "ic_modify", "Value"] = f"tau set {tau[i]}"
    
        # Update the effective radius and contrail depth in the ice.in file
        ice_in_file = "/home/chinahg/GCresearch/contrailuncertainty/LRT/ice.in"
        with open(ice_in_file, "r") as f:
            ice_in_content = f.readlines()
        
        # Update the effective radius line (4th line, index 3)
        # Ensure minimum radius of 5.96 um for solid-column to comply with mechanism minimum
        if radii[i] < min_radius:
            radius = min_radius
        else:
            radius = radii[i]

        # Update the contrail depth line (4th line, index 3)
        contrail_depth = 10.7 - depth[i]/1000 # Convert from [m] to [km]

        ice_in_content[3] = f"{contrail_depth:.3f} " + ice_in_content[3].split()[1] + f" {radius:.3f}"
        with open(ice_in_file, "w") as f:
            f.writelines(ice_in_content)
        print(f"Updated ice.in file with contrail depth: {contrail_depth:.3f} [m]")
        print(f"Updated ice.in file with effective radius: {radius:.3f} [um]")

        # LW flux calculation
        LW_RF[i] = np.abs(LRTlib.calculate_LW_Flux(LW_contrail_options_cocip, True) - LRTlib.calculate_LW_Flux(LW_clearsky_options_cocip, False))
        # SW flux calculation
        SW_RF[i] = np.abs(LRTlib.calculate_SW_Flux(SW_contrail_options_cocip, True) - LRTlib.calculate_SW_Flux(SW_clearsky_options_cocip, False))

        # Save the times and LW radiation forcing values to a csv
        LW_RF_df = pd.DataFrame({
            "Contrail_Age_hours": times,
            "LW_Radiative_Forcing_W_m2": LW_RF
        })

        # Save the times and SW radiation forcing values to a csv
        SW_RF_df = pd.DataFrame({
            "Contrail_Age_hours": times,
            "SW_Radiative_Forcing_W_m2": SW_RF
        })

    return LW_RF_df, SW_RF_df


#######################################################################################################################
# Main script
#######################################################################################################################
test_id = '110T218L25'
model_type = ['CoCiP', 'APCEMM', 'LLES'] # 'CoCiP', 'APCEMM', 'LLES'
print(f"Running LRT calculations for: {test_id}")

### COCIP BYPASS DATASET ###
# Open each CoCiP bypass dataset
cocip_ds = xr.open_dataset(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/{test_id}-bypass.nc")
cocip_tau = cocip_ds['tau_contrail'].values
cocip_times = cocip_ds['age_hours'].values # Time in hours
cocip_radii = cocip_ds['r_ice_vol'].values*1e6 # [um]
cocip_depth = cocip_ds['depth'].values # [m]

print("max cocip radius [um]:", np.max(cocip_radii))

### APCEMM BYPASS DATASET ###
APCEMM_data = pipelinelib.read_apcemm_data(
    f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/outputs')
APCEMM_ds = APCEMM_data.ds_t
n_bins = 38

# Extract integrated optical depth and width, handle erroneous values
APCEMM_intOD = np.zeros(len(APCEMM_ds))
APCEMM_widths = np.zeros(len(APCEMM_ds)) # [m]
APCEMM_depth = np.zeros(len(APCEMM_ds)) # [m]
APCEMM_radii_bins = APCEMM_ds[0]['r'].values*1e6 # [um]
APCEMM_radii_counts = np.zeros((n_bins, len(APCEMM_ds)))

for i in range(len(APCEMM_ds)):
    width = APCEMM_ds[i]['width'].values.item()
    if i > 0 and width == 0:
        width = APCEMM_widths[i-1]
    APCEMM_widths[i] = width

    APCEMM_radii_counts[:, i] = APCEMM_ds[i]["Overall size distribution"].values
    APCEMM_depth[i] = APCEMM_ds[i]['depth'].values.item()

    val = APCEMM_ds[i]["intOD"].values.item()
    if val > 1200 and 0 < i < len(APCEMM_ds) - 1:
        prev_val = APCEMM_ds[i - 1]["intOD"].values.item()
        next_val = APCEMM_ds[i + 1]["intOD"].values.item()
        val = (prev_val + next_val) / 2
    APCEMM_intOD[i] = val

# Calculate tau and times
APCEMM_tau = np.array([intOD / width for intOD, width in zip(APCEMM_intOD, APCEMM_widths)])
APCEMM_times = np.arange(len(APCEMM_tau)) * 10 / 60  # time in hours (10 min per step)

### LES DATASET ###
# Import surface areas from csv
ice_surface_area = pd.read_csv(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/ice_surface_area/{test_id}_S.csv", header=None) # time [hr], surface area [m^2/m^2]

# Import contrail depth from digitized LLES data
LLES_depths = pd.read_csv(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/contrail_depth/{test_id}_D.csv", header=None) # time [hr], depth [m]

# Import contrail width from digitized LLES data
LLES_widths = pd.read_csv(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/contrail_width/{test_id}_W.csv", header=None) # time [hr], width [m]

LLES_times = ice_surface_area[0]

# Fit the data points to a smoothed curve
LLES_width_coeffs = np.polyfit(LLES_widths[0], LLES_widths[1], deg=8)
LLES_width_polynomial = np.poly1d(LLES_width_coeffs)
LLES_widths_fitted = LLES_width_polynomial(LLES_times)

LLES_depth_coeffs = np.polyfit(LLES_depths[0], LLES_depths[1], deg=4)
LLES_depth_polynomial = np.poly1d(LLES_depth_coeffs)
LLES_depths_fitted = LLES_depth_polynomial(LLES_times)

# LLES tau calculation, from Lewellen 2014 part 1
LLES_tau = calculate_LLES_tau(ice_surface_area[1], LLES_widths_fitted)

# Import radius distribution data and extract bins and counts
# Radius is reported in micrometers [um]
PDF_dir = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/ice_crystal_radius/110T205L25/"
PDF_paths = sorted([os.path.join(PDF_dir, fname) for fname in os.listdir(PDF_dir) if fname.endswith('.csv')])
LLES_PDFs = []
for k in range(len(PDF_paths)):
    LLES_PDFs.append(pd.read_csv(PDF_paths[k], header=None))

### GHM and solid-column ###
### LW and SW CALCULATION FOR COCIP DATASET ###
for hour in ["12", "18"]: # Midnight, noon, 3 PM, 6 PM. The date is June 29, 2025, the sun set at 8:25pm EDT, "15",
    for habit_type in ["ghm","solid-column"]:
        LW_RF_cocip, SW_RF_cocip = calculate_RF(cocip_times, cocip_tau, cocip_depth, cocip_radii, habit_type, hour)
        
        # Save to csv
        SW_RF_cocip.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/SW_RF_cocip_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"SW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/SW_RF_cocip_{habit_type}_{hour}h.csv.")

        # Save to csv
        LW_RF_cocip.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/LW_RF_cocip_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"LW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/LW_RF_cocip_{habit_type}_{hour}h.csv.")

### LW and SW CALCULATION FOR APCEMM DATASET ###

# Calculate the effective radius using the radio of the 2nd and 3rd moment of the radius distribution
APCEMM_radii = np.zeros(len(APCEMM_times))

for p in range(len(APCEMM_times)):
    APCEMM_radii[p] = calculate_sauter_mean(APCEMM_radii_bins, APCEMM_radii_counts[:, p]) # in [um]

for hour in ["0", "12", "15", "18"]: # Midnight, noon, 3 PM, 6 PM. The date is June 29, 2025, the sun set at 8:25pm EDT, "15",
    for habit_type in ["ghm", "solid-column"]:
        LW_RF_APCEMM, SW_RF_APCEMM = calculate_RF(APCEMM_times, APCEMM_tau, APCEMM_depth, APCEMM_radii, habit_type, hour)

        # Save to csv
        SW_RF_APCEMM.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/SW_RF_APCEMM_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"SW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/SW_RF_APCEMM_{habit_type}_{hour}h.csv.")

        # Save to csv
        LW_RF_APCEMM.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/LW_RF_APCEMM_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"LW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/LW_RF_APCEMM_{habit_type}_{hour}h.csv.")

### LW and SW CALCULATION FOR LLES DATASET ###
# Calculate the effective radius using the radio of the 2nd and 3rd moment of the radius distribution
LLES_radii = np.zeros(len(LLES_times))
idx_12 = np.argmin(np.abs(LLES_times - 12)) + 1 # truncate data at 12 hours due to limited information on ice crystal size distribution

# Define the times (in minutes) corresponding to each PDF file
pdf_minutes = [5, 15, 30, 60, 90, 180, 360, 540, 720]

for p in range(len(LLES_times)):
    # Find the index of the closest PDF time to the current LLES_times[p] * 60
    current_min = LLES_times[p] * 60
    closest_idx = np.argmin(np.abs(np.array(pdf_minutes) - current_min))
    LLES_radii_bins = LLES_PDFs[closest_idx][0].values # [um]
    LLES_radii_counts = LLES_PDFs[closest_idx][1].values
    LLES_radii[p] = calculate_sauter_mean(LLES_radii_bins, LLES_radii_counts) # in [um]

for hour in ["12","18"]: # Midnight, noon, 3 PM, 6 PM. The date is June 29, 2025, the sun set at 8:25pm EDT, "15",    
    for habit_type in ["ghm", "solid-column"]:
        LW_RF_LLES, SW_RF_LLES = calculate_RF(LLES_times[:idx_12], LLES_tau, LLES_depths_fitted, LLES_radii, habit_type, hour)

        # Save to csv
        SW_RF_LLES.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF/SW_RF_LLES_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"SW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF/SW_RF_LLES_{habit_type}_{hour}h.csv.")

        LW_RF_LLES.to_csv(
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF/LW_RF_LLES_{habit_type}_{hour}h.csv",
            index=False
        )
        print(f"LW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF/LW_RF_LLES_{habit_type}_{hour}h.csv.")
        print(f"SW RF results saved to /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF/SW_RF_LLES_{habit_type}_{hour}h.csv.")