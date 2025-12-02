### libRadtran FUNCTION LIBRARY ###
import subprocess
import os
import numpy as np
import pandas as pd

def updateInput(base_filepath, filepath, attributes):
    """
    Update the input file with the given attributes.

    Parameters:
    - filepath (str): The path of the input file to be updated.
    - attributes (object): An object containing the attributes to be written to the input file.
    - contrail (bool): A flag indicating whether the input file is for contrail simulation or not.

    Returns:
    None
    """
    # Read in the existing file
    with open(base_filepath, "r") as f:
        file = f.readlines()

    # Write each attribute as "Name Value # Description" from the attributes dataframe
    with open(filepath, "w") as f:
        for row in attributes.iterrows():
            name = row[1].Name
            value = str(row[1].Value)
            description = row[1].Description if row[1].Description != "" else "No description provided"
            f.writelines(f"{name} {value}             # {description}\n")

def calculate_SW_Flux(attributes, base_solar_path, solar_path):

    # Run the shortwave (solar) radiative forcing simulation.
    updateInput(base_solar_path, solar_path, attributes)

    cmd = ["/home/iross/misc-code/libRadtran/bin/uvspec"]
    with open(solar_path, 'r') as f:
        SW_output = subprocess.run(
            cmd,
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LD_LIBRARY_PATH": "/data/home/chinahg/.conda/envs/afca-test/lib:" + os.environ.get("LD_LIBRARY_PATH","")},
            text=True,
        )
    print(f"SW error: {SW_output.stderr}")
    print(f"SW output: {SW_output.stdout}")
    SW_output = reformatResults(SW_output.stdout)  # Get the last element which is the net TOA flux
    return SW_output

def calculate_LW_Flux(attributes, base_thermal_path, thermal_path):

    # Run the longwave (thermal) radiative forcing simulation.
    updateInput(base_thermal_path, thermal_path, attributes)

    cmd = ["/home/iross/misc-code/libRadtran/bin/uvspec"]
    with open(thermal_path, 'r') as f:
        LW_output = subprocess.run(
            cmd,
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LD_LIBRARY_PATH": "/data/home/chinahg/.conda/envs/afca-test/lib:" + os.environ.get("LD_LIBRARY_PATH","")},
            text=True,
        )
    print(f"LW error: {LW_output.stderr}")
    print(f"LW output: {LW_output.stdout}")
    LW_output = reformatResults(LW_output.stdout)  # Get the last element which is the net TOA flux
    
    return LW_output

def reformatResults(resultsRaw):
    string = str(resultsRaw.strip().replace("  ", " "))
    li = list(string.split(" "))
    flux = float(li[-2])  # The second last element is the net TOA flux
    return flux

def make_LW_options(habit, hour, ice_in_path):
    LW_contrail_options_BASE = [
        ["data_files_path", "data", "Location of libRadtran data files"],
        ["source", "thermal", "Calculate the longwave radiation"],
        ["latitude", "N 45", "Latitude of the location"],
        ["longitude", "W 45", "Longitude of the location"],
        ["time", f"2025 6 29 {hour} 0 0", "Local time YYYY MM DD hh mm ss"],  # Example times
        ["albedo", "0.06", "Surface albedo over open ocean"],
        ["rte_solver", "disort", "Radiative transfer equation solver"],
        ["mol_abs_param", "reptran", "Provides correlated-k absorption coefficients"],
        ["number_of_streams", "16", "Number of discrete zenith-angle directions DISORT uses"],
        ["wavelength", "OVERWRITE", "Wavelength range [nm]"],
        ["zout", "TOA", "Sum at the top of atmosphere"],
        ["ic_file", f"1D {ice_in_path}", "Ice properties input file"],
        ["ic_habit", "OVERWRITE", "Ice habit"],
        ["output_process", "integrate", "Spectrally integrate the output"],
        ["output_user", "edir eglo edn eup enet esum",
         "Return direct/global/downward/upward irradiance. Net = global - upward."],
        ["quiet", "", "Suppress output dialog"]
    ]

    if habit == "ghm": # Baum 2005a parametrization for the general habit mixture
        LW_contrail_options = LW_contrail_options_BASE.copy()
        for i, option in enumerate(LW_contrail_options):
            if option[0] == "ic_habit":
                LW_contrail_options[i][1] = habit
            if option[0] == "wavelength":
                # Longwave wavelengths are approximately 3000-100000 nm
                # Baum_v36 spans 202-99000 nm but interpolation of the table values requires a buffer, hence 86000 nm
                LW_contrail_options[i][1] = "3000 86000"  

        LW_contrail_options.append(["ic_properties", "baum_v36 interpolate", "Ice properties"])

    elif habit == "solid_column" or habit == "droxtal": # Yang 2013 parameterization for solid_column and droxtal habits
        LW_contrail_options = LW_contrail_options_BASE.copy()

        for i, option in enumerate(LW_contrail_options):
            if option[0] == "ic_habit":
                LW_contrail_options[i][0] = "ic_habit_yang2013"
                LW_contrail_options[i][1] = habit + " moderate"
            if option[0] == "wavelength":
                # Longwave wavelengths are approximately 3000-100000 nm
                # Yang 2013 builds optical properties on the internal grid, so we don’t rely on precomputed table edges
                LW_contrail_options[i][1] = "3000 86000" # Match Baum_v36 range for consistency

        LW_contrail_options.append(["ic_properties", "yang2013", "Ice properties"]) # Use Yang 2013 ice optical properties
        LW_contrail_options.append(["ic_properties", "yang interpolate", "Ice properties"]) # Tell libRadtran to interpolate the Yang 2013 data even though it's expensive

    else:
        raise ValueError("Habit must be either 'ghm', 'solid_column', or 'droxtal'.")

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
        # Bulk of shortwave solar radiation at the TOA is between 200-800 nm, higher than 800 nm and the atmosphere 
        # absorbs more strongly due to water vapor and less light reaches the cloud
        # 202 is chosen specifically because of lower wavelength limit of Baum_v36
        "202 800", "Wavelength range [nm]" 
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

def updateIceIn(base_ice_in_file, new_ice_in_file, depth, IWC, radii, min_radius):
    with open(base_ice_in_file, "r") as f:
        ice_in_content = f.readlines()

    radius = max(radii, min_radius)
    contrail_depth = 10.7 - depth / 1000  # Convert m → km
    contrail_IWC = IWC  # [g/m^3]

    ice_in_content[2] = f"{contrail_depth:.3f}    {contrail_IWC:.16e}     {radius:.3f}\n"
    with open(new_ice_in_file, "w") as f:
        f.writelines(ice_in_content)

    print(f"Updated ice.in: depth={contrail_depth:.3f} km, radius={radius:.3f} µm, IWC={contrail_IWC:.16e} g/m³")

def calculate_RF(times, IWC, depth, radii, habit, hour, ice_in_path, thermal_cloud_path, thermal_clear_path, solar_cloud_path, solar_clear_path):
    print(f"Calculating LW and SW RF for dataset with habit: {habit}...")

    LW_RF = np.zeros(len(times))
    SW_RF = np.zeros(len(times))
    LW_RF_CONTRAIL = np.zeros(len(times))
    LW_RF_CLEAR = np.zeros(len(times))
    SW_RF_CONTRAIL = np.zeros(len(times))
    SW_RF_CLEAR = np.zeros(len(times))

    if habit == "solid_column":
        min_radius = 5.961  # Minimum radius for solid-column [um]
    elif habit == "ghm":
        min_radius = 5.01  # Minimum radius for ghm [um]
    elif habit == "droxtal":
        min_radius = 9.481  # Minimum radius for droxtal [um]
    elif habit == "spheroid":
        min_radius = 6.581  # Minimum radius for spheroidal [um]
    else:
        raise ValueError(f"Habit must be either 'ghm', 'solid_column', 'droxtal', or 'spheroid'. Habit provided: {habit}")

    for i in range(len(times)):
        # Update the effective radius, IWC, and contrail depth in ice.in
        if i == 0: # For the first time, create new ice.in, thermal_clear.in, thermal-cloud.in, solar-clear.in, solar-cloud.in files
            base_ice_in_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/ice.in"
            base_thermal_cloud_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/thermal-cloud.in"
            base_thermal_clear_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/thermal-clear.in"
            base_solar_cloud_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/solar-cloud.in"
            base_solar_clear_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/solar-clear.in"
        else:
            base_ice_in_path = ice_in_path  # Use the previously created ice.in file
            base_thermal_cloud_path = thermal_cloud_path
            base_thermal_clear_path = thermal_clear_path
            base_solar_cloud_path = solar_cloud_path
            base_solar_clear_path = solar_clear_path

        updateIceIn(base_ice_in_path, ice_in_path, depth[i], IWC[i], radii[i], min_radius) # Makes a new ice.in file or updates the existing copy
        
        # Define LW and SW input configurations
        LW_contrail_options, LW_clearsky_options = make_LW_options(habit, hour, ice_in_path)
        SW_contrail_options, SW_clearsky_options = make_SW_options(LW_contrail_options)

        LW_RF_CONTRAIL[i] = calculate_LW_Flux(LW_contrail_options, base_thermal_cloud_path, thermal_cloud_path)
        LW_RF_CLEAR[i] = calculate_LW_Flux(LW_clearsky_options, base_thermal_clear_path, thermal_clear_path)
        SW_RF_CONTRAIL[i] = calculate_SW_Flux(SW_contrail_options, base_solar_cloud_path, solar_cloud_path)
        SW_RF_CLEAR[i] = calculate_SW_Flux(SW_clearsky_options, base_solar_clear_path, solar_clear_path)
        LW_RF[i] = np.abs(np.abs(LW_RF_CONTRAIL[i]) - np.abs(LW_RF_CLEAR[i]))
        SW_RF[i] = np.abs(np.abs(SW_RF_CONTRAIL[i]) - np.abs(SW_RF_CLEAR[i]))

    LW_RF_df = pd.DataFrame({"Contrail_Age_hours": times, "LW_Radiative_Forcing_W_m2": LW_RF, "LW_RF_CLEAR_W_m2": LW_RF_CLEAR, "LW_RF_CONTRAIL_W_m2": LW_RF_CONTRAIL})
    SW_RF_df = pd.DataFrame({"Contrail_Age_hours": times, "SW_Radiative_Forcing_W_m2": SW_RF, "SW_RF_CLEAR_W_m2": SW_RF_CLEAR, "SW_RF_CONTRAIL_W_m2": SW_RF_CONTRAIL})
    return LW_RF_df, SW_RF_df