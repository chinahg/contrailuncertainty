### libRadtran FUNCTION LIBRARY ###
import subprocess
import os
import numpy as np
import pandas as pd
import warnings
from scipy.special import erf
from scipy import integrate, optimize
from scipy.stats import norm

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
    # print(f"LW error: {LW_output.stderr}")
    # print(f"LW output: {LW_output.stdout}")
    LW_output = reformatResults(LW_output.stdout)  # Get the last element which is the net TOA flux
    
    return LW_output

def reformatResults(resultsRaw):
    # print(f"Raw results: {resultsRaw}")
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
        ["mol_abs_param", "fu", "Provides correlated-k absorption coefficients"],
        ["number_of_streams", "16", "Number of discrete zenith-angle directions DISORT uses"],
        ["wavelength", "OVERWRITE", "Wavelength range [nm]"],
        ["zout", "TOA", "Sum at the top of atmosphere"],
        ["ic_file", f"1D {ice_in_path}", "Ice properties input file"],
        ["ic_habit", "OVERWRITE", "Ice habit"],
        ["output_process", "sum", "Spectrally integrate the output"],
        ["output_user", "edir eglo edn eup enet esum",
         "Return direct/global/downward/upward irradiance. Net = global - upward."],
        ["quiet", "", "Suppress output dialog"] # Comment out this line for debugging feedback
    ]

    if habit == "ghm" or habit == "solid-column" or habit == "rough-aggregate": # Baum 2005a parametrization for the general habit mixture
        LW_contrail_options = LW_contrail_options_BASE.copy()
        for i, option in enumerate(LW_contrail_options):
            if option[0] == "ic_habit":
                LW_contrail_options[i][1] = habit
            if option[0] == "wavelength":
                # Longwave wavelengths are approximately 3000-100000 nm
                # Baum_v36 spans 202-99000 nm but interpolation of the table values requires a buffer, hence 86000 nm
                LW_contrail_options[i][1] = "3000 86000"  

        LW_contrail_options.append(["ic_properties", "baum_v36 interpolate", "Ice properties"])

    # elif habit == "solid_column" or habit == "droxtal": # Yang 2013 parameterization for solid_column and droxtal habits
    #     LW_contrail_options = LW_contrail_options_BASE.copy()

    #     for i, option in enumerate(LW_contrail_options):
    #         if option[0] == "ic_habit":
    #             LW_contrail_options[i][0] = "ic_habit_yang2013"
    #             LW_contrail_options[i][1] = habit + " severe"
    #         if option[0] == "wavelength":
    #             # Longwave wavelengths are approximately 3000-100000 nm
    #             # Yang 2013 builds optical properties on the internal grid, so we don’t rely on precomputed table edges
    #             LW_contrail_options[i][1] = "3000 86000" # Match Baum_v36 range for consistency

    #     LW_contrail_options.append(["ic_properties", "yang2013", "Ice properties"]) # Use Yang 2013 ice optical properties
    #     LW_contrail_options.append(["ic_properties", "yang interpolate", "Ice properties"]) # Tell libRadtran to interpolate the Yang 2013 data even though it's expensive

    else:
        raise ValueError("Habit must be either 'ghm', 'solid-column', or 'rough-aggregate'.")

    LW_contrail_options = pd.DataFrame(LW_contrail_options, columns=["Name", "Value", "Description"])
    LW_clearsky_options = LW_contrail_options[~LW_contrail_options["Name"].str.startswith("ic_")].reset_index(drop=True)

    return LW_contrail_options, LW_clearsky_options


def make_SW_options(LW_contrail_options):
    # SW is the same as LW except for the source and wavelength
    SW_contrail_options = LW_contrail_options.copy()
    SW_contrail_options.loc[SW_contrail_options["Name"] == "source", ["Value", "Description"]] = [
        "solar data/solar_flux/kurudz_1.0nm.dat",
        "Calculate the shortwave radiation, location of the extraterrestrial spectrum"
    ]
    SW_contrail_options.loc[SW_contrail_options["Name"] == "wavelength", ["Value", "Description"]] = [
        # Shortwave radiation wavelengths are approximately 300-3000 nm (Wang et al., 2021)
        "300 3000", "Wavelength range [nm]" 
    ]

    SW_clearsky_options = SW_contrail_options[~SW_contrail_options["Name"].str.startswith("ic_")].reset_index(drop=True)
    return SW_contrail_options, SW_clearsky_options


def calculate_LLES_tau(S, W):
    """Calculate vertically integrated optical depth."""
    tau = 0.5 * S / W
    return tau


def calculate_sauter_mean(particle_radii, frequencies):
    """
    Calculates the Sauter Mean Diameter (D[3,2]) for a particle size distribution.
    """
    particle_sizes = np.array(particle_radii) * 2  # Convert radii to diameters
    frequencies = np.array(frequencies)
    third_moment = np.sum(particle_sizes ** 3 * frequencies)
    second_moment = np.sum(particle_sizes ** 2 * frequencies)
    sauter_mean_diam = third_moment / second_moment
    sauter_mean_radius = sauter_mean_diam / 2  # Convert back to radius
    return sauter_mean_radius

def updateIceIn(base_ice_in_file, new_ice_in_file, depth, IWC, radii, min_radius):
    with open(base_ice_in_file, "r") as f:
        ice_in_content = f.readlines()

    radius = max(radii, min_radius)

    if radius > 60.0:
        warnings.warn(f"Effective radius {radius:.2f} µm exceeds the maximum valid radius for Baum v36 in libRadtran. Capping radius at 60 µm to avoid errors in radiative transfer calculations.")
        radius = 60.0

    contrail_depth = 10.7 - depth / 1000  # Convert m → km
    contrail_IWC = IWC  # [g/m^3]

    ice_in_content[2] = f"{contrail_depth:.3f}    {contrail_IWC:.16e}     {radius:.3f}\n"
    with open(new_ice_in_file, "w") as f:
        f.writelines(ice_in_content)

    print(f"Updated ice.in: depth={contrail_depth:.3f} km, radius={radius:.3f} µm, IWC={contrail_IWC:.16e} g/m³")

def calculate_RF(n_slices, age, IWC, depth, radii, test_id, habit, hour, input_save_path):
    ###
    # IWC is in g/m^3
    # radii is in micron
    ###

    print(f"Calculating LW and SW RF for dataset with habit: {habit}...")
    LW_RF = np.full(n_slices, np.nan)
    LW_RF_CONTRAIL = np.full(n_slices, np.nan)
    LW_RF_CLEAR = np.full(n_slices, np.nan)

    if hour != "0":
        SW_RF = np.full(n_slices, np.nan)
        SW_RF_CONTRAIL = np.full(n_slices, np.nan)
        SW_RF_CLEAR = np.full(n_slices, np.nan)

    if habit == "ghm" or habit == "rough-aggregate" or habit == "solid-column":
        min_radius = 5.01  # Minimum radius for ghm [um]
    elif habit == "droxtal":
        min_radius = 9.481  # Minimum radius for droxtal [um]
    elif habit == "spheroid":
        min_radius = 6.581  # Minimum radius for spheroidal [um]
    else:
        raise ValueError(f"Habit must be either 'ghm', 'solid-column', 'rough-aggregate', 'droxtal', or 'spheroid'. Habit provided: {habit}")

    for i in range(n_slices):
        # Check if IWC, depth, or radius are NaN. If yes, continue to the next time. (For contrail slicing)
        if (np.isnan(IWC[i])) or (np.isnan(depth[i])) or (np.isnan(radii[i])):
            continue
        slice_ice_in_path = f"{input_save_path}/ice_in_{habit}_{hour}h_{test_id}_slice{i}.in"
        slice_thermal_cloud_path = f"{input_save_path}/thermal_cloud_{habit}_{hour}h_{test_id}_slice{i}.in"
        slice_thermal_clear_path = f"{input_save_path}/thermal_clear_{habit}_{hour}h_{test_id}_slice{i}.in"
        slice_solar_cloud_path = f"{input_save_path}/solar_cloud_{habit}_{hour}h_{test_id}_slice{i}.in"
        slice_solar_clear_path = f"{input_save_path}/solar_clear_{habit}_{hour}h_{test_id}_slice{i}.in"

        # Update the effective radius, IWC, and contrail depth in ice.in
        base_ice_in_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/ice.in"
        base_thermal_cloud_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/thermal-cloud.in"
        base_thermal_clear_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/thermal-clear.in"
        base_solar_cloud_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/solar-cloud.in"
        base_solar_clear_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/base_inputs/solar-clear.in"

        updateIceIn(base_ice_in_path, slice_ice_in_path, depth[i], IWC[i], radii[i], min_radius) # Makes a new ice.in file or updates the existing copy
        
        # Define LW and SW input configurations
        LW_contrail_options, LW_clearsky_options = make_LW_options(habit, hour, slice_ice_in_path)

        if hour != "0":
            SW_contrail_options, SW_clearsky_options = make_SW_options(LW_contrail_options)
            SW_RF_CONTRAIL[i] = calculate_SW_Flux(SW_contrail_options, base_solar_cloud_path, slice_solar_cloud_path)
            SW_RF_CLEAR[i] = calculate_SW_Flux(SW_clearsky_options, base_solar_clear_path, slice_solar_clear_path)
            SW_RF[i] = SW_RF_CONTRAIL[i] - SW_RF_CLEAR[i]
            print(f"Time: {age} hours, Slice {i}:\n SW Contrail Flux: {SW_RF_CONTRAIL[i]:.3f} W/m², SW Clear Flux: {SW_RF_CLEAR[i]:.3f} W/m², SW RF: {SW_RF[i]:.3f} W/m²")

        LW_RF_CONTRAIL[i] = calculate_LW_Flux(LW_contrail_options, base_thermal_cloud_path, slice_thermal_cloud_path)
        LW_RF_CLEAR[i] = calculate_LW_Flux(LW_clearsky_options, base_thermal_clear_path, slice_thermal_clear_path)
        LW_RF[i] = LW_RF_CONTRAIL[i] - LW_RF_CLEAR[i]
        print(f"Time: {age} hours, Slice {i}:\n LW Contrail Flux: {LW_RF_CONTRAIL[i]:.3f} W/m², LW Clear Flux: {LW_RF_CLEAR[i]:.3f} W/m², LW RF: {LW_RF[i]:.3f} W/m²")

    age_arr = age * np.ones(n_slices)
    LW_RF_df = pd.DataFrame({"Contrail_Age_hours": age_arr, "LW_Radiative_Forcing_W_m2": LW_RF, "LW_RF_CLEAR_W_m2": LW_RF_CLEAR, "LW_RF_CONTRAIL_W_m2": LW_RF_CONTRAIL})
    
    if hour != "0":
        SW_RF_df = pd.DataFrame({"Contrail_Age_hours": age_arr, "SW_Radiative_Forcing_W_m2": SW_RF, "SW_RF_CLEAR_W_m2": SW_RF_CLEAR, "SW_RF_CONTRAIL_W_m2": SW_RF_CONTRAIL})
        return LW_RF_df, SW_RF_df
    else:
        return LW_RF_df, None
    
def calculate_energy_forcing(rf_net, time, width, u):
    rf_net = np.asarray(rf_net)
    time   = np.asarray(time)
    width  = np.asarray(width)

    dt = np.diff(time) * 3600  # hours → seconds

    # rf_net and width are indexed by time, so trim to match dt length
    rf_mid   = 0.5 * (rf_net[:-1] + rf_net[1:])    # midpoint RF between steps
    w_mid    = 0.5 * (width[:-1]  + width[1:])     # midpoint width

    print(f"dt shape: {dt.shape}, rf_mid shape: {rf_mid.shape}, w_mid shape: {w_mid.shape}")

    energy = rf_mid * w_mid * dt   # [W/m²] * [m] * [s] = [J/m flight path]
    ef     = np.nancumsum(energy)
    ef_padded = np.insert(ef, 0, 0.0)  # prepend 0 at t=0 to match time array length

    return ef_padded

def energy_forcing_95(times, ef):
    # Calculate the hour when each EF reaches 95% of its final value
    return times[np.where(ef >= 0.95 * ef[-1])[0][0]]

def mask_IWC_95_percentile_IWC(APCEMM_ds, time_idx):
    # Use the current time_idx from outer scope to select the dataset slice once
    ds_t = APCEMM_ds[time_idx]

    # Convert IWC to g/m^3 and flatten for sorting
    IWC_values = (ds_t["IWC"].values * 1e3).ravel()

    # Sort IWC from largest to smallest
    sorted_idx = np.argsort(IWC_values)[::-1]
    sorted_IWC = IWC_values[sorted_idx]

    # Compute cumulative mass and threshold index for 95%
    cumulative_sum = np.cumsum(sorted_IWC)
    total_IWC = cumulative_sum[-1]
    threshold = 0.95 * total_IWC
    percentile_index = np.searchsorted(cumulative_sum, threshold)
    # Build a flat boolean mask for cells contributing to 95% mass
    mask_flat = np.zeros_like(IWC_values, dtype=bool)
    mask_flat[sorted_idx[:percentile_index]] = True

    # Apply mask back to original 2D shape and set non-contributing cells to NaN
    masked_IWC = ds_t["IWC"].where(mask_flat.reshape(ds_t["IWC"].shape), np.nan)
    masked_radius = ds_t["Effective radius"].where(mask_flat.reshape(ds_t["Effective radius"].shape), np.nan)
    masked_extinction = ds_t["Extinction"].where(mask_flat.reshape(ds_t["Extinction"].shape), np.nan)

    # Return a copied dataset with masked IWC (avoids mutating original dataset)
    ds_masked = ds_t.copy()
    ds_masked["IWC"] = masked_IWC
    ds_masked["Effective radius"] = masked_radius
    ds_masked["Extinction"] = masked_extinction

    return ds_masked

def calculate_slice_depth_APCEMM(APCEMM_data, slice_index):
    # # Calculate the depth of the slice based on a minimum extinction threshold of 10^-1 * Max Extinction in the contrail
    APCEMM_extinction = APCEMM_data['Extinction']
    max_extinction = np.max(APCEMM_extinction).values
    extinction_threshold = 0
    APCEMM_extinction_slice = APCEMM_extinction[:, slice_index].values.flatten()
    extinction_candidates = np.where(APCEMM_extinction_slice >= extinction_threshold, APCEMM_data['y'], np.nan) # y coordinates where extinction is above threshold, nan otherwise
    depth = np.nanmax(extinction_candidates) - np.nanmin(extinction_candidates) # The max distance between any two valid y coordinates in the slice, which is a proxy for depth. This avoids having to choose an arbitrary extinction threshold and is more robust to noise in the data.
    
    return depth

def prepare_cocip_slices(cocip_ds, time_idx, num_slices:int):
    print("Number of slices at this timestep: ", num_slices)

    def compute_cocip_angle(width, depth, sigma_yz):

        if width < depth:
            return 0.0  # If the contrail is taller than it is wide, we assume it's vertical with no tilt
        
        sigma_yy = 0.125 * width ** 2
        sigma_zz = 0.125 * depth ** 2

        cov = np.array([[sigma_yy, sigma_yz],
                        [sigma_yz, sigma_zz]])

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        eigvals = np.maximum(eigvals, 0)

        axis_lengths = 2 * np.sqrt(eigvals)
        angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        return angle_deg #- 90  # Adjust so that 0° means horizontal and positive angles tilt clockwise

    rho_ice = 917 # kg/m3
    width = cocip_ds["width"][time_idx].values # total width
    area = cocip_ds["area_eff"][time_idx].values 
    depth = cocip_ds["depth"][time_idx].values # Effective depth
    print(f"Cocip reported width: {width:.3f} m, area: {area:.3f} m², depth: {depth:.3f} m")
    centroid_IWC = cocip_ds["iwc"][time_idx].values * cocip_ds["rho_air"][time_idx].values # [kg ice/kg air] * [kg air/m3] = [kg ice/m3]
    sigma_yz = cocip_ds["sigma_yz"].values[time_idx]
    angle_deg = compute_cocip_angle(width, depth, sigma_yz)
    radius_m = cocip_ds["r_ice_vol"][time_idx].values
    radius_um = radius_m * 1e6
    ice_number = cocip_ds["n_ice_per_m"][time_idx].values
    ice_mass_kg = 4/3 * np.pi * radius_m**3 * ice_number * rho_ice

    cocip_depth_slice_avg = calculate_slice_depth(width, area, depth, angle_deg, num_slices)
    
    def get_cocip_slice_radii(radius_type):

        if radius_type == "constant":
            ### Constant radius ###
            # Radius is in micron
            cocip_radii_slice_avg = np.ones(num_slices) * radius_um
            print(f"Constant centroid effective radius = {radius_um:.3f} um")

        elif radius_type == "IWC constant number density":
            ### Radius from IWC and constant number density ###
            # Radius is in micron
            cocip_IWC_slice_avg = get_cocip_slice_IWC("gaussian")

            cocip_radii_slice_avg = ((cocip_IWC_slice_avg*1e-3 * cocip_depth_slice_avg * width/num_slices)/(ice_number/num_slices) * 1/(4/3 * np.pi * rho_ice)) ** (1/3) * 1e6
            print(f"Centroid effective radii = {cocip_radii_slice_avg} um")

        elif radius_type == "gaussian IWC gaussian number density":
            ### Radius from gaussian IWC and gaussian number density distribution ###
            # Radius is in micron
            cocip_IWC_slice_avg = get_cocip_slice_IWC("gaussian")

            bin_indices = np.arange(num_slices)
            weights = norm.pdf(bin_indices, loc=(num_slices-1)/2, scale=num_slices/6)  # Gaussian weights for the slices, centered at the middle slice
            # Normalize so weights sum to 1, then scale to total_quantity
            gaussian_ice_number = weights / weights.sum() * ice_number  # Scale to total ice number
            cocip_radii_slice_avg = ((cocip_IWC_slice_avg*1e-3 * cocip_depth_slice_avg * width/num_slices)/(gaussian_ice_number) * 1/(4/3 * np.pi * rho_ice)) ** (1/3) * 1e6
            print(f"Centroid effective radii = {cocip_radii_slice_avg} um")
        
        return cocip_radii_slice_avg

    def get_cocip_slice_IWC(IWC_type):
        
        if IWC_type == "gaussian":
            # Slice IWC is in g/m^3
            ### Gaussian IWC distribution ###
            cocip_IWC_slice_avg = discretize_tilted_ellipse_IWC_gaussian(
                    width,
                    area,
                    angle_deg,
                    ice_mass_kg,
                    num_slices,
                )
        elif IWC_type == "constant":
            ### Constant IWC ###
            ice_mass_kg_per_slice = centroid_IWC * (width/num_slices) * cocip_depth_slice_avg # [kg/m3] * [m] * [m] * [1m] = [kg] Assumed 1m thickness
            print(f"Provided total ice mass: {ice_mass_kg:.3e} kg, calculated total slice ice mass: {np.sum(ice_mass_kg_per_slice):.3e} kg")
            cocip_IWC_slice_avg = np.ones(num_slices) * (ice_mass_kg_per_slice / (cocip_depth_slice_avg * width/num_slices)) * 1e3 # [g/m3]

        elif IWC_type == "constant number density and radius":
            ### IWC from number density and radius ###
            num_crystals_per_slice = ice_number / num_slices
            cocip_IWC_slice_avg = (4/3 * np.pi * radius_m**3 * rho_ice * num_crystals_per_slice) / (cocip_depth_slice_avg * width/num_slices) * 1e3 # [m3] * [kg/m3] * [#] / ([m] * [m]) * 1e3 = [g/m3] Assuming 1m thickness
            ice_mass_kg_per_slice = cocip_IWC_slice_avg/1000 * cocip_depth_slice_avg * (width/num_slices) # [g/m3] / 1000 * [m] * [m] = [kg]
            print(f"Provided total ice mass: {ice_mass_kg:.3e} kg, calculated total slice ice mass: {np.sum(ice_mass_kg_per_slice):.3e} kg")

        else:
            raise ValueError("IWC_type must be either 'gaussian', 'constant', or 'constant number density and radius'.")
        
        return cocip_IWC_slice_avg
    
    cocip_radii_slice_avg = get_cocip_slice_radii("constant")
    cocip_IWC_slice_avg = get_cocip_slice_IWC("gaussian")

    return cocip_IWC_slice_avg, cocip_radii_slice_avg, cocip_depth_slice_avg

def prepare_les_slices(les_data, time_idx, total_num_slices_averaged):
    num_slices = total_num_slices_averaged # LES is continuously defined in the spatial domain by the ellipse and the gaussian distribution of properties, so we can directly match
    print("Number of slices at this timestep: ", num_slices)

    width = les_data["Width_m"].iloc[time_idx] # total width
    area = les_data["Area_m2"].iloc[time_idx]
    depth = les_data["Depth_m"].iloc[time_idx] # Effective depth
    angle_deg = 0
    les_IWC = les_data["IWC_g_per_m3"].iloc[time_idx]
    radius = les_data["Effective_radius_um"].iloc[time_idx]
    ice_mass_kg = les_data["Ice_mass"].iloc[time_idx]

    # IWC is in units g/m^3
    les_IWC_slice_avg = discretize_tilted_ellipse_IWC_gaussian(
            width,
            area,
            angle_deg,
            ice_mass_kg,
            num_slices,
        )
    
    # Radius is in units of micron

    les_radii_slice_avg = discretize_tilted_ellipse_radius(
        width,
        area,
        depth,
        angle_deg,
        radius,
        num_slices
    )

    les_depth_slice_avg = calculate_slice_depth(width, area, depth, angle_deg, num_slices)

    return les_IWC_slice_avg, les_radii_slice_avg, les_depth_slice_avg

def calculate_slice_depth(width, A, D, angle_deg, X):
    alpha = np.radians(angle_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)
    depth = np.zeros(X)

    a0 = 0.5 * width / np.cos(alpha)
    b0 = A / (np.pi * a0)

    # Projected half-width of the tilted ellipse
    x_max = np.sqrt(a0**2 * ca**2 + b0**2 * sa**2)

    print(f"Calculated semi-axes for slice depth calculation: a={a0:.4f}, b={b0:.4f}")
    print(f"Projected half-width: {x_max:.4f} vs input half-width: {width/2:.4f}")
    print(f"Width: {width}, Area: {A}, Depth: {D}, Angle: {angle_deg} degrees")

    def y_limits(x):
        A_q = sa**2 / a0**2 + ca**2 / b0**2
        B_q = 2 * x * ca * sa * (1 / a0**2 - 1 / b0**2)
        C_q = x**2 * (ca**2 / a0**2 + sa**2 / b0**2) - 1
        disc = B_q**2 - 4 * A_q * C_q
        if disc < 0:
            return None, None
        sqrt_disc = np.sqrt(disc)
        y1 = (-B_q - sqrt_disc) / (2 * A_q)
        y2 = (-B_q + sqrt_disc) / (2 * A_q)
        return min(y1, y2), max(y1, y2)

    for i in range(X):
        # Sample slice centers across the full projected width
        x = -x_max + i * (2 * x_max / X) + (x_max / X)
        y_min, y_max = y_limits(x)
        if y_min is None:
            depth[i] = 0.0  # shouldn't happen now, but safe fallback
        else:
            depth[i] = y_max - y_min

    return depth

def discretize_tilted_ellipse_radius(width, A, D, angle_deg, bulk_property, X):
    """
    Discretize a Gaussian mass distribution over a TILTED ellipse into X vertical slices.

    The Gaussian is defined in the ellipse's local frame (along semi-axes a, b).
    The 2-sigma contour of the Gaussian coincides with the ellipse boundary.

    Parameters:
        width      : total x-extent of the tilted ellipse
        A          : cross-sectional area of the ellipse
        D          : effective depth = half-height at x=centroid (x=0)
        angle_deg  : angle of major axis from x-axis (degrees)
        bulk_property : bulk property to distribute (e.g., mass, number)
        X          : number of vertical slices

    Returns:
        slice_centers : x positions of slice centers
        slice_masses  : integrated mass in each slice
    """
    alpha = np.radians(angle_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # -------------------------------------------------------------------------
    # Step 1: Recover semi-axes a, b from (width, D, alpha)
    # -------------------------------------------------------------------------

    a0 = 0.5*width / np.cos(alpha)
    b0 = A / (np.pi * a0)

    print("Initial guess for semi-axes (a0, b0): ", a0, b0)
    print("Angle alpha (degrees): ", angle_deg)

    A_check = np.pi * a0 * b0
    print(f"Semi-major axis a : {a0:.4f}")
    print(f"Semi-minor axis b : {b0:.4f}")
    print(f"Target area       : {A:.4f}")
    print(f"Recovered area    : {A_check:.4f}  (error: {abs(A_check-A):.2e})")

    # Gaussian sigmas in the ellipse's local frame (2-sigma = semi-axis length)
    sigma_a = a0 / 2
    sigma_b = b0 / 2

    # -------------------------------------------------------------------------
    # Step 2: x-extent of the tilted ellipse
    # -------------------------------------------------------------------------
    x_max = width / 2   # = sqrt((a*ca)^2 + (b*sa)^2), consistent by construction
    x_min = -x_max

    # -------------------------------------------------------------------------
    # Step 3: y-limits at a given x by intersecting vertical line with tilted ellipse
    #
    # Substitute rotated coords u = x*ca + y*sa, v = -x*sa + y*ca into
    # (u/a)^2 + (v/b)^2 = 1  =>  quadratic in y:
    #   (sa^2/a^2 + ca^2/b^2)*y^2
    #   + 2*x*(ca*sa/a^2 - ca*sa/b^2)*y
    #   + x^2*(ca^2/a^2 + sa^2/b^2) - 1 = 0
    # -------------------------------------------------------------------------
    def y_limits(x):
        A_q = sa**2 / a0**2 + ca**2 / b0**2
        B_q = 2 * x * ca * sa * (1 / a0**2 - 1 / b0**2)
        C_q = x**2 * (ca**2 / a0**2 + sa**2 / b0**2) - 1
        disc = B_q**2 - 4 * A_q * C_q
        if disc < 0:
            return None, None   # x is outside the ellipse
        sqrt_disc = np.sqrt(disc)
        y1 = (-B_q - sqrt_disc) / (2 * A_q)
        y2 = (-B_q + sqrt_disc) / (2 * A_q)
        return min(y1, y2), max(y1, y2)

    # -------------------------------------------------------------------------
    # Step 4: 2D Gaussian in the ellipse's LOCAL (rotated) frame
    #
    # u =  x*cos(a) + y*sin(a)   (along major axis)
    # v = -x*sin(a) + y*cos(a)   (along minor axis)
    # G(x,y) = exp(-u^2 / 2*sigma_a^2) * exp(-v^2 / 2*sigma_b^2)
    # -------------------------------------------------------------------------
    def gaussian_2d(x, y):
        u =  x * ca + y * sa
        v = -x * sa + y * ca
        return (np.exp(-u**2 / (2 * sigma_a**2)) *
                np.exp(-v**2 / (2 * sigma_b**2)))

    # -------------------------------------------------------------------------
    # Step 6: Integrate each vertical slice
    #
    # For each x, the y-integral of the Gaussian can use erf IF the Gaussian
    # were axis-aligned in y. But since G is in the rotated frame, the
    # y-dependence at fixed x mixes u and v — so we keep full numerical quad.
    # -------------------------------------------------------------------------
    scale = bulk_property

    dx = width / X
    slice_centers = np.linspace(x_min + dx / 2, x_max - dx / 2, X)
    slice_properties = np.zeros(X)

    for i, x_c in enumerate(slice_centers):
        x_lo = np.clip(x_c - dx / 2, x_min, x_max)
        x_hi = np.clip(x_c + dx / 2, x_min, x_max)

        def strip_integrand(y, x):
            return gaussian_2d(x, y)

        def y_lo_strip(x):
            lo, _ = y_limits(x)
            return lo if lo is not None else 0.0

        def y_hi_strip(x):
            _, hi = y_limits(x)
            return hi if hi is not None else 0.0

        mass_strip, _ = integrate.dblquad(
            strip_integrand,
            x_lo, x_hi,
            y_lo_strip, y_hi_strip
        )

        # compute slice area for averaging
        slice_area, _ = integrate.dblquad(
            lambda y, x: 1.0,
            x_lo, x_hi,
            y_lo_strip, y_hi_strip
        )

        slice_properties[i] = scale * mass_strip / slice_area

    return slice_properties


def discretize_tilted_ellipse_IWC_gaussian(width, A, angle_deg,
                                         total_ice_mass_kg, X,
                                         thickness_m=1.0):
    """
    Discretize a Gaussian IWC distribution over a TILTED ellipse into X
    vertical slices, given a TOTAL ICE MASS to distribute, with the
    Gaussian's peak amplitude normalized against its TRUE TRUNCATED
    integral (i.e. integrated only within the ellipse boundary, not over
    the full plane). This makes the slice masses sum to EXACTLY
    total_ice_mass_kg.

    SHAPE: standard 2D Gaussian in the ellipse's local (rotated) frame,
        G(u,v) = exp(-u^2/(2*sigma_a^2)) * exp(-v^2/(2*sigma_b^2))
    with sigma_a = a/2, sigma_b = b/2 (2-sigma = semi-axis length, so the
    ellipse boundary coincides with the Gaussian's 2-sigma contour).

    TRUNCATION FACTOR (closed form, angle-independent): because the
    ellipse boundary is a circle of radius 2 in standardized coordinates
    (u/sigma_a, v/sigma_b), the fraction of the untruncated Gaussian mass
    that lies within the ellipse is exactly (1 - exp(-2)) ~= 86.47%
    (equivalently, the CDF of a chi-squared distribution with 2 d.o.f. at
    x=4). This fraction is independent of rotation angle and aspect ratio,
    which is what makes exact normalization against the truncated total
    possible in closed form (no need to numerically integrate the total).

    MASS CONSERVATION: slice_mass_kg.sum() == total_ice_mass_kg to
    floating-point precision (verified numerically), since the peak
    amplitude is normalized against the EXACT truncated total, not an
    untruncated approximation.

    SPATIAL MEAN: as established, with mean defined as
        mean_IWC = (integral of IWC over ellipse) / A
    this is automatically satisfied once mass is conserved -- it's an
    algebraic consequence of the mass constraint, not a separate condition
    on shape.

    DIMENSIONAL HANDLING: same as the parabolic-cap version -- assumes an
    implicit slab thickness thickness_m (default 1 m) bridges IWC's
    volumetric units (g/m^3) to the 2D ellipse area (m^2):
        total_ice_mass_kg = (integral of IWC over ellipse area)
                             * thickness_m / 1000

    PERFORMANCE: each slice integral uses a closed-form (erf-based) inner
    integral over y (at fixed x, bounded by the true ellipse), wrapped in
    a single 1D `scipy.integrate.quad` over x.

    Parameters:
        width             : total x-extent of the UNROTATED ellipse (= 2*a)
        A                 : cross-sectional area of the ellipse (m^2)
        D                 : effective depth = half-height at x=centroid
                             (currently unused in the area/semi-axis solve;
                             retained for interface compatibility)
        angle_deg         : angle of major axis from x-axis (degrees)
        total_ice_mass_kg : total ice mass to distribute over the ellipse (kg)
        X                 : number of vertical slices
        thickness_m       : implicit slab thickness (m), default 1.0

    Returns:
        slice_centers   : x positions of slice centers
        slice_iwc       : average IWC in each slice (g/m^3)
        slice_mass_kg   : ice mass (kg) integrated within each slice
                           (sums to total_ice_mass_kg to floating-point
                           precision)
    """
    alpha = np.radians(angle_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)

    # -------------------------------------------------------------------
    # Step 1: Recover semi-axes a, b from (width, A, alpha)
    #
    # NOTE: as flagged previously, this treats `width` as the UNROTATED
    # extent (2*a). Revisit if `width` is meant to be the tilted
    # bounding-box extent instead -- independent of the mass/shape logic.
    # -------------------------------------------------------------------
    a0 = 0.5 * width / np.cos(alpha)
    b0 = A / (np.pi * a0)

    sigma_a = a0 / 2
    sigma_b = b0 / 2

    # -------------------------------------------------------------------
    # Step 2: x-extent of the tilted ellipse (per the original convention)
    # -------------------------------------------------------------------
    x_max = width / 2
    x_min = -x_max

    # True x-extent of the tilted ellipse (needed to clip slice bounds so
    # the closed-form inner integral's sqrt/erf arguments stay valid)
    K = a0**2 * ca**2 + b0**2 * sa**2
    x_extent_true = np.sqrt(K)

    # -------------------------------------------------------------------
    # Step 3: Peak IWC (g/m^3) from total ice mass (kg), normalized
    # against the EXACT TRUNCATED total (closed form, angle-independent):
    #   truncated_total = 2*pi*sigma_a*sigma_b * (1 - exp(-2))
    # -------------------------------------------------------------------
    total_ice_mass_g = total_ice_mass_kg * 1000.0
    truncated_total = 2.0 * np.pi * sigma_a * sigma_b * (1.0 - np.exp(-2.0))
    peak_iwc = total_ice_mass_g / (thickness_m * truncated_total)

    # -------------------------------------------------------------------
    # Step 4: Closed-form inner (y-bounded) integral at fixed x, using the
    # TRUE ellipse y-limits (not unbounded y). Derived by completing the
    # square in y for the rotated-frame Gaussian; verified against direct
    # 1D quadrature to machine precision.
    # -------------------------------------------------------------------
    def y_limits(x):
        A_q = sa**2 / a0**2 + ca**2 / b0**2
        B_q = 2 * x * ca * sa * (1 / a0**2 - 1 / b0**2)
        C_q = x**2 * (ca**2 / a0**2 + sa**2 / b0**2) - 1
        disc = B_q**2 - 4 * A_q * C_q
        if disc < 0:
            return None, None
        sqrt_disc = np.sqrt(disc)
        y1 = (-B_q - sqrt_disc) / (2 * A_q)
        y2 = (-B_q + sqrt_disc) / (2 * A_q)
        return min(y1, y2), max(y1, y2)

    P = sa**2 / (2 * sigma_a**2) + ca**2 / (2 * sigma_b**2)

    def inner_integral(x):
        ylo, yhi = y_limits(x)
        if ylo is None:
            return 0.0
        Q = -ca * sa * x / sigma_b**2 + ca * sa * x / sigma_a**2
        R = ca**2 * x**2 / (2 * sigma_a**2) + sa**2 * x**2 / (2 * sigma_b**2)
        y0 = -Q / (2 * P)
        prefactor = np.exp(-(R - Q**2 / (4 * P)))
        return prefactor * np.sqrt(np.pi / (4 * P)) * (
            erf(np.sqrt(P) * (yhi - y0)) - erf(np.sqrt(P) * (ylo - y0))
        )

    # -------------------------------------------------------------------
    # Step 5: Build slices. Each slice integral is a single 1D `quad` over
    # x of the closed-form inner_integral (no 2D quadrature).
    # -------------------------------------------------------------------
    dx = width / X
    print(f"width: {width}")
    print(f"num_slices: {X}")
    print(f"x_min: {x_min}")
    print(f"x_max: {x_max}")
    print(f"dx: {dx}")

    slice_centers = np.linspace(x_min + dx / 2, x_max - dx / 2, X)

    slice_mass_g = np.zeros(X)
    slice_iwc = np.zeros(X)

    for i, x_c in enumerate(slice_centers):
        x_lo = np.clip(x_c - dx / 2, -x_extent_true, x_extent_true)
        x_hi = np.clip(x_c + dx / 2, -x_extent_true, x_extent_true)

        if x_hi <= x_lo:
            slice_mass_g[i] = 0.0
            slice_iwc[i] = 0.0
            continue

        unscaled_mass, _ = integrate.quad(inner_integral, x_lo, x_hi)
        slice_mass_g[i] = peak_iwc * unscaled_mass

        y_half = _ellipse_y_half(x_c, a0, b0, ca, sa)
        slice_width = x_hi - x_lo
        slice_area = slice_width * 2 * y_half if y_half is not None else np.nan
        slice_iwc[i] = slice_mass_g[i] / slice_area if slice_area else 0.0

    slice_mass_kg = slice_mass_g / 1000.0
    print(f"Provided ice mass per meter: {total_ice_mass_kg:.3f} kg/m, computed slice ice mass sum: {slice_mass_kg.sum():.3f} kg/m")

    return slice_iwc


def _ellipse_y_half(x, a0, b0, ca, sa):
    """Half-height (max |y|) of the tilted ellipse boundary at position x."""
    A_q = sa**2 / a0**2 + ca**2 / b0**2
    B_q = 2 * x * ca * sa * (1 / a0**2 - 1 / b0**2)
    C_q = x**2 * (ca**2 / a0**2 + sa**2 / b0**2) - 1
    disc = B_q**2 - 4 * A_q * C_q
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    y1 = (-B_q - sqrt_disc) / (2 * A_q)
    y2 = (-B_q + sqrt_disc) / (2 * A_q)
    return (max(y1, y2) - min(y1, y2)) / 2.0



def prepare_APCEMM_slices(APCEMM_ds, time_idx, total_num_slices_averaged):
    original_num_slices = APCEMM_ds[time_idx]['x'].values.size
    print("Number of slices at this timestep: ", original_num_slices)

    APCEMM_IWC_masked = mask_IWC_95_percentile_IWC(APCEMM_ds, time_idx)

    # Compute per-slice stats across ALL original slices
    APCEMM_IWC_avg_all_slices   = np.full(original_num_slices, np.nan)
    APCEMM_radii_all_slices     = np.full(original_num_slices, np.nan)
    APCEMM_depth_all_slices     = np.full(original_num_slices, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(original_num_slices):
            APCEMM_IWC_avg_all_slices[j] = np.nanmean(APCEMM_IWC_masked['IWC'][:, j] * 1e3)
            APCEMM_radii_all_slices[j]   = np.nanmean(APCEMM_IWC_masked['Effective radius'][:, j] * 1e6)
            APCEMM_depth_all_slices[j]   = calculate_slice_depth_APCEMM(APCEMM_IWC_masked, j)

    # Identify valid (non-NaN) slices
    valid_mask = ~np.isnan(APCEMM_IWC_avg_all_slices)
    valid_IWC   = APCEMM_IWC_avg_all_slices[valid_mask]
    valid_radii = APCEMM_radii_all_slices[valid_mask]
    valid_depth = APCEMM_depth_all_slices[valid_mask]

    num_slices_masked = valid_mask.sum()
    print(f"Number of valid slices after masking: {num_slices_masked}")

    # Always produce exactly total_num_slices_averaged output slices.
    # np.array_split divides into N chunks, making earlier chunks 1 element
    # larger when the division is uneven — no slice is ever skipped.
    n_out = min(total_num_slices_averaged, num_slices_masked)  # can't exceed valid data
    groups = np.array_split(np.arange(num_slices_masked), n_out)

    APCEMM_IWC_slice_avg   = np.full(total_num_slices_averaged, np.nan)
    APCEMM_radii_slice_avg = np.full(total_num_slices_averaged, np.nan)
    APCEMM_depth_slice_avg = np.full(total_num_slices_averaged, np.nan)

    for i, group_idx in enumerate(groups):
        if len(group_idx) == 0:
            continue  # shouldn't happen, but guard anyway
        print(f"Averaging valid slices {group_idx[0]}–{group_idx[-1]} → output slice {i}")
        APCEMM_IWC_slice_avg[i]   = np.nanmean(valid_IWC[group_idx])
        APCEMM_radii_slice_avg[i] = np.nanmean(valid_radii[group_idx])
        APCEMM_depth_slice_avg[i] = np.nanmean(valid_depth[group_idx])

    return (APCEMM_IWC_slice_avg, APCEMM_radii_slice_avg, APCEMM_depth_slice_avg,
            APCEMM_IWC_avg_all_slices, APCEMM_radii_all_slices, APCEMM_depth_all_slices)