#!/usr/bin/python
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import os
import xarray as xr
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/LRT/')
import LRT_fxnlib as LRTlib
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as pipelinelib
from concurrent.futures import ProcessPoolExecutor

test_id = str(sys.argv[1])
habits = sys.argv[2].split("+") # List of habits to process, passed as a command line argument
hours = sys.argv[3].split("+") # Midnight (0) and/or Noon (12)
ncpus = int(sys.argv[4]) # Number of CPUs to use for parallel processing, passed as a command line argument

def csv_to_netcdf_and_merge(csv_path, results_ds):
    # 1. Load CSV
    df = pd.read_csv(csv_path)
    
    # 2. Convert to xarray Dataset, indexed by 'index' to match results_ds
    csv_ds = xr.Dataset(
        {
            "IWC_g_per_m3":        (["time"], df["IWC_g_per_m3"]),
            "Depth_m":             (["time"], df["Depth_m"]),
            "Width_m":             (["time"], df["Width_m"]),
            "Effective_radius_um": (["time"], df["Effective_radius_um"]),
            "Ice_mass":            (["time"], df["Ice_mass"]),
            "Ice_number":          (["time"], df["Ice_number"]),
            "Ice_surface_area":    (["time"], df["Ice_surface_area"]),
            "Mean_altitude_m":     (["time"], df["Mean_altitude_m"]),
            "Area_m2":             (["time"], df["Area_m2"]),
            "Time_hours":          (["time"], df["Time_hours"]),
        },
        coords={"time": df["Time_hours"]}  # must match results_ds time coord
    )

    # 3. Merge — xr.merge aligns on shared coords (time)
    merged_ds = xr.merge([results_ds, csv_ds])

    return merged_ds

def calculate_RF_les(LES_data, test_id, habit, hour):
    if hour == "0":
        print(f"Processing LES data for {test_id} at hour {hour} with habit {habit} (Midnight)")
        tod = "midnight"
    else:
        print(f"Processing LES data for {test_id} at hour {hour} with habit {habit} (Noon)")
        tod = "noon"

    LES_data_path = f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/processed_data/all_data/{test_id}.csv'
    print(f"LES data location: {LES_data_path}")

    num_timesteps = len(LES_data['Time_hours'])
    n_slices = 25
    slice_coord = np.arange(n_slices)

    print(f"Total number of timesteps in dataset: {num_timesteps}")

    # Pre-allocate result arrays
    lw_slices_midnight_list  = np.full((num_timesteps, n_slices), np.nan)
    sw_slices_noon_list      = np.full((num_timesteps, n_slices), np.nan)
    lw_slices_noon_list      = np.full((num_timesteps, n_slices), np.nan)
    iwc_slices               = np.full((num_timesteps, n_slices), np.nan)
    radii_slices             = np.full((num_timesteps, n_slices), np.nan)
    depth_slices             = np.full((num_timesteps, n_slices), np.nan)
    ef_slices                = np.full((num_timesteps, n_slices), np.nan)
    # Scalar-per-timestep arrays (computed inline; no second loop needed)
    lw_rf_midnight_list      = np.full(num_timesteps, np.nan)
    sw_rf_noon_list          = np.full(num_timesteps, np.nan)
    lw_rf_noon_list          = np.full(num_timesteps, np.nan)
    ef                       = np.full(num_timesteps, np.nan)
    # Scalar-per-timestep arrays (Averaged inputs: AI)
    lw_midnight_AI_list      = np.full(num_timesteps, np.nan)
    sw_noon_AI_list          = np.full(num_timesteps, np.nan)
    lw_noon_AI_list          = np.full(num_timesteps, np.nan)
    iwc_AI_list              = np.full(num_timesteps, np.nan)
    depth_AI_list            = np.full(num_timesteps, np.nan)
    radius_AI_list           = np.full(num_timesteps, np.nan)
    ef_AI                    = np.full(num_timesteps, np.nan)
    ef_cumsum_AI             = np.full(num_timesteps, np.nan)

    for time_idx in range(num_timesteps):
        base_save_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/RF_EF_results/slicing_results/{test_id}/RF" # Where to save the input and output files
        input_save_path  = f"{base_save_path}/inputs/timestep_{time_idx}"
        output_save_path = f"{base_save_path}/outputs/timestep_{time_idx}"
        for path in [input_save_path, output_save_path]:
            os.makedirs(path, exist_ok=True)

        contrail_age = LES_data["Time_hours"].iloc[time_idx] # Hours

        # IWC [g/m3], radii [um], depth [m]
        LES_IWC_slice_avg, LES_radii_slice_avg, LES_depth_slice_avg = LRTlib.prepare_les_slices(LES_data, time_idx, n_slices)
        # --- Calculate RF from Averaged Inputs (Avg IWC, radius, and depth per timestep) ---
        iwc_AI = LES_data["IWC_g_per_m3"].iloc[time_idx] # [g/m3]
        depth_AI = LES_data["Depth_m"].iloc[time_idx] # [m]
        radius_AI = LES_data["Effective_radius_um"].iloc[time_idx] # radius in micron [um]

        iwc_AI_list[time_idx] = iwc_AI
        depth_AI_list[time_idx] = depth_AI
        radius_AI_list[time_idx] = radius_AI

        # Store slice data ang AI data in pre-allocated arrays
        iwc_slices[time_idx,   :n_slices] = LES_IWC_slice_avg[:n_slices]
        radii_slices[time_idx, :n_slices] = LES_radii_slice_avg[:n_slices]
        depth_slices[time_idx, :n_slices] = LES_depth_slice_avg[:n_slices]
        combined_iwc    = list(LES_IWC_slice_avg) + [iwc_AI]
        combined_depth  = list(LES_depth_slice_avg) + [depth_AI]
        combined_radius = list(LES_radii_slice_avg) + [radius_AI]

        # Single call to libRadTran
        LW_RF_combined, SW_RF_combined = LRTlib.calculate_RF(
            n_slices + 1, contrail_age, combined_iwc, combined_depth, combined_radius,
            test_id, habit, hour, input_save_path,
        )
        lw_vals = LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[:n_slices]
        lw_AI = float(LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[n_slices])
        
        if SW_RF_combined is not None: # LW + SW at noon
            sw_vals = SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[:n_slices]
            sw_slices_noon_list[time_idx, :len(sw_vals)] = sw_vals
            lw_slices_noon_list[time_idx, :len(lw_vals)] = lw_vals

            sw_AI = float(SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[n_slices])
            lw_noon_AI_list[time_idx] = lw_AI # LW at noon
            sw_noon_AI_list[time_idx] = sw_AI # SW at noon
        else: # LW at midnight
            lw_slices_midnight_list[time_idx, :len(lw_vals)] = lw_vals
            lw_midnight_AI_list[time_idx] = lw_AI

        # Average across slices inline — avoids a second full loop later
        lw_rf_midnight_list[time_idx]  = np.nanmean(lw_slices_midnight_list[time_idx])
        sw_rf_noon_list[time_idx] = np.nanmean(sw_slices_noon_list[time_idx])
        lw_rf_noon_list[time_idx] = np.nanmean(lw_slices_noon_list[time_idx])

        # Free LRT result objects as soon as we're done with them
        del LW_RF_combined, SW_RF_combined
                
    # Energy forcing
    # Calculate total RF per slice
    if hour != "0":
        rf_net_slices = lw_slices_noon_list + sw_slices_noon_list
        rf_net_AI = lw_noon_AI_list + sw_noon_AI_list
    else:
        rf_net_slices = lw_slices_midnight_list
        rf_net_AI = lw_midnight_AI_list

    widths = LES_data["Width_m"]
    slice_widths = widths/n_slices
    delta_t = 10 * 60 # min to seconds, all timesteps are equal

    for time_idx in range(num_timesteps):
        ef_slices[time_idx, :] = rf_net_slices[time_idx, :] * slice_widths[time_idx] * delta_t  # [W/m²] * [m] * [s] = [J/m]
        ef[time_idx] = np.nansum(ef_slices[time_idx, :])

        ef_AI[time_idx] =  rf_net_AI[time_idx] * widths[time_idx] * delta_t  # [W/m²] * [m] * [s] = [J/m]

    ef_cumsum = np.nancumsum(ef) # Want the cumulative sum
    ef_cumsum_AI = np.nancumsum(ef_AI)

    results_ds = xr.Dataset(
        {
            "LW RF slices midnight":    (["time", "slice"], lw_slices_midnight_list),
            "SW RF slices noon":        (["time", "slice"], sw_slices_noon_list),
            "LW RF slices noon":        (["time", "slice"], lw_slices_noon_list), 
            "IWC slices":               (["time", "slice"], iwc_slices),
            "Effective radius slices":  (["time", "slice"], radii_slices),
            "Depth slices":             (["time", "slice"], depth_slices), 
            "Energy forcing slices":    (["time", "slice"], ef_slices), 
            "LW RF midnight":           ("time", lw_rf_midnight_list),
            "SW RF noon":               ("time", sw_rf_noon_list),
            "LW RF noon":               ("time", lw_rf_noon_list),
            "Energy Forcing":           ("time", ef_cumsum),
            "LW RF midnight AI":        ("time", lw_midnight_AI_list),
            "SW RF noon AI":            ("time", sw_noon_AI_list),
            "LW RF noon AI":            ("time", lw_noon_AI_list),
            "IWC AI":                   ("time", iwc_AI_list),
            "Effective radius AI":      ("time", radius_AI_list),
            "Depth AI":                 ("time", depth_AI_list),
            "Energy Forcing AI":        ("time", ef_cumsum_AI),
            "Habit":                    (habit),
            "Hour":                     (hour),
        },
            coords={"time": LES_data["Time_hours"], "slice": slice_coord},
    )

    # Free large arrays before the merge to avoid holding two copies
    del lw_slices_midnight_list, sw_slices_noon_list, lw_slices_noon_list
    del iwc_slices, radii_slices, depth_slices, ef_slices
    del lw_rf_midnight_list, sw_rf_noon_list, lw_rf_noon_list, ef_cumsum
    del lw_midnight_AI_list, sw_noon_AI_list, lw_noon_AI_list, ef_cumsum_AI
    del rf_net_slices

    LES_data_merged = csv_to_netcdf_and_merge(LES_data_path, results_ds)

    expanded_nc_save_path = f"/home/chinahg/GCresearch/contrailuncertainty/LRT/RF_EF_results/LES/ef_final/{test_id}/"
    file_save_name = f"LES_{test_id}_{hour}h_{habit}.nc"
    os.makedirs(os.path.dirname(expanded_nc_save_path), exist_ok=True)
    LES_data_merged.to_netcdf(os.path.join(expanded_nc_save_path, file_save_name))
    print(f"Expanded .nc file saved to {os.path.join(expanded_nc_save_path, file_save_name)}")

futures = []
with ProcessPoolExecutor(ncpus) as executor:
    LES_data_path = f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/LES/processed_data/all_data/{test_id}.csv'
    LES_data = pd.read_csv(LES_data_path)
    for habit in habits:
        for hour in hours:
            futures.append(executor.submit(calculate_RF_les, LES_data, test_id, habit, hour))

[f.result() for f in futures]