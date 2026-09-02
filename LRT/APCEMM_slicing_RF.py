#!/usr/bin/python
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import os
import gc
import xarray as xr
import sys

sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/LRT/')
import LRT_fxnlib as LRTlib
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as pipelinelib
from concurrent.futures import ProcessPoolExecutor

test_id = sys.argv[1]
habits = sys.argv[2].split("+")
hours = sys.argv[3].split("+")
initialization_type = "1000-bins-optimized"  # or "default-settings", points to different directories of APCEMM output data
OVERWRITE_EXISTING_RESULTS = True  # Set to True to force recalculation and overwriting of existing results

def _calculate_RF_EF_single(APCEMM_ds, test_id, habit, hour):
    """
    Inner function: receives an already-loaded ds_t list and computes
    RF/EF for one (habit, hour) combination.
    Accumulates results in plain lists; builds xarray only once at the end.
    """

    save_dir = f"/home/chinahg/GCresearch/contrailuncertainty/LRT/RF_EF_results/APCEMM/ef_final/{test_id}/"
    result_path = save_dir + f"APCEMM_{test_id}_{hour}h_{habit}.nc"

    if not OVERWRITE_EXISTING_RESULTS and os.path.exists(result_path):
        print(f"Results for {test_id} at hour {hour} with habit {habit} already exist. Skipping.")
        return

    num_timesteps = len(APCEMM_ds)
    time = np.linspace(0, (num_timesteps -1) * 10 / 60, num_timesteps)  # hours
    n_slices = 25

    # --- Accumulators (plain lists; no per-timestep xarray mutations) ---
    sw_rf_noon_list          = [] # Average SW RF from nanmean of all slice RFs in timestep time_idx
    lw_rf_noon_list          = [] # Average LW RF from nanmean of all slice RFs in timestep time_idx
    lw_rf_midnight_list      = []
    lw_slices_noon_list      = [] 
    sw_slices_noon_list      = []
    lw_slices_midnight_list  = []
    lw_midnight_AI_list      = [] # Average LW RF from using averaged inputs: One value of depth, radius, and IWC per timestep time_idx
    sw_noon_AI_list          = [] # Average SW RF from using averaged inputs: One value of depth, radius, and IWC per timestep time_idx
    lw_noon_AI_list          = [] # Average LW RF from using averaged inputs: One value of depth, radius, and IWC per timestep time_idx
    eff_radii_list           = []
    iwc_list                 = []
    depth_list               = []
    width_list               = []
    ef_slices_arr            = np.full((num_timesteps, n_slices), np.nan)
    ef_arr                   = np.full(num_timesteps, np.nan)
    ef_arr_AI                = np.full(num_timesteps, np.nan) # Average EF (cumsum) from using averaged inputs: One value of depth, radius, and IWC per timestep time_idx
    APCEMM_IWC_AI_list       = []
    APCEMM_depth_AI_list     = []
    APCEMM_radius_AI_list    = []

    for time_idx in range(num_timesteps):
        base_save_path = (
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/"
            f"APCEMM/slicing_results/{initialization_type}/{test_id}/RF"
        )
        input_save_path  = f"{base_save_path}/inputs/timestep_{time_idx}"
        output_save_path = f"{base_save_path}/outputs/timestep_{time_idx}"
        print(f"Running LRT calculations for: {test_id}  habit={habit}  hour={hour}  step={time_idx}\n")

        for path in [input_save_path, output_save_path]:
            os.makedirs(path, exist_ok=True)

        (APCEMM_IWC_slice_avg,
         APCEMM_radii_slice_avg,
         APCEMM_depth_slice_avg,
         APCEMM_IWC,
         APCEMM_radii,
         APCEMM_depth) = LRTlib.prepare_APCEMM_slices(APCEMM_ds, time_idx, n_slices)

        iwc_list.append(APCEMM_IWC_slice_avg)
        eff_radii_list.append(APCEMM_radii_slice_avg)
        depth_list.append(APCEMM_depth_slice_avg)

        contrail_age = (time_idx * 10 / 60) ## Hours

        # --- Averaged Inputs (Avg IWC, radius, and depth per timestep from original dataset) ---
        APCEMM_IWC_AI = np.nanmean(APCEMM_ds[time_idx]["IWC"].values) * 1e3 # [g/m3]
        APCEMM_depth_AI = np.nanmean(APCEMM_ds[time_idx]["depth"].values) # [m]
        APCEMM_radius_AI = np.nanmean(APCEMM_ds[time_idx]["Effective radius"]) * 1e6 # [um]

        APCEMM_IWC_AI_list.append(APCEMM_IWC_AI)
        APCEMM_depth_AI_list.append(APCEMM_depth_AI)
        APCEMM_radius_AI_list.append(APCEMM_radius_AI)

        print(f"{APCEMM_IWC_AI}")
        print(f"{APCEMM_depth_AI}")
        print(f"{APCEMM_radius_AI}")

        combined_iwc    = list(APCEMM_IWC_slice_avg) + [APCEMM_IWC_AI]
        combined_depth  = list(APCEMM_depth_slice_avg) + [APCEMM_depth_AI]
        combined_radius = list(APCEMM_radii_slice_avg) + [APCEMM_radius_AI]

        # Single call to libRadTran: n_slices profile slices + 1 averaged-input profile
        LW_RF_combined, SW_RF_combined = LRTlib.calculate_RF(
            n_slices + 1,
            contrail_age,
            combined_iwc,
            combined_depth,
            combined_radius,
            test_id, habit, hour,
            input_save_path,
        )

        # --- Build padded slice arrays (NaN-filled, no xarray overhead) ---
        lw_slice_arr = np.full(n_slices, np.nan)
        sw_slice_arr = np.full(n_slices, np.nan)

        lw_slice_arr[:n_slices] = LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[:n_slices]
        lw_AI = float(LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[n_slices])
        print(f"lw_AI = {lw_AI}")

        if SW_RF_combined is not None:
            sw_slice_arr[:n_slices] = SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[:n_slices]
            lw_slices_noon_list.append(lw_slice_arr)
            sw_slices_noon_list.append(sw_slice_arr)
            lw_slices_midnight_list.append(np.full(n_slices, np.nan))

            sw_AI = float(SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[n_slices])
            lw_noon_AI_list.append(lw_AI)
            sw_noon_AI_list.append(sw_AI)
            lw_midnight_AI_list.append(np.nan)
        else:
            lw_slices_midnight_list.append(lw_slice_arr)
            sw_slices_noon_list.append(np.full(n_slices, np.nan))
            lw_slices_noon_list.append(np.full(n_slices, np.nan))

            lw_noon_AI_list.append(np.nan)
            sw_noon_AI_list.append(np.nan)
            lw_midnight_AI_list.append(lw_AI)

        # --- Scalar RF means ---
        if SW_RF_combined is not None:
            lw_noon_mean = np.nanmean(lw_slice_arr) if not np.isnan(lw_slice_arr).all() else np.nan
            sw_noon_mean     = np.nanmean(sw_slice_arr) if not np.isnan(sw_slice_arr).all() else np.nan
            lw_midnight_mean = np.nan
        else:
            lw_midnight_mean = np.nanmean(lw_slice_arr) if not np.isnan(lw_slice_arr).all() else np.nan
            sw_noon_mean     = np.nan
            lw_noon_mean     = np.nan

        lw_rf_midnight_list.append(lw_midnight_mean)
        sw_rf_noon_list.append(sw_noon_mean)
        lw_rf_noon_list.append(lw_noon_mean)

        # --- Width (needed for EF) ---
        width_list.append(APCEMM_ds[time_idx]["width"].values[0])

        # Free LRT result dicts explicitly
        del LW_RF_combined, SW_RF_combined, lw_slice_arr, sw_slice_arr

    # --- Energy forcing (uses full time series) ---
    width_arr = np.asarray(width_list)
    slice_widths = width_arr/n_slices
    delta_t = 10 * 60 # min to seconds, all timesteps are equal

    # Calculate the net RF
    if hour != '0': # It is noon
        rf_net_slices = np.array(lw_slices_noon_list) + np.array(sw_slices_noon_list)
        rf_net_AI = np.array(lw_noon_AI_list) + np.array(sw_noon_AI_list)
    else: # It is midnight
        rf_net_slices = np.array(lw_slices_midnight_list)
        rf_net_AI = np.array(lw_midnight_AI_list)

    for time_idx in range(num_timesteps):
        ef_slices_arr[time_idx, :] = rf_net_slices[time_idx, :] * slice_widths[time_idx] * delta_t # [W/m²] * [m] * [s] = [J/m]
        ef_arr[time_idx] = np.nansum(ef_slices_arr[time_idx, :])

        ef_arr_AI[time_idx] = rf_net_AI[time_idx] * width_arr[time_idx] * delta_t

    ef_cumsum = np.nancumsum(ef_arr) # Want the cumulative sum
    ef_cumsum_AI = np.nancumsum(ef_arr_AI)

    print(f"ef_cumsum_AI = {ef_cumsum_AI}")

    # --- Build RF/EF dataset ONCE ---
    ds_out = xr.Dataset(
        {   "LW RF slices midnight":    (["t", "slice"], np.vstack(lw_slices_midnight_list)),
            "SW RF slices noon":        (["t", "slice"], np.vstack(sw_slices_noon_list)),
            "LW RF slices noon":        (["t", "slice"], np.vstack(lw_slices_noon_list)), 
            "IWC slices":               (["t", "slice"], np.vstack(iwc_list)),
            "Effective radius slices":  (["t", "slice"], np.vstack(eff_radii_list)),
            "Depth slices":             (["t", "slice"], np.vstack(depth_list)), 
            "Energy forcing slices":    (["t", "slice"], ef_slices_arr), 
            "LW RF midnight":           ("t", np.asarray(lw_rf_midnight_list)),
            "SW RF noon":               ("t", np.asarray(sw_rf_noon_list)),
            "LW RF noon":               ("t", np.asarray(lw_rf_noon_list)),
            "Energy Forcing":           ("t", np.asarray(ef_cumsum)),
            "LW RF midnight AI":        ("t", np.asarray(lw_midnight_AI_list)),
            "SW RF noon AI":            ("t", np.asarray(sw_noon_AI_list)),
            "LW RF noon AI":            ("t", np.asarray(lw_noon_AI_list)),
            "IWC AI":                   ("t", np.asarray(APCEMM_IWC_AI_list)),
            "Effective radius AI":      ("t", np.asarray(APCEMM_radius_AI_list)),
            "Depth AI":                 ("t", np.asarray(APCEMM_depth_AI_list)),
            "Energy Forcing AI":        ("t", np.asarray(ef_cumsum_AI)),
            "Habit":                    (habit),
            "Hour":                     (hour),  
        },
        coords={"t": time, "slice": np.arange(n_slices)},
    )

    # Free accumulator lists now that ds_out holds the data
    del lw_slices_midnight_list, sw_slices_noon_list, lw_slices_noon_list
    del iwc_list, eff_radii_list, depth_list, ef_slices_arr
    del lw_rf_midnight_list, sw_rf_noon_list, lw_rf_noon_list, ef_cumsum
    del lw_midnight_AI_list, sw_noon_AI_list, lw_noon_AI_list, ef_cumsum_AI
    del ef_arr, width_list, ef_arr_AI
    gc.collect()

    # --- Concatenate original APCEMM timesteps ONCE ---
    # after concat, force the t coordinate to exactly match ds_out
    APCEMM_ds_combined = xr.concat(APCEMM_ds, dim="t").assign_coords(t=time)

    print(f"ds EF AI = {ds_out["Energy Forcing AI"].values}")

    # Single merge of two full datasets — no per-timestep intermediates
    APCEMM_ds_combined = xr.merge([APCEMM_ds_combined, ds_out])
    print("APCEMM_ds_combined t coord:", APCEMM_ds_combined["t"].values)
    print("ds_out t coord:", ds_out["t"].values)
    del ds_out

    # --- Save ---
    os.makedirs(save_dir, exist_ok=True)
    APCEMM_ds_combined.to_netcdf(result_path)
    print(f"Saved: {result_path}")

    del APCEMM_ds_combined
    gc.collect()


def process_test_id(test_id, habit, hour):
    """
    Loads APCEMM data ONCE per test_id, then iterates over all
    (habit, hour) combinations — avoiding redundant heavyweight loads
    when the same test_id is shared across workers.
    """
    APCEMM_data_path = (
        f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/'
        f'APCEMM/{initialization_type}/{test_id}/outputs'
    )
    print(f"Loading APCEMM data for {test_id} from {APCEMM_data_path}")
    APCEMM_data = pipelinelib.read_apcemm_data(APCEMM_data_path)
    APCEMM_ds = APCEMM_data.ds_t  # list of per-timestep datasets

    try:
        _calculate_RF_EF_single(APCEMM_ds, test_id, habit, hour)
    finally:
        # Always free the large dataset, even if a (habit, hour) combo fails
        del APCEMM_data, APCEMM_ds
        gc.collect()


# --- Parallelise over test_ids only (data loaded once per worker) ---
ncpus = int(sys.argv[4])  # 1 cpu per (habit, hour) combo
with ProcessPoolExecutor(max_workers=ncpus) as executor:
    futures = [executor.submit(process_test_id, test_id, habit, hour) for hour in hours for habit in habits]

# Propagate any exceptions from workers
for f in futures:
    f.result()