#!/usr/bin/python
import warnings
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
habits   = sys.argv[2].split("+")
hours    = sys.argv[3].split("+")
ncpus    = int(sys.argv[4])
print(f"Received arguments: test_id={test_id}, habits={habits}, hours={hours}, ncpus={ncpus}")


def _calculate_RF_cocip_single(CoCiP_subsampled, test_id, habit, hour):
    """
    Inner function: receives an already-open dataset and computes RF/EF
    for one (habit, hour) combination. Does not open or close the dataset.
    """
    save_path = (
        f"/home/chinahg/GCresearch/contrailuncertainty/LRT/RF_EF_results/"
        f"CoCiP/{test_id}/CoCiP_{test_id}_{hour}h_{habit}.nc"
    )
    
    # if os.path.exists(save_path):
    #     print(f"Results for {test_id} hour={hour} habit={habit} already exist. Skipping.\n")
    #     return

    tod = "midnight" if hour == "0" else "noon"
    print(f"Processing CoCiP data for {test_id} at hour {hour} with habit {habit} ({tod})\n")

    num_timesteps          = len(CoCiP_subsampled['age_hours'].values) # Process at 10 minute increments
    print(f"number of timesteps:{num_timesteps}")
    n_slices = int(25)
    slice_coord            = np.arange(n_slices)

    print(f"Total number of timesteps in dataset: {num_timesteps}\n")

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
        print(f"--- Processing CoCiP timestep {time_idx}"
              f"(age_hours={CoCiP_subsampled['age_hours'].values[time_idx]}) ---\n")

        base_save_path   = (
            f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/"
            f"CoCiP/RF_EF_results/slicing_results/{test_id}/RF"
        )
        input_save_path  = f"{base_save_path}/inputs/timestep_{time_idx}"
        output_save_path = f"{base_save_path}/outputs/timestep_{time_idx}"
        for path in [input_save_path, output_save_path]:
            os.makedirs(path, exist_ok=True)

        # IWC [g/m3], radii [um], depth [m]
        CoCiP_IWC_slice_avg, CoCiP_radii_slice_avg, CoCiP_depth_slice_avg = \
            LRTlib.prepare_cocip_slices(CoCiP_subsampled, time_idx, n_slices)

        contrail_age = CoCiP_subsampled['age_hours'].values[time_idx] # hours

        # Store slice physical quantities
        iwc_slices[time_idx,   :n_slices] = CoCiP_IWC_slice_avg[:n_slices]
        radii_slices[time_idx, :n_slices] = CoCiP_radii_slice_avg[:n_slices]
        depth_slices[time_idx, :n_slices] = CoCiP_depth_slice_avg[:n_slices]

        # --- Averaged Inputs (Avg IWC, radius, and depth per timestep) ---
        IWC_AI = CoCiP_subsampled['iwc'][time_idx].values * CoCiP_subsampled["rho_air"][time_idx].values * 1e3 # [kg ice/kg air] * [kg air/m3] * 1e3 = [g ice/m3]
        depth_AI = CoCiP_subsampled['depth'][time_idx].values # [m]
        radius_AI = CoCiP_subsampled['r_ice_vol'][time_idx].values * 1e6 # radius in micron [um]

        iwc_AI_list[time_idx] = IWC_AI
        depth_AI_list[time_idx] = depth_AI
        radius_AI_list[time_idx] = radius_AI

        print(f"IWC_AI = {IWC_AI:.2f}\n depth_AI = {depth_AI:.2f}\n radius_AI = {radius_AI:.2f}")

        combined_iwc    = list(CoCiP_IWC_slice_avg) + [IWC_AI]
        combined_depth  = list(CoCiP_depth_slice_avg) + [depth_AI]
        combined_radius = list(CoCiP_radii_slice_avg) + [radius_AI]

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

        lw_vals = LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[:n_slices]
        lw_AI = float(LW_RF_combined['LW_Radiative_Forcing_W_m2'].values[n_slices])

        if SW_RF_combined is not None:
            sw_vals = SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[:n_slices]
            sw_slices_noon_list[time_idx, :len(sw_vals)] = sw_vals
            lw_slices_noon_list[time_idx, :len(lw_vals)] = lw_vals
            del sw_vals

            sw_AI = float(SW_RF_combined['SW_Radiative_Forcing_W_m2'].values[n_slices])
            print(f"AI LW RF = {lw_AI:.2f}")
            lw_noon_AI_list[time_idx] = lw_AI
            sw_noon_AI_list[time_idx] = sw_AI
            lw_midnight_AI_list[time_idx] = np.nan
        else:
            lw_slices_midnight_list[time_idx, :len(lw_vals)] = lw_vals

            print(f"AI LW RF = {lw_AI:.2f}")
            lw_noon_AI_list[time_idx] = np.nan
            sw_noon_AI_list[time_idx] = np.nan
            lw_midnight_AI_list[time_idx] = lw_AI

        # Average across slices inline — avoids a second full loop later
        lw_rf_midnight_list[time_idx]  = np.nanmean(lw_slices_midnight_list[time_idx])
        sw_rf_noon_list[time_idx] = np.nanmean(sw_slices_noon_list[time_idx])
        lw_rf_noon_list[time_idx] = np.nanmean(lw_slices_noon_list[time_idx])

        # Free LRT result objects as soon as we're done with them
        del LW_RF_combined, SW_RF_combined, lw_vals

    print("-------------------------------------------------")
    print("--- libRadtran calculations complete. Post-processing... ---")
    print("-------------------------------------------------")

    # Energy forcing
    # Calculate total RF per slice
    if hour != "0": # If it is noon
        rf_net_slices = lw_slices_noon_list + sw_slices_noon_list
        rf_net_AI = lw_noon_AI_list + sw_noon_AI_list
    else: # It is midnight
        rf_net_slices = lw_slices_midnight_list
        rf_net_AI = lw_midnight_AI_list

    widths = CoCiP_subsampled["width"].values
    slice_widths = widths/n_slices
    delta_t = 10 * 60 # min to seconds, all timesteps are equal

    for time_idx in range(num_timesteps):
        ef_slices[time_idx, :] = rf_net_slices[time_idx, :] * slice_widths[time_idx] * delta_t  # [W/m²] * [m] * [s] = [J/m]
        ef[time_idx] = np.nansum(ef_slices[time_idx, :])

        ef_AI[time_idx] = rf_net_AI[time_idx] * widths[time_idx] * delta_t  # [W/m²] * [m] * [s] = [J/m]

    ef_cumsum = np.nancumsum(ef) # Want the cumulative sum
    ef_cumsum_AI = np.nancumsum(ef_AI)

    # Build result dataset and merge with original — all in one step
    results_ds = xr.Dataset(
        {
            "LW RF slices midnight":    (["t", "slice"], lw_slices_midnight_list),
            "SW RF slices noon":        (["t", "slice"], sw_slices_noon_list),
            "LW RF slices noon":        (["t", "slice"], lw_slices_noon_list), 
            "IWC slices":               (["t", "slice"], iwc_slices),
            "Effective radius slices":  (["t", "slice"], radii_slices),
            "Depth slices":             (["t", "slice"], depth_slices), 
            "Energy forcing slices":    (["t", "slice"], ef_slices), 
            "LW RF midnight":           ("t", lw_rf_midnight_list),
            "SW RF noon":               ("t", sw_rf_noon_list),
            "LW RF noon":               ("t", lw_rf_noon_list),
            "Energy Forcing":           ("t", ef_cumsum),
            "LW RF midnight AI":        ("t", lw_midnight_AI_list),
            "SW RF noon AI":            ("t", sw_noon_AI_list),
            "LW RF noon AI":            ("t", lw_noon_AI_list),
            "IWC AI":                   ("t", iwc_AI_list),
            "Effective radius AI":      ("t", radius_AI_list),
            "Depth AI":                 ("t", depth_AI_list),
            "Energy Forcing AI":        ("t", ef_cumsum_AI),
            "Habit":                    (habit),
            "Hour":                     (hour),
        },
        coords={"t": CoCiP_subsampled["t"].values, "slice": slice_coord},
    )

    # Free large arrays before the merge to avoid holding two copies
    del lw_slices_midnight_list, sw_slices_noon_list, lw_slices_noon_list
    del iwc_slices, radii_slices, depth_slices, ef_slices
    del lw_rf_midnight_list, sw_rf_noon_list, lw_rf_noon_list, ef_cumsum
    del lw_midnight_AI_list, sw_noon_AI_list, lw_noon_AI_list, ef_cumsum_AI
    del rf_net_slices

    CoCiP_ds_expanded = xr.merge(
        [CoCiP_subsampled.assign_coords(slice=slice_coord),
         results_ds]
    )
    del results_ds

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    CoCiP_ds_expanded.to_netcdf(save_path)
    print(f"Saved: {save_path}\n")

    del CoCiP_ds_expanded
    gc.collect()


def process_test_id(test_id, hour, habit):
    """
    Opens the CoCiP dataset ONCE per (test_id, hour) pair — the file
    differs between midnight and noon — then loops over habits inside
    the same worker, avoiding redundant file opens.
    """
    tod = "midnight" if hour == "0" else "noon"
    data_path = (
        f'/home/chinahg/GCresearch/contrailuncertainty/'
        f'APCEMM_vs_CoCiP_vs_LES/CoCiP/{test_id}/{test_id}-bypass_{tod}.nc'
    )
    print(f"Loading CoCiP dataset for {test_id} ({tod}) from {data_path}\n")

    # Use a context manager so the file handle is always released
    with xr.open_dataset(data_path, decode_times=False) as CoCiP_ds:
        # Keep every 10th index (0, 10, 20, ... 60)
        age_hours = CoCiP_ds["age_hours"].values

        max_age = age_hours.max()
        target_ages = np.arange(0, max_age + 10/60, 10/60)  # every 10 min in hours

        # For each target age, find the index position of the nearest age_hours value
        nearest_positions = np.array([
            np.argmin(np.abs(age_hours - t)) for t in target_ages
        ])
    
        CoCiP_subsampled = CoCiP_ds.isel(index=nearest_positions)
    
        # Rename dimension and relabel to clean integers
        CoCiP_subsampled = CoCiP_subsampled.rename({"index": "t"})
        CoCiP_subsampled = CoCiP_subsampled.assign_coords(t=np.arange(len(CoCiP_subsampled["t"])))
    
        try:
            _calculate_RF_cocip_single(CoCiP_subsampled, test_id, habit, hour)
        finally:
            # gc inside the finally so memory is freed before the next hour
            gc.collect()


# Parallelise over test_ids only; habits/hours are looped inside each worker
with ProcessPoolExecutor(max_workers=ncpus) as executor:
    futures = [executor.submit(process_test_id, test_id, hour, habit) for hour in hours for habit in habits]

# Propagate any worker exceptions
for f in futures:
    f.result()