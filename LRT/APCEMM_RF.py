#!/usr/bin/python
import matplotlib.pyplot as plt
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

test_ids = [str(sys.argv[1])] #['110T205L25', '110T218L25', '110T225L25', '130T205L25', '130T218L25', '130T225L25'] # 110% RHi, 205K, 2.5K/km
habits = ["rough-aggregate", "solid-column", "ghm"] # baum-2005a for ghm
hours = ["0", "12"] # Midnight (0) and/or Noon (12)

def calculate_RF_APCEMM(test_id, habit_type, hour):
    base_save_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/RF_EF_results/{test_id}/RF/" # Where to save the input and output files
    input_save_path = f"{base_save_path}/inputs"
    output_save_path = f"{base_save_path}/outputs"
    APCEMM_data_path = f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/outputs'
    print(f"Running LRT calculations for: {test_id}")
    print(f"APCEMM data location: {APCEMM_data_path}")
    print(f"Results save path: {output_save_path}")

    results_path = f"{output_save_path}/LW_RF_APCEMM_{habit_type}_{hour}h_IWC_{test_id}.csv"
    if os.path.exists(results_path):
        print(f"RF results for {test_id} at {hour}h for {habit_type} already exist. Skipping calculation.")
        return

    # Read in APCEMM data
    APCEMM_data = pipelinelib.read_apcemm_data(APCEMM_data_path)

    APCEMM_ds = APCEMM_data.ds_t
    n_bins = 38 # Number of radii bins in APCEMM
    APCEMM_depth = np.zeros(len(APCEMM_ds))
    APCEMM_radii_bins = APCEMM_ds[0]['r'].values * 1e6  # [µm] Bin centers
    APCEMM_radii_counts = np.zeros((n_bins, len(APCEMM_ds))) # Number of crystals in each bin
    APCEMM_IWC_avg = np.zeros(len(APCEMM_ds)) # Average IWC per time step

    timesteps = len(APCEMM_ds)
    for i in range(timesteps):
        APCEMM_depth[i] = APCEMM_ds[i]['depth'].values.item()

        # Count number of crystals in each radii bin
        APCEMM_radii_counts[:, i] = APCEMM_ds[i]["Overall size distribution"].values

        # Average IWC, masking out IWC values lower than 0.3e-5 g/m³ to avoid skewing the average
        APCEMM_IWC = APCEMM_ds[i]['IWC'].values * 1e3  # [kg/m³] → [g/m³]
        APCEMM_IWC_masked = np.where(APCEMM_IWC > 0.3e-5, APCEMM_IWC, np.nan)
        APCEMM_IWC_avg[i] = np.nanmean(APCEMM_IWC_masked)

    APCEMM_times = np.arange(timesteps) * 10 / 60  # hours (10 min/step)

    # LW and SW for APCEMM
    APCEMM_radii = np.zeros(len(APCEMM_times))
    # Calculate the mean radius for each time step
    for p in range(len(APCEMM_times)):
        APCEMM_radii[p] = LRTlib.calculate_sauter_mean(APCEMM_radii_bins, APCEMM_radii_counts[:, p])

    ice_in_path = f"{input_save_path}/ice_in_{habit_type}_{hour}h_{test_id}.in"
    thermal_cloud_path = f"{input_save_path}/thermal_cloud_{habit_type}_{hour}h_{test_id}.in"
    thermal_clear_path = f"{input_save_path}/thermal_clear_{habit_type}_{hour}h_{test_id}.in"
    solar_cloud_path = f"{input_save_path}/solar_cloud_{habit_type}_{hour}h_{test_id}.in"
    solar_clear_path = f"{input_save_path}/solar_clear_{habit_type}_{hour}h_{test_id}.in"
    LW_RF_APCEMM, SW_RF_APCEMM = LRTlib.calculate_RF(APCEMM_times, APCEMM_IWC_avg, APCEMM_depth, APCEMM_radii, habit_type, hour, ice_in_path, thermal_cloud_path, thermal_clear_path, solar_cloud_path, solar_clear_path)
    
    if SW_RF_APCEMM is not None: # Returns None if hour == "0". AKA at midnight, no SW calculations are done.
        SW_RF_APCEMM.to_csv(
            f"{output_save_path}/SW_RF_APCEMM_{habit_type}_{hour}h_IWC_{test_id}.csv",
            index=False
        )
        print(f"SW RF results saved to .../SW_RF_APCEMM_{habit_type}_{hour}h_IWC_{test_id}.csv")

    LW_RF_APCEMM.to_csv(
        f"{output_save_path}/LW_RF_APCEMM_{habit_type}_{hour}h_IWC_{test_id}.csv",
        index=False
    )
    print(f"LW RF results saved to .../LW_RF_APCEMM_{habit_type}_{hour}h_IWC_{test_id}.csv")

ncpus = 6
futures = []
with ProcessPoolExecutor(ncpus) as executor:
    for test_id in test_ids:
        for habit_type in habits:
            for hour in hours:
                futures.append(executor.submit(calculate_RF_APCEMM, test_id, habit_type, hour))

[f.result() for f in futures]