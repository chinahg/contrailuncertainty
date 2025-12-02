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

test_id = '110T218L25'
model_type = ['APCEMM']  # 'CoCiP', 'APCEMM', 'LLES'
base_save_path = "/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/testing/IWC_RF_slicing/slicing_1" # Where to save the input and output files
print(f"Running LRT calculations for: {test_id}")

### GHM ###
habit_type = ["droxtal"] # yang-2013 for droxtal and solid-column, baum-2005a for ghm
hours = ["12"] # Midnight and/or Noon

# Read in APCEMM data
APCEMM_data = pipelinelib.read_apcemm_data(
    f'/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/{test_id}/outputs'
)

APCEMM_ds = APCEMM_data.ds_t

n_bins = 38
timesteps = [6] #, 6, 18, 30, 72] # 10 min, 1 hr, 3 hr, 5 hr, 12 hr

# number of x coordinates at each timestep
num_slices = np.zeros(len(timesteps), dtype=int)
for i in range(len(timesteps)):
    time_idx = timesteps[i]
    num_slices[i] = APCEMM_ds[time_idx]['x'].values.size

APCEMM_depth = np.zeros(len(APCEMM_ds))
APCEMM_radii = np.zeros((n_bins, len(APCEMM_ds)))
APCEMM_IWC_avg = np.zeros((len(timesteps), num_slices.max()))
APCEMM_radii = np.zeros((len(timesteps), num_slices.max()))
print(num_slices)
print(np.shape(APCEMM_ds[time_idx]['Effective radius'].values * 1e6))

for i in range(len(timesteps)):
    time_idx = timesteps[i]
    for j in range(num_slices[i]):
        print(f"Processing time step {i}, slice {j}")
        APCEMM_depth[i] = APCEMM_ds[time_idx]['depth'].values.item()

        # Average IWC, masking out IWC values lower than 0.3e-5 g/m³ to avoid skewing the average
        APCEMM_IWC = APCEMM_ds[time_idx]['IWC'].values * 1e3  # [kg/m³] → [g/m³]
        APCEMM_IWC_masked = np.where(APCEMM_IWC > 0.1e-8, APCEMM_IWC, np.nan)
        APCEMM_IWC_avg[i][j] = np.nanmean(APCEMM_IWC_masked[:,j]) # IWC at time i and slice j
    
        # Effective radius at time i and slice j, averaged over the slice
        APCEMM_radii[i][j] = np.mean(APCEMM_ds[time_idx]['Effective radius'].values[:,j] * 1e6)  # [µm]
print("IWC at slice 1, 50, 100: ", APCEMM_IWC_avg[:,1], APCEMM_IWC_avg[:,50], APCEMM_IWC_avg[:,100])
APCEMM_times = np.array(timesteps) * 10 / 60  # hours (10 min/step)

# LW and SW for APCEMM
for j in range(num_slices.max()):
    for habit in habit_type:
        for hour in hours:
            ice_in_path = f"{base_save_path}/ice_in_{habit}_{hour}h_{test_id}_slice{j}.in"
            thermal_cloud_path = f"{base_save_path}/thermal_cloud_{habit}_{hour}h_{test_id}_slice{j}.in"
            thermal_clear_path = f"{base_save_path}/thermal_clear_{habit}_{hour}h_{test_id}_slice{j}.in"
            solar_cloud_path = f"{base_save_path}/solar_cloud_{habit}_{hour}h_{test_id}_slice{j}.in"
            solar_clear_path = f"{base_save_path}/solar_clear_{habit}_{hour}h_{test_id}_slice{j}.in"
            
            LW_RF_APCEMM, SW_RF_APCEMM = LRTlib.calculate_RF(APCEMM_times, APCEMM_IWC_avg[:,j], APCEMM_depth, APCEMM_radii[:,j], habit, hour, ice_in_path, thermal_cloud_path, thermal_clear_path, solar_cloud_path, solar_clear_path)
            SW_RF_APCEMM.to_csv(
                f"{base_save_path}/SW_RF_APCEMM_{habit}_{hour}h_IWC_{test_id}_slice{j}.csv",
                index=False
            )
            print(f"SW RF results saved to .../SW_RF_APCEMM_{habit}_{hour}h_IWC_{test_id}_slice{j}.csv")
            LW_RF_APCEMM.to_csv(
                f"{base_save_path}/LW_RF_APCEMM_{habit}_{hour}h_IWC_{test_id}_slice{j}.csv",
                index=False
            )
            print(f"LW RF results saved to .../LW_RF_APCEMM_{habit}_{hour}h_IWC_{test_id}_slice{j}.csv")