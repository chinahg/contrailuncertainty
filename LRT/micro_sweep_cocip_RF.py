from pathlib import Path
import xarray as xr
import time
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/LRT/')
import LRT_fxnlib as LRTlib
from concurrent.futures import ProcessPoolExecutor

test_id = sys.argv[1]  # Specify the test ID to process if only want to run for that case (ex. '110T218L25' OR 'None' to run all)
base_dir = Path(f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/")

# Recursively find all .nc files
nc_files = sorted([str(p) for p in base_dir.rglob("1??T???L25/??????????-bypass.nc")])

if not nc_files:
    print(f"No .nc files found under {base_dir}")
else:
    print(f"Found {len(nc_files)} .nc files. Opening with xarray.open_mfdataset...")
    print("Opening files individually into a list 'datasets'.")
    datasets = [xr.open_dataset(f) for f in nc_files]
    ds = None

habits = ["ghm", "rough-aggregate", "solid-column"] # yang-2013 for droxtal and solid-column, baum-2005a for ghm
hours = ["0", "12"] # Midnight and Noon

print("Processing CoCiP data...")

def calculate_RF_cocip(ds, f, habit_type, hour):
    print(f"Calculating RF for file: {f}")
    # stem = Path(f).stem
    stem = Path(f).parts[-2]  # Get the test_id from the parent directory name

    cocip_ds = ds
    cocip_times = cocip_ds['age_hours'].values
    cocip_radii = cocip_ds['r_ice_vol'].values * 1e6  # [µm]
    cocip_depth = cocip_ds['depth'].values  # [m]
    cocip_IWC = cocip_ds['iwc'].values * cocip_ds['rho_air'] *1e3 # [kg ice/ kg air] * [kg air/m3] * 1e3 # Convert to [g/m^3]
    save_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/RF_EF_results/{stem}/RF"

    # LW and SW for CoCiP
    # # Skip calculating RF if results already exist
    # pattern = f"{stem}_??_{habit_type}_{hour}h_IWC.csv"
    # output_path = Path(save_path) / "outputs"
    # matches = list(output_path.glob(pattern))

    # if matches:
    #     print(f"RF results for {stem} at {hour}h for {habit_type} already exist. Skipping calculation.\n")
    #     return

    print(f"Processing file: {stem}")
    print(f"Calculating RF for radiative time of {hour}h...")

    # Define path to ice.in file that will be updated during libRadtrans calls
    ice_in_path = f"{save_path}/inputs/ice_in_{stem}_{habit_type}_{hour}h.in"
    thermal_cloud_path = f"{save_path}/inputs/thermal_cloud_{stem}_{habit_type}_{hour}h.in"
    thermal_clear_path = f"{save_path}/inputs/thermal_clear_{stem}_{habit_type}_{hour}h.in"
    solar_cloud_path = f"{save_path}/inputs/solar_cloud_{stem}_{habit_type}_{hour}h.in"
    solar_clear_path = f"{save_path}/inputs/solar_clear_{stem}_{habit_type}_{hour}h.in"
    LW_RF_cocip, SW_RF_cocip = LRTlib.calculate_RF(cocip_times, 
                                                   cocip_IWC, 
                                                   cocip_depth, 
                                                   cocip_radii, 
                                                   habit_type, 
                                                   hour, 
                                                   ice_in_path,
                                                   thermal_cloud_path,
                                                   thermal_clear_path,
                                                   solar_cloud_path,
                                                   solar_clear_path)
    
    print(f"Calculated LW and SW RF for CoCiP at {hour}h.")

    if SW_RF_cocip is not None: # Returns None if hour == "0". AKA at midnight, no SW calculations are done.
        SW_RF_cocip.to_csv(
            f"{save_path}/outputs/{stem}_SW_{habit_type}_{hour}h_IWC.csv",
            index=False
        )
        print(f"SW RF results saved to {save_path}/outputs/{stem}_SW_{habit_type}_{hour}h_IWC.csv")
    LW_RF_cocip.to_csv(
        f"{save_path}/outputs/{stem}_LW_{habit_type}_{hour}h_IWC.csv",
        index=False
    )
    print(f"LW RF results saved to {save_path}/outputs/{stem}_LW_{habit_type}_{hour}h_IWC.csv")

ncpus = 6
futures = []
with ProcessPoolExecutor(ncpus) as executor:
    for ds, f in zip(datasets, nc_files):
        if test_id is not None and Path(f).parts[-2] != test_id:
                    continue
        print(f"Submitting job for file: {f}")
        for habit_type in habits:
            for hour in hours:
                futures.append(executor.submit(calculate_RF_cocip, ds, f, habit_type, hour))

[f.result() for f in futures]