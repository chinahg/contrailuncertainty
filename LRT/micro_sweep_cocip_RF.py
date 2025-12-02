from pathlib import Path
import xarray as xr
import time
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/LRT/')
import LRT_fxnlib as LRTlib
from concurrent.futures import ProcessPoolExecutor

start_time = time.time()

base_dir = Path("/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/EInvpm")

# Recursively find all .nc files
nc_files = sorted([str(p) for p in base_dir.rglob("*.nc")])

if not nc_files:
    print(f"No .nc files found under {base_dir}")
else:
    print(f"Found {len(nc_files)} .nc files. Opening with xarray.open_mfdataset...")
    print("Opening files individually into a list 'datasets'.")
    datasets = [xr.open_dataset(f) for f in nc_files]
    ds = None

model_type = ['CoCiP']

### GHM ###
habits = ["ghm", "droxtal", "solid_column"] # yang-2013 for droxtal and solid-column, baum-2005a for ghm
hours = ["0", "12"] # Midnight and Noon

print("Processing CoCiP data...")

def calculate_RF_cocip(ds, f, habit_type, hour):
    stem = Path(f).stem

    cocip_ds = ds
    cocip_times = cocip_ds['age_hours'].values
    cocip_radii = cocip_ds['r_ice_vol'].values * 1e6  # [µm]
    cocip_depth = cocip_ds['depth'].values  # [m]
    cocip_IWC = cocip_ds['iwc'].values * cocip_ds['rho_air'] *1e3 # [kg ice/ kg air] * [kg air/m3] * 1e3 # Convert to [g/m^3]
    save_path = "/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results"

    # LW and SW for CoCiP

    # Skip calculating RF if results already exist
    pattern = f"{stem}_??_{habit_type}_{hour}h_IWC.csv"
    matches = list(Path(save_path).glob(pattern))

    if matches:
        print(f"RF results for {stem} at {hour}h for {habit_type} already exist. Skipping calculation.")
        return

    print(f"Processing file: {stem}")
    print(f"Calculating RF for radiative time of {hour}h...")

    # Define path to ice.in file that will be updated during libRadtrans calls
    ice_in_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/input_files/ice_in_{stem}_{habit_type}_{hour}h.in"
    thermal_cloud_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/input_files/thermal_cloud_{stem}_{habit_type}_{hour}h.in"
    thermal_clear_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/input_files/thermal_clear_{stem}_{habit_type}_{hour}h.in"
    solar_cloud_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/input_files/solar_cloud_{stem}_{habit_type}_{hour}h.in"
    solar_clear_path = f"/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/input_files/solar_clear_{stem}_{habit_type}_{hour}h.in"
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

    SW_RF_cocip.to_csv(
        f"{save_path}/{stem}_SW_{habit_type}_{hour}h_IWC.csv",
        index=False
    )
    print(f"SW RF results saved to .../{stem}_SW_RF_cocip_{habit_type}_{hour}h_IWC.csv")
    LW_RF_cocip.to_csv(
        f"{save_path}/{stem}_LW_{habit_type}_{hour}h_IWC.csv",
        index=False
    )
    print(f"LW RF results saved to .../{stem}_LW_RF_cocip_{habit_type}_{hour}h_IWC.csv")

ncpus = 20
futures = []
with ProcessPoolExecutor(ncpus) as executor:
    for ds, f in zip(datasets, nc_files):
        for habit_type in habits:
            for hour in hours:
                futures.append(executor.submit(calculate_RF_cocip, ds, f, habit_type, hour))

[f.result() for f in futures]