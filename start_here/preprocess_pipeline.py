### Pipeline Step 2a: Convert GRIB to NetCDF, calculate RHi

##############################################################################################################################################
# Imports
import pipeline_fxn_lib as lib
from download_pipeline.py import files2convert, GRUAN_date_sites
import numpy as np
import subprocess
import time
##############################################################################################################################################

# Where did you save the ERA5 data?
data_dir = '/home/chinahg/GCresearch/ERA5_downloads/'

# Convert any GRIB ERA5 files to .nc
lib.convert_files(data_dir, files2convert)

### Pipeline Step 2b: Calculate RHi and RHw
# Remake the list of matching files after conversion
matching_data, matching_files, files2convert, files2download = lib.match_files(GRUAN_date_sites) # Format for matching_data is [ERA5 file name, GRUAN site name, GRUAN datetime object]

# Calculate RHi and RHw for all the ERA5 files and add as a new variable
lib.add_RHi_RHw(matching_files)

### Pipeline Step 2c: Consolidate ERA5 data and GRUAN data into a single file
# Base directories
gruan_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/'
era5_base_dir = '/home/chinahg/GCresearch/ERA5_downloads/'

# Arrays to store the paths
gruan_file_paths = []
era5_file_paths = []

gruan_file_paths, era5_file_paths = lib.construct_complementary_paths(gruan_base_dir, era5_base_dir)

# Define the start and end file indices for processing, as ideally we will split the files into batches for slurm
files_per_batch = 1000  # Number of files to process per slurm batch
batches = len(gruan_file_paths) // files_per_batch + (1 if len(gruan_file_paths) % files_per_batch != 0 else 0)
start_indices = len(batches)*np.NaN
end_indices = len(batches)*np.NaN

for i in range(batches):
    start_indices[i] = i * files_per_batch
    end_indices[i] = start_indices[i] + files_per_batch
    if end_indices[i] > len(gruan_file_paths):
        end_indices[i] = len(gruan_file_paths)
    
for i in range(batches):
    start_index = start_indices[i]
    end_index = end_indices[i]
    batch = i + 1  # Batch number for slurm job
    subprocess.run(["sbatch", "/home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.sh"])

    # Wait 5 seconds before starting the next job
    time.sleep(5)

### WAIT FOR THE JOBS TO FINISH THEN GO TO input_dist_pipeline.py
