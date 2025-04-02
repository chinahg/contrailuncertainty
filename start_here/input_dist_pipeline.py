### Pipeline Step 3a: Create APCEMM meteorological input files
# This script samples fluctuations in relative humidity values, updates the dataset with these sampled values, and sets initial meteorological conditions for contrail formation. 
# It saves the modified dataset to new NetCDF files to be used as meteorological input files to APCEMM.

##############################################################################################################################################
# Imports
import pipeline_fxn_lib as lib
import os
import subprocess
##############################################################################################################################################

# Define the base APCEMM meteorlogical file path
input_file_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/BASE_APCEMM_met.nc'

set_type = "training" # Is this data for validating or training the model? Choose "validation" or "training".

# Define the meteorological APCEMM file dimensions
altitudes = 125
timesteps = 24

# Initial condition distribution details
mean_norm = 0
std_norm = 0.5
IC_scaled_mean = 117

# RHi temporal distribution details
std_norm_time = mean_ensemble_RHi
mean_norm_time = 0

num_met_files = 20 # Number of met and YAML files to generate
test_num = 3 # Test number the met and YAML files are associated with

# Create the necessary directories if they don't exist
# Create the base directory for the specified test number
base_dir = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{test_num}"

# Create the necessary subdirectories
os.makedirs(base_dir, exist_ok=False)
os.makedirs(f"{base_dir}/inputs/{set_type}", exist_ok=False)

# Create the output directories for each run
for i in range(1, num_met_files + 1):
    os.makedirs(f"{base_dir}/outputs/{set_type}/test_{test_num}_run_{i}", exist_ok=False)

# Generate the met files
lib.generate_apcemm_input_files(input_file_path, num_met_files, test_num, set_type)

# Run batches of APCEMM on slurm
start_run = 1
end_run = 20

for i in range(start_run, end_run+1):
    arg1 = str(i)
    arg2 = set_type
    arg3 = str(test_num)
    subprocess.run(["sbatch", "--export=ARG1="+arg1+",ARG2="+arg2+",ARG3="+arg3, "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/run_apcemm.sh"])

# WAIT FOR THE JOBS TO FINISH THEN GO TO pce_input_pipeline.py