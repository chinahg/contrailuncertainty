### Pipeline Step 3a: Create APCEMM meteorological input files
# This script samples fluctuations in relative humidity values, updates the dataset with these sampled values, and sets initial meteorological conditions for contrail formation. 
# It saves the modified dataset to new NetCDF files to be used as meteorological input files to APCEMM.

##############################################################################################################################################
# Imports
import pipeline_fxn_lib as lib
import os
import subprocess
import importlib.util

# Import the user defined variables from the met_matching.py script
import_path = "/home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.py"
pipeline = importlib.util.spec_from_file_location("met_matching", import_path)
met_matching = importlib.util.module_from_spec(pipeline)
pipeline.loader.exec_module(met_matching)

# Import only the specific variables you need
base_parquet_dir = getattr(met_matching, 'base_save_dir', None)  # Fallback to None if not found

##############################################################################################################################################

## Explanation of PCE training and validation processes
# 0) Choose the type of data you want to generate: training or validation.
#    - Training data is used to train the machine learning model.
#    - Validation data is used to test the performance of the machine learning model.
# 1) Define which aircraft and engine we are basing this case-study on
# 2) Load in th necessary variables that APCEMM needs to define the aircraft and engine
# 3) Load in distributions of RHi and MLD based on the GRUAN and ERA5 data we just processed
# 4) Sample from the distributions to create a set of RHi and MLD values
# 5) Use the sampled values to update the meteorological input file
# 6) Save the modified meteorological input file to a new NetCDF file
# 7) Run the APCEMM model using the new input file
# 8) Save the output of the APCEMM model to a new NetCDF file
# 9) Repeat steps 4-8 for a set number of iterations to create a set of meteorological input files and APCEMM output files

def create_pdf(data, variable):
    # Create a probability density function (PDF) for the given variable using Kernel Density Estimation (KDE)
    # This function will take in the data and the variable name and return the PDF

    # Check if the variable is in the data
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' not found in data.")

    # Extract the data for the variable
    variable_data = data[variable].dropna()

    # Create a KDE for the variable
    pdf = lib.create_kde(variable_data)

    return pdf

def get_initial_dist(base_parquet_dir):
    # Create a probability density function (PDF) for the RHi and MLD distributions
    # This function will load the GRUAN and ERA5 data and create a PDF for each variable

    # Combine all parquet files in the base directory into a single DataFrame
    combined_df = lib.combine_parquet_files(base_parquet_dir)

    # Extract the RHi and MLD columns from the DataFrame, keeping the GRUAN and ERA5 data separate
    gruan_RHi = combined_df['RHi_GRUAN'].dropna()
    era5_RHi = combined_df['RHi_ERA5'].dropna()
    gruan_MLD = combined_df['MLD_GRUAN'].dropna()
    era5_MLD = combined_df['MLD_ERA5'].dropna()

    # Create a PDF for each variable using the GRUAN and ERA5 data using KDE (Kernel Density Estimation)
    gruan_RHi_pdf = create_pdf(gruan_RHi, 'RHi_GRUAN')
    era5_RHi_pdf = create_pdf(era5_RHi, 'RHi_ERA5')
    gruan_MLD_pdf = create_pdf(gruan_MLD, 'MLD_GRUAN')
    era5_MLD_pdf = create_pdf(era5_MLD, 'MLD_ERA5')

    return gruan_RHi_pdf, era5_RHi_pdf, gruan_MLD_pdf, era5_MLD_pdf

# get pdfs for each variable
gruan_RHi_pdf, era5_RHi_pdf, gruan_MLD_pdf, era5_MLD_pdf = get_initial_dist(base_parquet_dir)
# plot the pdfs for each variable using matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(gruan_RHi_pdf['x'], gruan_RHi_pdf['pdf'], label='GRUAN RHi PDF')
plt.title('GRUAN RHi PDF')
plt.subplot(2, 2, 2)
plt.plot(era5_RHi_pdf['x'], era5_RHi_pdf['pdf'], label='ERA5 RHi PDF')
plt.title('ERA5 RHi PDF')
plt.subplot(2, 2, 3)
plt.plot(gruan_MLD_pdf['x'], gruan_MLD_pdf['pdf'], label='GRUAN MLD PDF')
plt.title('GRUAN MLD PDF')
plt.subplot(2, 2, 4)
plt.plot(era5_MLD_pdf['x'], era5_MLD_pdf['pdf'], label='ERA5 MLD PDF')
plt.title('ERA5 MLD PDF')
plt.tight_layout()
plt.show()
##############################################################################################################################################


# # Load in the meteorological GRUAN and ERA5 distributions to sample from for the initial conditions
# initial_RHi_pdf = 
# initial_MLD_pdf =

# # Define the base APCEMM meteorlogical file path that we'll be modifying
# input_file_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/BASE_APCEMM_met.nc'

# set_type = "training" # Is this data for validating or training the model? Choose "validation" or "training".

# # Define the meteorological APCEMM file dimensions
# altitudes = 125
# timesteps = 24

# # Initial condition distribution details
# mean_norm = 0
# std_norm = 0.5
# IC_scaled_mean = 117

# # RHi temporal distribution details
# std_norm_time = 0.5
# mean_norm_time = 0

# num_met_files = 20 # Number of met and YAML files to generate
# test_num = 3 # Test number the met and YAML files are associated with

# # Create the necessary directories if they don't exist
# # Create the base directory for the specified test number
# base_dir = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_{test_num}"

# # Create the necessary subdirectories
# os.makedirs(base_dir, exist_ok=False)
# os.makedirs(f"{base_dir}/inputs/{set_type}", exist_ok=False)

# # Create the output directories for each run
# for i in range(1, num_met_files + 1):
#     os.makedirs(f"{base_dir}/outputs/{set_type}/test_{test_num}_run_{i}", exist_ok=False)

# # Generate the met files
# lib.generate_apcemm_input_files(input_file_path, num_met_files, test_num, set_type)

# # Run batches of APCEMM on slurm
# start_run = 1
# end_run = 20

# for i in range(start_run, end_run+1):
#     arg1 = str(i)
#     arg2 = set_type
#     arg3 = str(test_num)
#     subprocess.run(["sbatch", "--export=ARG1="+arg1+",ARG2="+arg2+",ARG3="+arg3, "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/run_apcemm.sh"])

# # WAIT FOR THE JOBS TO FINISH THEN GO TO pce_input_pipeline.py