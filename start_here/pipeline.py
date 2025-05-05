# You made it! Here's where to start if you want to run the code in this repository.
### Pipeline Step 1: Download GRUAN and ERA5 Data

#############################################################################################################################################
# Imports
import subprocess
import time
import importlib.util
import numpy as np

# Import functions from the pipeline_fxn_lib.py script
function_library_path = "/home/chinahg/GCresearch/contrailuncertainty/start_here/pipeline_fxn_lib.py"
spec = importlib.util.spec_from_file_location("lib", function_library_path)
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
#############################################################################################################################################

# ## Download GRUAN data of interest
# ### Use FTP ftp://ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/ to download the data

# ## Download ERA5 data correpsonding to the available GRUAN data
# # What years of the data are you interested in?
# years = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] # Years of interest

# # Where did you download the GRUAN data?
# GRUAN_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/'

# # Check which files you might already have ERA5 data for and if they are .nc or GRIB files
# files2convert, GRUAN_date_sites = lib.check_files(GRUAN_base_dir, years)

# #################################################
# # EDITABLE

# # Download any missing ERA5 files
# period_type = "custom" # "continuous" or "custom"

# # Process all dates in the file?
# entire_file =False

# # Only specify if entire_file is False
# start_index = 0 # Index of days to start downloading at
# end_index = 10 # Index of days to stop downloading at

# # Path to the CSV file containing the list of files to download
# csv_path = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/files2download.csv"
# #################################################

def submit_job_and_get_id(script_path, job_type, export_args):
    # Submit the job using sbatch and capture the output
    if job_type == "ERA5":
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True
        )
    elif job_type == "APCEMM":
        result = subprocess.run(
        ["sbatch", "--export=" + export_args, script_path],
        capture_output=True,
        text=True
    )

    # Extract the job ID from the output
    output = result.stdout.strip()
    job_id = output.split()[-1]  # Get the last word from the output
    print(f"Submitted job with ID: {job_id}")
    return job_id

def wait_for_specific_jobs(job_ids):
    job_ids_str = ",".join(job_ids)
    while True:
        result = subprocess.run(
            ["squeue", "-j", job_ids_str],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error from squeue: {result.stderr}")
            break
        # Check if only header line is present
        if len(result.stdout.strip().splitlines()) <= 1:
            print("All specified jobs are completed!")
            break
        else:
            print("Waiting for jobs to complete...")
            time.sleep(300) #  # Wait for 10 minutes before checking again

# # Download the missing ERA5 files using slurm
# # Submit the job and get the job ID
# # job_id = submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/era5_download.sh", "ERA5", None)

# # # Wait for the job to complete
# # wait_for_specific_jobs([job_id])

# # print("Job completed, continuing with the script!")
# ### WAIT FOR THE JOB TO FINISH THEN CONTINUE


# ### Pipeline Step 2a: Convert GRIB to NetCDF, calculate RHi

# ##############################################################################################################################################
# # Imports
# import numpy as np
# ##############################################################################################################################################

# # Where did you save the ERA5 data?
# data_dir = '/home/chinahg/GCresearch/ERA5_downloads/'

# # Convert any GRIB ERA5 files to .nc
# lib.convert_files(data_dir, files2convert)

# ### Pipeline Step 2b: Calculate RHi and RHw
# # Remake the list of matching files after conversion
# matching_data, matching_files, files2convert, files2download = lib.match_files(GRUAN_date_sites) # Format for matching_data is [ERA5 file name, GRUAN site name, GRUAN datetime object]

# # # Calculate RHi and RHw for all the ERA5 files and add as a new variable
# # lib.add_RHi_RHw(matching_files)

# ### Pipeline Step 2c: Consolidate ERA5 data and GRUAN data into a single file
# # Base directories
# gruan_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/'
# era5_base_dir = '/home/chinahg/GCresearch/ERA5_downloads/'

# gruan_file_paths, era5_file_paths = lib.construct_complementary_paths(gruan_base_dir, era5_base_dir)

# # Define the start and end file indices for processing, as ideally we will split the files into batches for slurm
# files_per_batch = 10  # Number of files to process per slurm batch
# batches = 2 #len(gruan_file_paths) // files_per_batch + (1 if len(gruan_file_paths) % files_per_batch != 0 else 0)
# start_indices = np.full(batches, np.NaN)
# end_indices = np.full(batches, np.NaN)

# for i in range(batches):
#     start_indices[i] = i * files_per_batch
#     end_indices[i] = start_indices[i] + files_per_batch
#     if end_indices[i] > len(gruan_file_paths):
#         end_indices[i] = len(gruan_file_paths)

# job_ids = []  # Initialize an array to store job IDs
# for i in range(batches):
#     start_index = int(start_indices[i])
#     end_index = int(end_indices[i])
#     batch = i + 1  # Batch number for slurm job
    
#     # Save the required arguments to pass to met_processing.py into a csv file that can be easily read in
#     arguments = f"{start_index},{end_index},{files_per_batch},{batches},{csv_path}"
#     with open("/home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_processing_args.csv", "w") as f:
#         f.write(arguments)

#     # Submit the job and get the job ID
#     print(f"Submitting batch {batch} with files from {start_index} to {end_index}")
#     job_ids.append(submit_job_and_get_id("/home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.sh"))
#     time.sleep(10)  # Optional: wait a bit before submitting the next job

# # Wait for the job to complete
# wait_for_specific_jobs(job_ids)

# print("Job completed, continuing with the script!")

# ### WAIT FOR THE JOBS TO FINISH THEN CONTINUE

# ### Pipeline Step 3a: Create APCEMM meteorological input files
# # This script samples fluctuations in relative humidity values, updates the dataset with these sampled values, and sets initial meteorological conditions for contrail formation. 
# # It saves the modified dataset to new NetCDF files to be used as meteorological input files to APCEMM.

# ##############################################################################################################################################
# # Imports
# import os
# import subprocess
# from scipy.stats import gaussian_kde
# import pandas as pd
# ##############################################################################################################################################

# ## Explanation of PCE training and validation processes
# # 0) Choose the type of data you want to generate: training or validation.
# #    - Training data is used to train the machine learning model.
# #    - Validation data is used to test the performance of the machine learning model.
# # 1) Define which aircraft and engine we are basing this case-study on
# # 2) Load in th necessary variables that APCEMM needs to define the aircraft and engine
# # 3) Load in distributions of RHi and MLD based on the GRUAN and ERA5 data we just processed
# # 4) Sample from the distributions to create a set of RHi and MLD values
# # 5) Use the sampled values to update the meteorological input file
# # 6) Save the modified meteorological input file to a new NetCDF file
# # 7) Run the APCEMM model using the new input file
# # 8) Save the output of the APCEMM model to a new NetCDF file
# # 9) Repeat steps 4-8 for a set number of iterations to create a set of meteorological input files and APCEMM output files

# def get_initial_dist(base_parquet_dir):
#     # Create a probability density function (PDF) for the RHi and MLD distributions
#     # This function will load the GRUAN and ERA5 data and create a PDF for each variable

#     # Combine all parquet files in the base directory into a single DataFrame
#     combined_parquet_name = 'combined_data.parquet'
#     combined_df = lib.combine_parquet_files(base_parquet_dir, combined_parquet_name) # Check the first few rows of the combined DataFrame

#     # Extract the RHi and MLD columns from the DataFrame, keeping the GRUAN and ERA5 data separate
#     print(combined_df.head())
#     print(combined_df['G_RHi'][0])
#     print("Extracting GRUAN and ERA5 data...")
#     gruan_RHi = combined_df['G_RHi'].dropna()
#     era5_RHi = combined_df['E_RHi'].dropna()
#     gruan_MLD = combined_df['G_MLD'].dropna()
#     era5_MLD = combined_df['E_MLD'].dropna()

#     # Create a PDF for each variable using the GRUAN and ERA5 data using KDE (Kernel Density Estimation)
#     print("Creating PDFs for GRUAN and ERA5 data...")
#     gruan_RHi = gruan_RHi.values.reshape(1, -1)
#     print(gruan_RHi[0][0])
#     gruan_RHi_pdf = gaussian_kde(np.random.normal(0, 1, 100))
#     era5_RHi_pdf = gaussian_kde(era5_RHi)
#     gruan_MLD_pdf = gaussian_kde(gruan_MLD)
#     era5_MLD_pdf = gaussian_kde(era5_MLD)

#     return gruan_RHi_pdf, era5_RHi_pdf, gruan_MLD_pdf, era5_MLD_pdf

# base_parquet_dir = '/home/chinahg/GCresearch/contrailuncertainty/Met_processing/' # Directory where the parquet files are saved: should match that in met_matching.py

# # get pdfs for each variable
# gruan_RHi_pdf, era5_RHi_pdf, gruan_MLD_pdf, era5_MLD_pdf = get_initial_dist(base_parquet_dir)
# # plot the pdfs for each variable using matplotlib
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.plot(gruan_RHi_pdf['x'], gruan_RHi_pdf['pdf'], label='GRUAN RHi PDF')
# plt.title('GRUAN RHi PDF')
# plt.subplot(2, 2, 2)
# plt.plot(era5_RHi_pdf['x'], era5_RHi_pdf['pdf'], label='ERA5 RHi PDF')
# plt.title('ERA5 RHi PDF')
# plt.subplot(2, 2, 3)
# plt.plot(gruan_MLD_pdf['x'], gruan_MLD_pdf['pdf'], label='GRUAN MLD PDF')
# plt.title('GRUAN MLD PDF')
# plt.subplot(2, 2, 4)
# plt.plot(era5_MLD_pdf['x'], era5_MLD_pdf['pdf'], label='ERA5 MLD PDF')
# plt.title('ERA5 MLD PDF')
# plt.tight_layout()
# plt.savefig('/home/chinahg/GCresearch/contrailuncertainty/Met_processing/initial_distributions.png')
# plt.show()
##############################################################################################################################################

# # Load in the meteorological GRUAN and ERA5 distributions to sample from for the initial conditions
# initial_RHi_pdf = 
# initial_MLD_pdf =
import os

# Create a test object containing the necessary parameters for the PCE model
test_specifications = lib.APCEMMConfig()

# Update the test_specifications object with the necessary parameters
# Specify the test
test_specifications.aircraft_engine = "B737-800_CFM56"  # Which aircraft and engine are we using?
test_specifications.test_id = "test_12"  # Test number the met and YAML files are associated with
test_specifications.training_runs = 5  # Number of APCEMM runs to process for the test
test_specifications.validation_runs = 5  # Number of APCEMM runs to process for the test
test_specifications.polynomial_degree = 1  # # Number of random variables
test_specifications.maximum_degree = 2  # Maximum value of sum of degrees across the number of uncertain variables (AKA sum of each row of alpha_set must be less than or equal to the max_deg)

# Define the meteorological APCEMM file dimensions
test_specifications.APCEMM_altitudes = 125  # Number of altitudes recorded in meteorological file
test_specifications.APCEMM_timesteps = 24  # Hours recorded in meteorological file

# RHi initial condition distribution details
test_specifications.IC_std_rhi = 10
test_specifications.IC_mean_amplitude_rhi = 117
# RHi temporal distribution details
test_specifications.time_std_rhi = 0
test_specifications.time_mean_amplitude_rhi = 117

# # MLD initial condition distribution details
# test_specifications.IC_std_mld = 20
# test_specifications.IC_mean_amplitude_mld = 100
# # MLD temporal distribution details
# test_specifications.time_std_mld = 0
# test_specifications.time_mean_amplitude_mld = 0

# # Define the base APCEMM meteorlogical file path that we'll be modifying
# input_file_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/BASE_APCEMM_met.nc'
# job_ids = []  # Initialize an array to store job IDs

# for set_type in ["training", "validation"]: # Run the pipeline for both training and validation sets
#     print(f"Generating {set_type} data...")
    
#     num_met_files = test_specifications.training_runs if set_type == "training" else test_specifications.validation_runs # Number of met files to generate
    
#     # Create the base directory for the specified test number
#     base_dir = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_specifications.test_id}"

#     # Create the necessary subdirectories
#     if not os.path.exists(base_dir):
#         os.makedirs(base_dir, exist_ok=False)
#     os.makedirs(f"{base_dir}/inputs/{set_type}", exist_ok=False)

#     # Create the output directories for each run
#     for i in range(1, num_met_files + 1):
#         os.makedirs(f"{base_dir}/outputs/{set_type}/{test_specifications.test_id}_run_{i}", exist_ok=False)

#     # Generate the met files    mean_norm, std_norm, mean_norm_time, std_norm_time, IC_scaled_mean, timesteps
#     lib.generate_apcemm_input_files(input_file_path, num_met_files, set_type, test_specifications)

#     # Run batches of APCEMM on slurm
#     for i in range(num_met_files):
#         arg1 = str(i+1)
#         arg2 = set_type
#         arg3 = str(test_specifications.test_id)

#         export_args = f"ARG1={arg1},ARG2={arg2},ARG3={arg3}"
#         bash_path = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/run_apcemm.sh"

#         # Submit the job and get the job ID
#         job_ids.append(submit_job_and_get_id(bash_path, "APCEMM", export_args))
#         time.sleep(5)  # Optional: wait a bit before submitting the next job

# # Wait for the job to complete
# wait_for_specific_jobs(job_ids)

# # WAIT FOR THE JOBS TO FINISH THEN CONTINUE

# ##################################################################################################################################################
# ### Pipeline Step 3b: Save APCEMM input and output variables to a PCE readable format

# lib.create_sample_matrix(test_specifications, set_type="training") # Create the training sample matrix for the specified test run
# lib.create_sample_matrix(test_specifications, set_type="validation") # Create the validation sample matrix for the specified test run

# print("Training and validation sample matrices created successfully!")
# # The sample matrix is saved as a .npy file in the outputs directory of the specified test run.

##################################################################################################################################################
### Pipeline Step 4: Create the PCE model
# This script creates a PCE model using the sample matrix created in the previous step.
# hermitePoly.py and totalOrderMultiIndexSet.py must be in the same directory as this script.

# Define APCEMM constants
timesteps = 73 # Number of samples per APCEMM run (this is the number of timesteps per run) # 20 hours

# Load in APCEMM inputs and outputs
# APCEMM inputs that were trained on, and we also have the True solution for
raw_training_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/outputs/training/training_sample_matrix.npy'.format(test_specifications.test_id)) # Import APCEMM training data
# APCEMM inputs that were not trained on, and we also have the True solution for
raw_validation_samples_matrix_offline = np.load('/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{}/outputs/validation/validation_sample_matrix.npy'.format(test_specifications.test_id)) # Import APCEMM validation data

# APCEMM has issues with artifacts, smooth those out as pre-processing
clean_training_samples_matrix_offline = np.copy(raw_training_samples_matrix_offline)
clean_validation_samples_matrix_offline = np.copy(raw_validation_samples_matrix_offline)

for i in range(test_specifications.training_runs):
    clean_training_samples_matrix_offline[1, i, :] = lib.smooth_artifacts(raw_training_samples_matrix_offline[1,i,:])
for i in range(test_specifications.validation_runs):
    clean_validation_samples_matrix_offline[1, i, :] = lib.smooth_artifacts(raw_validation_samples_matrix_offline[1,i,:])

# Save the smoothed training and validation sample matrices
# Specify the directory to save the results
results_dir = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/PCE_results/APCEMM_PCE_results/{test_specifications.test_id}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=False)
    
np.save(f"{results_dir}/training_samples_matrix_offline_cleaned.npy", clean_training_samples_matrix_offline)
np.save(f"{results_dir}/validation_samples_matrix_offline_cleaned.npy", clean_validation_samples_matrix_offline)

# Create the PCE with the training data
coefficients, alpha = lib.create_PCE(test_specifications, "training")

print("PCE model created successfully!")

##################################################################################################################################################
### Pipeline Step 5: Validate the PCE model

# Predict the validation solutions based on the trained PCE
predicted_validation_solutions = lib.predicted_validation_solutions(coefficients, alpha, test_specifications)

print("PCE validation tests ran successfully!")

#################################################################################################################################################
### Pipeline Step 6: Save the PCE model and validation results

# Save the PCE coefficients and alpha set
np.save(f"{results_dir}/PCE_coefficients.npy", coefficients)
np.save(f"{results_dir}/PCE_alpha_set.npy", alpha)

# Save the predicted and true validation solutions
np.save(f"{results_dir}/predicted_validation_solutions.npy", predicted_validation_solutions)
np.save(f"{results_dir}/true_validation_solutions.npy", lib.true_validation_solutions("validation", test_specifications.test_id))

# Save the training solutions
np.save(f"{results_dir}/true_training_solutions.npy", lib.true_training_solutions("training", test_specifications.test_id))

#################################################################################################################################################
### Pipeline Step 7: Sobol Sensitivity Analysis

S_T = lib.compute_total_effect_sensitivity_indices(coefficients, test_specifications.polynomial_degree)

# Print the total effect sensitivity indices for the first few timesteps
print("Total Effect Sensitivity Indices (S_T) for first few timesteps:")
for key, values in S_T.items():
    print(f"{key}: {values[:5]}")  # Print first 5 timesteps for brevity