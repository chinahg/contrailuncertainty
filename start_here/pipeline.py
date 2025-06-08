# You made it! Here's where to start if you want to run the code in this repository.
#############################################################################################################################################
# Imports
import time
import numpy as np
import os
from scipy.stats import gaussian_kde
import pandas as pd
import yaml
import tqdm

# Import functions from the pipeline_fxn_lib.py script
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
function_library_path = "/home/chinahg/GCresearch/contrailuncertainty/start_here/pipeline_fxn_lib.py"
import pipeline_fxn_lib as lib

#############################################################################################################################################
### Pipeline Step 0: Choose which parts of the pipeline to run and specify details

### Step 1: Download ERA5 and GRUAN data ###
already_downloaded = True # Set to True if you have already downloaded the data, skips the download step

generated_files_dir = '/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files' # Directory where the generated files are saved

# Where did you download or will download the GRUAN data?
GRUAN_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/'
# Where did you/do you want to save the ERA5 data?
ERA5_base_dir = '/home/chinahg/GCresearch/ERA5_downloads/'
# This path is the base directory for the ERA5 data
# The ERA5 data is saved in the format /home/chinahg/GCresearch/ERA5_downloads/YYYY/YYYY_MM_DD.nc
# The ERA5 save paths are specified in /home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_download.py under the save_path variable

# What years of the data have you downloaded or want to download?
years = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] # Years of interest

# Edit these download parameters which will be accessed if already_downloaded == False
if already_downloaded == False:
    ## Download GRUAN data of interest
    ### Use FTP ftp://ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/ to download the data

    # Process all dates in the file in one slurm job?
    entire_file = False
    # If processing data in batches, specify the variables below
    if entire_file == False:
        # Only specify below if entire_file is False
        files_per_batch = 50 # Number of files to process per slurm batch

    # Path to the CSV file containing the list of files to download
    files2download_path = "/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/files2download.csv"

### Step 2: Calculate and add RHi and RHw keys to ERA5 nc files ###
# Did you already add RHi and RHw keys to the downloaded ERA5 files?
already_added_RH = True # Set to True if you have already added RHi and RHw to the files, skips the calculation step

### Step 3: Calculate MLD then consolidate ERA5 data and GRUAN data into a single file ###
already_preprocessed = True # Set to True if you have already preprocessed the data, skips the preprocessing step
# Produces the dataframes that have co-located GRUAN and ERA5 data, in the form of parquet files that must be consolidated

### Step 4: Create APCEMM meteorological input files, run APCEMM, and save the output ###
already_created_datasets = True # Set to True if you have already created your APCEMM input files

# Must define the dierctory where the meteorlogical data is saved, and the test_specifications for the surrogate being trained and validated
base_parquet_dir = '/home/chinahg/GCresearch/contrailuncertainty/Met_processing/parquet_files/' # Directory where the parquet files are saved: should match that in /home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.py

# Define the base APCEMM meteorlogical file path that we'll copy and modify
# This is the file that will be used as the base for the meteorological input files
input_file_path = '/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/BASE_APCEMM_met.nc'
test_id = "test_17"  # Test number the met and YAML files are associated with, MUST be in the format test_X

# Set the test specifications for the PCE model if the training and validation sets have not already been created
if already_created_datasets == False:
    # Create a test object containing the necessary parameters for the PCE model
    # Define the test specifications as a dictionary
    test_specifications_dict = {
        "aircraft_engine": "B737-800_CFM56",  # Which aircraft and engine are we using?
        "test_id": test_id,  # Test number the met and YAML files are associated with, MUST be in the format test_X
        "training_runs": 5,  # Number of APCEMM runs to do for training
        "validation_runs": 5,  # Number of APCEMM runs to do for validation
        "novel_runs": 5,  # Number of APCEMM runs to do for novel inputs
        "polynomial_degree": 1,  # Number of random variables
        "maximum_degree": 2,  # Maximum value of sum of degrees across the number of uncertain variables
        "APCEMM_altitudes": 125,  # Number of altitudes recorded in meteorological file
        "APCEMM_timesteps": 24,  # Hours recorded in meteorological file
        "APCEMM_output_timesteps": 72,  # Number of timesteps in the APCEMM output file
    }

    # Update the test_specifications object with the dictionary
    test_specifications = lib.APCEMMConfig(**test_specifications_dict)  # Unpack the dictionary into the APCEMMConfig object
    print(f"Test id set to: {test_specifications.test_id}")
    # Save the test specifications to a YAML file
    test_specifications_path = f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_specifications.test_id}/"
    
    if not os.path.exists(test_specifications_path):
        # Create the directory if it doesn't exist
        print(f"Creating directory for test specifications: {os.path.dirname(test_specifications_path)}")
        os.makedirs(os.path.dirname(test_specifications_path), exist_ok=False)  # Create the directory if it doesn't exist

    with open(f'{test_specifications_path}/test_specifications.yaml', "w") as f:
        yaml.dump(test_specifications.__dict__, f)
else:
    # Load the test specifications from the YAML file
    print(f"Loading test specifications from {test_id}...")
    test_specifications_path = f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_id}/test_specifications.yaml"
    
    with open(test_specifications_path, "r") as f:
        test_specifications_dict = yaml.safe_load(f)
    test_specifications = lib.APCEMMConfig(**test_specifications_dict)  # Unpack the dictionary into the APCEMMConfig object

### Step 5: Create the PCE model ###
already_created_PCE = False # Set to True if you have already created the PCE using the npy files, skips the PCE creation step

### Step 6: Run the surrogate for novel inputs ###
already_run_novel_inputs = False # Set to True if you have already run the surrogate for novel inputs, skips the surrogate bulk run step

#############################################################################################################################################
### Pipeline Step 1: Download GRUAN and ERA5 Data

# Check which files you might already have ERA5 data for and if they are .nc or GRIB files
print("Step 0: Checking for existing GRUAN and ERA5 data...")
files2convert, files2download, GRUAN_date_sites = lib.check_files(GRUAN_base_dir, years, generated_files_dir)

if already_downloaded == False:
    print("Step 1: Downloading GRUAN and ERA5 data...")

    print(lib.download_files(entire_file, files2download_path, files_per_batch, files2download))
    # Convert any GRIB ERA5 files to .nc
    lib.convert_files(ERA5_base_dir, files2convert)

# Make a list of matching files after conversion
print("Matching GRUAN dates with ERA5 files...")
matching_data, matching_files, files2convert, files2download = lib.match_files(GRUAN_date_sites) # Format for matching_data is [ERA5 file name, GRUAN site name, GRUAN datetime object]
    
##############################################################################################################################################
### Pipeline Step 2: Calculate RHi and RHw

if already_added_RH == False:
    print("Step 2: Adding RHi and RHw to ERA5 data...")
    # Calculate RHi and RHw for all the ERA5 files and add as a new variable
    print(lib.add_RHi_RHw(matching_files))
    
##############################################################################################################################################
### Pipeline Step 3: Calculate MLD and consolidate ERA5 data and GRUAN data into a single file

if already_preprocessed == False:
    print("Step 3: Calculating MLD and consolidating GRUAN and ERA5 data...")
    print(lib.process_met_data(GRUAN_base_dir, ERA5_base_dir, generated_files_dir))

#############################################################################################################################################
### Pipeline Step 4: Create APCEMM meteorological input files, run APCEMM, and save the output

if already_created_datasets == False:
    print("Step 4: Creating APCEMM meteorological input files, running APCEMM, and saving the output...")

    # initial_RHi_gaussian = norm.pdf(x_vals*100, mu, std)
    # initial_RHi_gaussian = initial_RHi_gaussian / np.sum(initial_RHi_gaussian)  # Normalize the distribution

    # # Save the details to the test specifications object
    # # RHi initial condition distribution details
    # test_specifications.IC_std_rhi = std
    # test_specifications.IC_mean_amplitude_rhi = mu
    # # RHi temporal distribution details
    # test_specifications.time_std_rhi = 0
    # test_specifications.time_mean_amplitude_rhi = 117

    # # Save the test specifications to a YAML file
    # test_specifications_path = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_specifications.test_id}/test_specifications.yaml"
    # with open(test_specifications_path, "w") as f:
    #     yaml.dump(test_specifications.__dict__, f)

    job_ids = []  # Initialize an array to store job IDs

    for set_type in ["training", "validation"]: # Run the pipeline for both training and validation sets
        print(f"Generating {set_type} data...")
        
        # Create the surrogate input matrix for RHi
        RHi_sampled = lib.make_surrogate_input(test_specifications, set_type) # Returns a matrix of sampled RHi values for each run [runs, timesteps]

        num_met_files = test_specifications.training_runs if set_type == "training" else test_specifications.validation_runs # Number of met files to generate

        # Create the base directory for the specified test number
        base_dir = f"/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/{test_specifications.test_id}"

        # Create the necessary subdirectories
        if not os.path.exists(f"{base_dir}/inputs/{set_type}"):
            os.makedirs(f"{base_dir}/inputs/{set_type}", exist_ok=False)
        
        if not os.path.exists(f"{base_dir}/outputs/{set_type}"):
            os.makedirs(f"{base_dir}/outputs/{set_type}", exist_ok=False)

        # Create the output directories for each run
        for i in range(1, num_met_files + 1):
            os.makedirs(f"{base_dir}/outputs/{set_type}/{test_specifications.test_id}_run_{i}", exist_ok=False)

        # Generate the met files mean_norm, std_norm, mean_norm_time, std_norm_time, IC_scaled_mean, timesteps
        lib.generate_apcemm_input_files(RHi_sampled, input_file_path, num_met_files, set_type, test_specifications)

        # Run batches of APCEMM on slurm
        for i in range(num_met_files):
            arg1 = str(i+1)
            arg2 = set_type
            arg3 = str(test_specifications.test_id)

            export_args = f"ARG1={arg1},ARG2={arg2},ARG3={arg3}"
            bash_path = "/home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/run_apcemm.sh"

            # Submit the job and get the job ID
            job_ids.append(lib.submit_job_and_get_id(bash_path, "has_args", export_args))
            time.sleep(5)  # Optional: wait a bit before submitting the next job

    # Wait for the job to complete
    lib.wait_for_specific_jobs(job_ids)

    # WAIT FOR THE JOBS TO FINISH THEN CONTINUE

    #Save APCEMM input and output variables to a PCE readable format
    lib.create_sample_matrix(test_specifications, set_type="training") # Create the training sample matrix for the specified test run
    lib.create_sample_matrix(test_specifications, set_type="validation") # Create the validation sample matrix for the specified test run
    # The sample matrix is saved as a .npy file in the outputs directory of the specified test run.

    print("Training and validation sample matrices created successfully!")

##################################################################################################################################################
### Pipeline Step 5: Create the PCE model
if already_created_PCE == False:
    print("Step 5: Creating and validating the surrogate model...")

    lib.create_and_validate_surrogate(test_specifications)

##################################################################################################################################################
### Step 6: Run the surrogate for novel inputs
if already_run_novel_inputs == False:
    print("Step 6: Running the surrogate for novel inputs...")
    start_time = time.time()
    print("Start time:", start_time)

    # Load in the surrogate coefficients and indices
    surrogate_coefficients = np.load(f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_specifications.test_id}/PCE_coefficients.npy")
    surrogate_alpha = np.load(f"/home/chinahg/GCresearch/contrailuncertainty/start_here/generated_files/results/{test_specifications.test_id}/PCE_alpha_set.npy")
    run_type = "novel"  # Specify the run type as "novel" for novel inputs

    # Sample the RHi and MLD distributions
    input_rhi_matrix = lib.make_surrogate_input(test_specifications, run_type) # Returns a matrix of size (num_runs, timesteps) with the sampled RHi values
    
    # Compute novel solutions and save them
    lib.novel_solutions(surrogate_coefficients, surrogate_alpha, test_specifications)
    print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Time taken for surrogate bulk run:", time.time() - start_time, "seconds")

##################################################################################################################################################
### Step 7: Perform the sensitivity analysis


##################################################################################################################################################
### Step 8: Calculate radiative forcing using LibRadTran
