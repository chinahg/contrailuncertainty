# You made it! Here's where to start if you want to run the code in this repository.
### Pipeline Step 1: Download GRUAN and ERA5 Data

#############################################################################################################################################
# Imports
import pipeline_fxn_lib as lib
import subprocess
#############################################################################################################################################

## Download GRUAN data of interest
### Use FTP ftp://ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/ to download the data

## Download ERA5 data correpsonding to the available GRUAN data
# What years of the data are you interested in?
years = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'] # Years of interest

# Where did you download the GRUAN data?
GRUAN_base_dir = '/home/chinahg/GCresearch/GRUAN_sondes/ftp.ncdc.noaa.gov/pub/data/gruan/processing/level2/RS92-GDP/version-002/'

# Check which files you might already have ERA5 data for and if they are .nc or GRIB files
lib.check_files(GRUAN_base_dir, years)

#################################################
# EDITABLE

# Download any missing ERA5 files
period_type = "custom" # "continuous" or "custom"

# Process all dates in the file?
entire_file =False

# Only specify if entire_file is False
start_index = 0 # Index of days to start downloading at
end_index = 100 # Index of days to stop downloading at

# Path to the CSV file containing the list of files to download
csv_path = "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/files2download.csv"
#################################################

# Download the missing files using slurm
subprocess.run(["sbatch", "/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/era5_download.sh"])

### WAIT FOR THE JOB TO FINISH THEN GO TO preprocess_pipeline.py
