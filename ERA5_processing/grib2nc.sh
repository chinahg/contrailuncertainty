#!/bin/bash                                   

#SBATCH --time=14-0:00
#SBATCH --job-name="ERA5 GRIB to NETCDF"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem=10000MB
#####################################

python /home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/grib2nc.py