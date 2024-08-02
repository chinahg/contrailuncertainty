#!/bin/bash                                   

#SBATCH --time=12:00:00
#SBATCH --job-name="GRUAN Processing"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/GRUAN_processing/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=1000MB
#####################################

python /home/chinahg/GCresearch/contrailuncertainty/GRUAN_processing/GRUAN_process.py