#!/bin/bash                                   

#SBATCH --time=01:00:00
#SBATCH --job-name="OAG Processing, 2022"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/OAG_processing/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem=5000MB
#####################################

python /home/chinahg/GCresearch/contrailuncertainty/OAG_processing/OAG_processing.py