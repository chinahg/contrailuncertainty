#!/bin/bash                                   

#SBATCH --time=14-0:00
#SBATCH --job-name="LRT PCE, 1_2_1"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/PCE/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=1000MB
#####################################

python /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_LRT_PCE.py