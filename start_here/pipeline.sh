#!/bin/bash                                   

#SBATCH --time=72:00:00
#SBATCH --job-name="Run Surrogate Pipeline"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/start_here/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=5000MB
#####################################

/home/chinahg/.conda/envs/contrails/bin/python /home/chinahg/GCresearch/contrailuncertainty/start_here/pipeline.py