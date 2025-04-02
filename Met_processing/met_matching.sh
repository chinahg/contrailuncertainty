#!/bin/bash                                   

#SBATCH --time=72:00:00
#SBATCH --job-name="Met Processing Batch"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/Met_processing/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=5000MB
#####################################

python /home/chinahg/GCresearch/contrailuncertainty/Met_processing/met_matching.py