#!/bin/bash                                   

#SBATCH --time=12:00:00
#SBATCH --job-name="APCEMM training run"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=5000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets

./../../../APCEMM/Code.v05-00/APCEMM input.yaml