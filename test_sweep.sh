#!/bin/bash                                   

#SBATCH --time=24:00:00
#SBATCH --job-name="Test uncertainty sweep"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_results/
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=50000MB
#####################################

cd /home/chinahg/GCresearch/APCEMM/rundirs/SampleRunDir
./../../Code.v05-00/APCEMM /home/chinahg/GCresearch/contrailuncertainty/input.yaml