#!/bin/bash                                   

#SBATCH --time=24:00:00
#SBATCH --job-name="Run IWC LRT"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/LRT/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=2000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."

python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/IWC_RF_compare.py