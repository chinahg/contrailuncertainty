#!/bin/bash                                   

#SBATCH --time=24:00:00
#SBATCH --job-name="Run APCEMM IWC LRT"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/testing/SW_ranges/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem=20000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."

python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/APCEMM_RF.py