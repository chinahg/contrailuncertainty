#!/bin/bash                                   
#SBATCH --time=06:00:00
#SBATCH --job-name="C130T225L25"
#SBATCH --constraint=tengig
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/RF_EF_results/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH --mem=20000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."
test_id=${SLURM_JOB_NAME#C}  # Extract test ID from job name (removing leading 'C')

python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/micro_sweep_cocip_RF.py "$test_id"