#!/bin/bash                                   

#SBATCH --time=06:00:00
#SBATCH --job-name="A130T205L25"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/RF_EF_results/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH --mem=25000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."

test_id=${SLURM_JOB_NAME#A}  # Extract test ID from job name (removing leading 'A')

python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/APCEMM_RF.py "$test_id"