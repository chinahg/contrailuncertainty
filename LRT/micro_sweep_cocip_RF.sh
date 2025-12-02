#!/bin/bash                                   

#SBATCH --time=72:00:00
#SBATCH --job-name="CoCiP IWC LRT"
#SBATCH --constraint=tengig
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/RF_results/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=normal
#SBATCH --mem=210000MB
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."

python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/micro_sweep_cocip_RF.py