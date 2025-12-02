#!/bin/bash                                   

#SBATCH --time=02:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name="CoCiP Batch Run"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/micro_sweeps_110_218/EInvpm/5e15/1e-6-slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=2500MB

#####################################

/home/chinahg/.conda/envs/pycontrails/bin/python /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/run_CoCiP_batches.py "$ARG1" "$ARG2" "$ARG3"
