#!/bin/bash                                   

#SBATCH --time=96:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name=12-C130T225L25
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=10000

#####################################

/home/chinahg/.conda/envs/pycontrails/bin/python /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/run_CoCiP_batches.py $ARG1
