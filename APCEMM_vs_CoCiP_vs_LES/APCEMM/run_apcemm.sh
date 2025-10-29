#!/bin/bash                                   

#SBATCH --time=96:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name="2K 10 min T Pert Bypass APCEMM"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/testing/5min_no_TP/130T225L25/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=25000MB

#####################################

cd /home/chinahg/GCresearch/APCEMM/examples/issl_rhi140
pwd
/home/chinahg/GCresearch/APCEMM/build/APCEMM $ARG1
