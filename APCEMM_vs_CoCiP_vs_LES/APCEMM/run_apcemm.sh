#!/bin/bash                                   

#SBATCH --time=29-00:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name=1000-bins-TP6-TT1-threaded_A130T205L25
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/testing/1000-bins-TP6-TT1-threaded/130T205L25/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=15000MB

#####################################

cd /home/chinahg/GCresearch/APCEMM/examples/issl_rhi140
pwd
/home/chinahg/GCresearch/APCEMM/build/APCEMM $ARG1
