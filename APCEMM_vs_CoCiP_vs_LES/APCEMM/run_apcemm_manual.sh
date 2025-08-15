#!/bin/bash                                   

#SBATCH --time=96:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name="Bypass 2.5 Lapse Full LES APCEMM run"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=20000MB

#####################################

cd /home/chinahg/GCresearch/APCEMM/examples/issl_rhi140
pwd
/home/chinahg/GCresearch/APCEMM/build/APCEMM /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/130T225L25/B767_LES_CoCiP_APCEMM_input.yaml
