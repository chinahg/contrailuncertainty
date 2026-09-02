#!/bin/bash                                   

#SBATCH --time=96:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name="PDF Bypass 130% 205K APCEMM"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/130T205L25/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=30000MB

#####################################

cd /home/chinahg/GCresearch/APCEMM/examples/issl_rhi140
pwd
/home/chinahg/GCresearch/APCEMM/build/APCEMM /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/130T205L25/B767_LES_CoCiP_APCEMM_input.yaml /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/APCEMM/epm_bypass/130T205L25/overlay-input.yaml
