#!/bin/bash                                   

#SBATCH --time=50:00
#SBATCH --nodelist=c040
#SBATCH --job-name="APCEMM training run 1"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/APCEMM_slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem=5000MB
#####################################

cd /home/chinahg/GCresearch/APCEMM/examples/issl_rhi140

./../../build/APCEMM /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_1/APCEMM_input_test_1.yaml