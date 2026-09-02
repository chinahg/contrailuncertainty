#!/bin/bash                                   

#SBATCH --time=72:00:00
#SBATCH --job-name=LRT-A130T225L25
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/LRT/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH --mem=150000
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT APCEMM slicing script..."

/home/chinahg/.conda/envs/pycontrails/bin/python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/APCEMM_slicing_RF.py $ARG1