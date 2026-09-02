#!/bin/bash                                   

#SBATCH --time=06:00:00
#SBATCH --job-name=LRT-L130T225L25
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/LRT/slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH --mem=17000
#####################################

cd /home/chinahg/GCresearch/contrailuncertainty/LRT
echo "Running LRT comparison script..."

/home/chinahg/.conda/envs/pycontrails/bin/python -u /home/chinahg/GCresearch/contrailuncertainty/LRT/LES_slicing_RF.py $ARG1