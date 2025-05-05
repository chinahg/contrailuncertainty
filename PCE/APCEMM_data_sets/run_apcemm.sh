#!/bin/bash                                   

#SBATCH --time=4:00:00
#SBATCH --constraint=tengig
#SBATCH --job-name="APCEMM run"
#SBATCH --mail-type=BEGIN,END
#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/APCEMM_slurm_outs/slurm-%j-out
#xSBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=4000MB
#####################################

# for i in {4..5}
# do
#     echo "Running APCEMM test set 2, run $i"
#     srun /home/chinahg/GCresearch/APCEMM/build/APCEMM /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_training_sets/test_2/inputs/training/APCEMM_input_run_$i.yaml
# done

srun /home/chinahg/GCresearch/APCEMM/build/APCEMM /home/chinahg/GCresearch/contrailuncertainty/PCE/APCEMM_data_sets/$ARG3/inputs/$ARG2/APCEMM_input_run_$ARG1.yaml