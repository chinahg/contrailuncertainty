import time
import numpy as np
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib

# Run batches of CoCiP slicing on slurm
test_ids = ['110T205L25', '110T218L25', '110T225L25', '130T205L25', '130T218L25', '130T225L25']
hours = '0+12' # Midnight (0) and/or Noon (12)
habits = "rough-aggregate+solid-column+ghm" # ["rough-aggregate", "solid-column", "ghm"]
num_cpus = 6  # Number of simulations to run in parallel inside each job (one job per test ID)
mem_per_job = 16000  # Memory per CPU in MB (set a fixed value or calculate based on test_id if needed)

bash_path = "/home/chinahg/GCresearch/contrailuncertainty/LRT/CoCiP_slicing.sh"


for i, test_id in enumerate(test_ids):

    export_args = f"ARG1={test_id} {habits} {hours} {num_cpus}"

    print(export_args)

    # Update where the slurm output file is saved to
    with open(bash_path, "r") as file:
        bash_lines = file.readlines()

    # Modify the output file path in the bash script
    for j, line in enumerate(bash_lines):
        if "#SBATCH --job-name=" in line:
            bash_lines[j] = f"#SBATCH --job-name=LRT-C{test_id}\n"
        if "#SBATCH -o" in line:
            bash_lines[j] = f"#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/LRT/slurm_outs/slurm-%j-out\n"
            
        if "#SBATCH --cpus-per-task=" in line:
            print(f"Setting number of CPUs to {num_cpus}")
            bash_lines[j] = f"#SBATCH --cpus-per-task={num_cpus}\n"
            
        if "#SBATCH --mem-per-cpu=" in line:
            print(f"Setting memory per CPU to {mem_per_job} MB")
            bash_lines[j] = f"#SBATCH --mem-per-cpu={mem_per_job}\n"
            
        
    # Write the modified bash script back to the file
    with open(bash_path, "w") as file:
        file.writelines(bash_lines)

    # Submit the job and get the job ID
    lib.submit_job_and_get_id(bash_path, "has_args", export_args)

    time.sleep(5)  # Optional: wait a bit before submitting the next job