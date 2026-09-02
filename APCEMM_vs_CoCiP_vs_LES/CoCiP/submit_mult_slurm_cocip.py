import time
import sys
sys.path.append('/home/chinahg/GCresearch/contrailuncertainty/start_here/')
import pipeline_fxn_lib as lib

# Run batches of CoCiP on slurm
test_ids = ['110T205L25', '110T218L25', '110T225L25', '130T205L25', '130T218L25', '130T225L25'] # ['110T205L25', '110T218L25', '110T225L25', '130T205L25', '130T218L25', '130T225L25']
times_of_day = ['midnight', 'noon']
num_sims = 2
make_new_met = "False"
mem_per_job = 10000  # Memory per CPU in MB (if make new met is True must be at least 195 GB, otherwise can be around 3 GB)

bash_path = "/home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/run_CoCiP_batches.sh"

for test_id in test_ids:
    for time_of_day in times_of_day:
        print(f"Submitting job for test ID: {test_id}")
        print(f"Contrail is modelled at {time_of_day}")
        time_of_day_flag = "0" if time_of_day == "midnight" else "12"
    
        export_args = f"ARG1={test_id} {time_of_day} {make_new_met}"
        print(export_args)
    
        # Update where the slurm output file is saved to
        with open(bash_path, "r") as file:
            bash_lines = file.readlines()
    
        # Modify the output file path in the bash script
        for j, line in enumerate(bash_lines):
            if "#SBATCH --job-name=" in line:
                bash_lines[j] = f"#SBATCH --job-name={time_of_day_flag}-C{test_id}\n"
            if "#SBATCH -o" in line:
                bash_lines[j] = f"#SBATCH -o /home/chinahg/GCresearch/contrailuncertainty/APCEMM_vs_CoCiP_vs_LES/CoCiP/slurm_outs/slurm-%j-out\n"
                
            if "#SBATCH --cpus-per-task=" in line:
                bash_lines[j] = f"#SBATCH --cpus-per-task={num_sims}\n"
                
            if "#SBATCH --mem-per-cpu=" in line:
                bash_lines[j] = f"#SBATCH --mem-per-cpu={mem_per_job}\n"
                
            
        # Write the modified bash script back to the file
        with open(bash_path, "w") as file:
            file.writelines(bash_lines)
    
        # Submit the job and get the job ID
        lib.submit_job_and_get_id(bash_path, "has_args", export_args)
    
        time.sleep(5)  # Optional: wait a bit before submitting the next job