### libRadtran FUNCTION LIBRARY ###
import subprocess
import os

def updateInput(filepath, attributes):
    """
    Update the input file with the given attributes.

    Parameters:
    - filepath (str): The path of the input file to be updated.
    - attributes (object): An object containing the attributes to be written to the input file.
    - contrail (bool): A flag indicating whether the input file is for contrail simulation or not.

    Returns:
    None
    """
    file = open(filepath,"w")
    # Write each attribute as "Name Value # Description" from the attributes dataframe
    for row in attributes.iterrows():
        name = row[1].Name
        value = str(row[1].Value)
        description = row[1].Description if row[1].Description != "" else "No description provided"
        file.write(f"{name} {value}             # {description}\n")
    file.close()

def calculate_SW_Flux(attributes, contrail_flag):
    # Run the shortwave (solar) radiative forcing simulation.
    if contrail_flag:
        # If contrail_flag is True, use the contrail input file
        input_file = "solar-cloud.in"
    else:
        # If contrail_flag is False, use the clear sky input file
        input_file = "solar-clear.in"

    updateInput(input_file, attributes)

    cmd = ["/home/iross/misc-code/libRadtran/bin/uvspec"]
    with open(input_file, 'r') as f:
        SW_output = subprocess.run(
            cmd,
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LD_LIBRARY_PATH": "/data/home/chinahg/.conda/envs/afca-test/lib:" + os.environ.get("LD_LIBRARY_PATH","")},
            text=True,
        )

    SW_output = reformatResults(SW_output.stdout)  # Get the last element which is the net TOA flux
    return SW_output

def calculate_LW_Flux(attributes, contrail_flag):
    # Run the longwave (thermal) radiative forcing simulation.
    if contrail_flag:
        # If contrail_flag is True, use the contrail input file
        input_file = "thermal-cloud.in"
    else:
        # If contrail_flag is False, use the clear sky input file
        input_file = "thermal-clear.in"

    updateInput(input_file, attributes)

    cmd = ["/home/iross/misc-code/libRadtran/bin/uvspec"]
    with open(input_file, 'r') as f:
        LW_output = subprocess.run(
            cmd,
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LD_LIBRARY_PATH": "/data/home/chinahg/.conda/envs/afca-test/lib:" + os.environ.get("LD_LIBRARY_PATH","")},
            text=True,
        )

    LW_output = reformatResults(LW_output.stdout)  # Get the last element which is the net TOA flux
    return LW_output

def reformatResults(resultsRaw):
    string = str(resultsRaw.strip().replace("  ", " "))
    li = list(string.split(" "))
    flux = float(li[-2])  # The second last element is the net TOA flux
    return flux