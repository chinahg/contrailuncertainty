# APCEMM Application of PCE with LRT
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.interpolate import interp1d
import totalOrderMultiIndexSet
from sklearn.neighbors import KernelDensity
from numpy.polynomial.hermite_e import HermiteE
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cdsapi

plt.rcParams["figure.figsize"] = (10, 6)
import tqdm
import time

# For LRT
#!/usr/bin/python
import sys
import os

import warnings
import subprocess
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

### FUNCTION LIBRARY ###

def PCE():
    
    print("Started!")
    print("--- %s minutes elapsed ---" % round(time.time()/60 - start_time/60,2))
    
    c, alpha_set = compute_c()
    multiindices = alpha_set.shape[0]
    
    print("Got c and alpha!")

    He_array = np.zeros((multiindices, mc_runs))
    u_evals = np.ones(timesteps)
    
    print("Started MC runs!")

    comparison_samples_matrix = get_samples_matrix(mc_runs)

    for i in tqdm.tqdm(range(mc_runs)): # for each MC run
        u_evals = np.vstack((u_evals, solve_u(comparison_samples_matrix)))
        
        for j in range(multiindices): # for each coefficient
            current_alpha = alpha_set[j, :] # look at one row at a time for all alpha describing a single coefficient
            samples_matrix_He = comparison_samples_matrix[i, :]
            He_array[j, i] = compute_He(samples_matrix_He, current_alpha)

    # Get rid of first row of ones
    true_output = u_evals[1:,:]

    # Solve for the Least Squares solution
    predicted_output = c.T @ He_array
    
    print("--- %s minutes elapsed ---" % round(time.time()/60 - start_time/60,2))
    
    return predicted_output.T, true_output, c, alpha_set

def compute_c():
    
    c_samples_matrix = get_samples_matrix(training_runs)
    
    alpha_set = get_alpha_set()
    print("alpha set shape: ", alpha_set.shape)

    c_multiindices = alpha_set.shape[0]

    V = np.zeros((training_runs, c_multiindices)) # Vandermonde Matrix: Timesteps x Degree of Polynomial

    for i in tqdm.tqdm(range(training_runs)):
        for j in range(c_multiindices):
            current_alpha = alpha_set[j, :]
            c_samples_matrix_He = c_samples_matrix[i, :]
            V[i, j] = compute_He(c_samples_matrix_He, current_alpha) / np.sqrt(np.product(sc.special.factorial(current_alpha)))
    
    f = solve_u(c_samples_matrix) # f should be same dimension as V (Training_runs x Degree of Polynomial)
    
    c = np.linalg.solve(V.T @ V, V.T @ f)
    
    return c, alpha_set

def dummy_func_eval(timesteps, samples_array):

    true_output = np.zeros_like(samples_array)
    for t in range(0, timesteps):
        true_output[:,t] = samples_array[:,t] + (np.sin(2*np.pi*(t+1)/timesteps))
    
    return true_output

def get_samples_matrix(runs):
    return np.random.normal(mean_Y, sigma_Y, size = (runs, timesteps)) # was 0,1

def get_alpha_set():
    return totalOrderMultiIndexSet.totalOrderMultiIndices(poly_dim, max_deg)

def compute_He(Z, alpha):
    res = 1
    for alpha_i, z_i in zip(alpha, Z):
        res = res * HermiteE.basis(deg = alpha_i)(z_i)
    return res

def solve_u(samples_matrix):
    
    u_sol = LRT_func_eval(timesteps, samples_matrix) # dummy_func_eval(timesteps, samples_matrix) #NOT SURE
    return u_sol

#-----------------------------------------------------------------------------------

def validate_PCE(samples_matrix, c, alpha_set):
    multiindices = alpha_set.shape[0]
    He_array = np.zeros((multiindices, validation_runs))
    u_evals = np.ones(timesteps)

    for i in tqdm.tqdm(range(validation_runs)): # for each MC run
        u_evals = np.vstack((u_evals, solve_u(samples_matrix)))
        
        for j in range(multiindices): # for each coefficient
            current_alpha = alpha_set[j, :] # look at one row at a time for all alpha describing a single coefficient
            samples_matrix_He = samples_matrix[i, :]
            He_array[j, i] = compute_He(samples_matrix_He, current_alpha)

    # Get rid of first row of ones
    true_output = u_evals[1:,:]

    # Solve for the Least Squares solution
    predicted_output = (c.T @ He_array).T

    return predicted_output, true_output

### END FUNCTION LIBRARY ###
#-----------------------------------------------------------------------------------
### LRT FUNCTION LIBRARY ###

def LRT_func_eval(timesteps, samples_array):
    true_output = np.zeros_like(samples_array)
    num_samples = samples_array.shape[0] # check if correct
    for j in range(0, num_samples):
        for t in range(0, timesteps):
            # Update contrail LWC
            sample = samples_array[j,t]
            updateOD(t, sample)
            contrailFluxRaw = contrailRF(attributes,"ice")
            contrailFlux = contrailFluxRaw[-1]
            clearFluxRaw = clearskyRF(attributes)
            clearFlux = clearFluxRaw[-1]
            true_output[j,t] = float(contrailFlux) - float(clearFlux)
        print("Contrail RF f(t): ", true_output[j,:])
            
    return true_output

def updateOD(timestep,sample):
    print("timestep: ", timestep)
    print("sample: ", sample)
    sample = (sample*10) + 25 # transform variance and mean of distribution
    if sample <= 0:
        sample = 0.0001
    attributes.loc[0,"ic_modify"] = str(sample)


def updateInput(filepath, attributes, contrail, type):
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
    if contrail == True and type == "water":
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n",
                            "wc_file 1D" +attributes.wc_file[0]+"\n",
                            "quiet"])
        file.close()
    elif contrail == True and type == "ice":
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n", 
                            "ic_habit "+str(attributes.ic_habit[0])+"\n",
                            "ic_properties "+str(attributes.ic_properties[0])+"\n",
                            "ic_file 1D "+attributes.ic_file[0]+"\n",
                            "ic_modify tau set "+attributes.ic_modify[0]+"\n",
                            "quiet"])
        file.close()
    elif contrail == False and type == "clear": 
        file.writelines(["rte_solver "+attributes.rte_solver[0]+"\n",
                            "source "+attributes.source[0]+"\n",
                            "sza "+attributes.sza[0]+"\n",
                            "wavelength "+attributes.wavelength[0]+"\n",
                            "mol_abs_param "+attributes.mol_abs_param[0]+"\n",
                            "umu "+attributes.umu[0]+"\n",
                            "output_user "+attributes.output_user[0]+"\n",
                            "zout "+attributes.zout[0]+"\n",
                            "output_process "+attributes.output_process[0]+"\n",
                            "atmosphere_file "+str(attributes.atmosphere_file[0])+"\n",
                            "quiet\n"])
        file.close()
    
def clearskyRF(attributes):
    """
    Run the clear sky radiative forcing simulation.

    Parameters:
    - attributes (object): An object containing the attributes for the simulation.

    Returns:
    - LRToutput (list): A list containing the output of the simulation.
    """
    print("Running clear sky RF simulation...")
    updateInput("/home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-clear.in", attributes, False, "clear")
    try:
        result = subprocess.check_output("/home/chinahg/GCresearch/contrailuncertainty/LRT/uvspec < /home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-clear.in", shell=True)
    except subprocess.CalledProcessError as e:
        print("Error: ", e.returncode, e.output)

    LRToutput = list(map(float, result.decode('utf-8').strip().split()))
    return LRToutput

def contrailRF(attributes, type):
    """
    Run the contrail radiative forcing simulation.

    Parameters:
    - attributes (object): An object containing the attributes for the simulation.

    Returns:
    - LRToutput (list): A list containing the output of the simulation.
    """
    print("Running contrail RF simulation...")
    updateInput("/home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-cloud.in", attributes, True, type)
    try:
        result = subprocess.check_output("/home/chinahg/GCresearch/contrailuncertainty/LRT/uvspec < /home/chinahg/GCresearch/contrailuncertainty/LRT/thermal-cloud.in", shell=True)
    except subprocess.CalledProcessError as e:
        print("Error: ", e.returncode, e.output)

    LRToutput = list(map(float, result.decode('utf-8').strip().split()))
    return LRToutput

### END LRT FUNCTION LIBRARY ###
#-----------------------------------------------------------------------------------
### MAIN ###

# Define constants
training_runs = 2 # Number of LRT runs for training
timesteps = 10 # Number of samples per LRT run (this is the number of timesteps per run)
max_deg = 2 # Maximum value of sum of degrees across the number of uncertain variables (AKA sum of each row of alpha_set must be less than or equal to the max_deg)
poly_dim = 1 # Number of random variables
mc_runs = 1 # Number of PCE runs

#FOR TESTING
mean_Y = 1
sigma_Y = 0.5 #squared value?

# Define LRT constants
# Set-up 
rte_solver = "disort" # 1D radiative transfer solver (DIScrete ORdinaTe solver)
source = "thermal" # Absorbing on the thermal spectrum (not solar)
sza = "0" # Solar zenith angle
wavelength = "2500 80000" # Wavelength range to compute over
mol_abs_param = "reptran fine" # spectral resolution (fine/medium)
umu = "1.0" # cosine of the viewing zenith angle
output_user = "edir eglo edn eup enet esum" # The direct, global, diffuse downward, 
                                            #and diffuse upward irradiance. Net is 
                                            #global - upward, sum is global + upward.
zout = "TOA" # Top of atmosphere (where total flux is calculated)
output_process = "integrate" # Integrate over wavelength
atmosphere_file = "midlatitude_summer" #, "midlatitude_winter", "subarctic_summer", "subarctic_winter", "tropics", "US-standard"] # Standard atmosphere type
ic_habit = "droxtal" #, "hollow-column", "rough-aggregate", "rosette-4", "rosette-6", "plate", "droxtal", "dendrite", "spheroid"]
ic_properties = "yang"
ic_file = "/home/chinahg/GCresearch/contrailuncertainty/LRT/ice.in"
wc_file = "/home/chinahg/GCresearch/contrailuncertainty/LRT/cloud.in"
ic_modify = "15"

inputs = {'rte_solver': [rte_solver], 'source': [source], 'sza': [sza], 'wavelength': [wavelength], 
                      'mol_abs_param': [mol_abs_param], 'umu': [umu], 'output_user': [output_user], 'zout': [zout], 
                      'output_process': [output_process], 'atmosphere_file': [atmosphere_file], 'ic_habit': [ic_habit], 
                      'ic_properties': [ic_properties], 'ic_file': [ic_file], 'wc_file': [wc_file], 'ic_modify': [ic_modify]}
attributes = pd.DataFrame(data = inputs)

start_time = time.time()

# Call PCE function
predicted_output, training_output, c, alpha = PCE()

validation_runs = 1
validation_matrix = get_samples_matrix(validation_runs)
v_predicted_output, v_true_output = validate_PCE(validation_matrix, c, alpha)

#-----------------------------------------------------------------------------------
### SAVE DATA TO CSV ###

# Save training data
print("Saving data to CSV...")
np.savetxt("training_output.csv", training_output, delimiter=",")
np.savetxt("predicted_output.csv", predicted_output, delimiter=",")
np.savetxt("c.csv", c, delimiter=",")
np.savetxt("alpha.csv", alpha, delimiter=",")
np.savetxt("validation_output.csv", v_true_output, delimiter=",")
np.savetxt("v_predicted_output.csv", v_predicted_output, delimiter=",")
print("--- %s minutes elapsed ---" % round(time.time()/60 - start_time/60,2))