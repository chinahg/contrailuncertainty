# libRadtran — README

## Overview
This directory contains code, scripts, notebooks and small input files for libRadtran-based radiative‑transfer (LRT) contrail computations. The libRadtran data are referenced via a symlink to Ian Ross' installation (/home/iross/misc-code/libRadtran/data).

## Description
Implements LRT Python wrapper for APCEMM and CoCiP modelled contrails. Drivers call core functions in LRT_fxnlib.py and produce radiatve forcing estimates and associated slurm logs.

## Files
| Filename | Purpose | Produces / notes |
|---|---:|---|
| LRT_fxnlib.py | Core functions / library | Calls LRT, helper functions for pre-processing inputs |
| APCEMM_RF.py | Wrapper for running LRT using APCEMM avg IWC outputs | CSVs containing LW and SW RF per timestep |
| APCEMM_RF.sh | SLURM wrapper | Cluster submission wrapper for APCEMM_RF.py |
| APCEMM_slicing_RF.py | Wrapper for running LRT using APCEMM outputs for multiple slices of the IWC per timestep | Runs LRT on sliced subsets and saves LW and SW results as CSVs |
| micro_sweep_cocip_RF.py | Parameter sweep driver | RF sweeps over habit types and times of day, RF saved as CSV |
| micro_sweep_cocip_RF.sh | SLURM wrapper | Cluster submission wrapper for the sweep |
| base_inputs/ | Input files | Directory with example helper input files used by drivers |
| data -> /home/iross/misc-code/libRadtran/data | Symlink to libRadtran data | Provides libRadtran lookup tables and inputs used by .in files |
| randomseed | Seed file | Contains a seed for reproducible runs |
| slurm_outs/ | SLURM outputs | Job stdout/stderr and job-specific logs produced by .sh submissions |
| README.md | This file | High‑level map of files and usage |

## Usage (examples)
- CoCiP
    - Open ./micro_sweep_cocip_RF.py and update desired arguments and paths
    - Open ./micro_sweep_cocip_RF.sh and update the number of CPUs and paths 
    - Call sbatch ./micro_sweep_cocip_RF.sh from your terminal on Hex
- APCEMM
    - Open ./APCEMM_RF.py OR ./APCEMM_slicing_RF.py and update desired arguments and paths
    - Open ./APCEMM_RF.sh and update the number of CPUs and paths
    - Call sbatch ./APCEMM_RF.sh from your terminal on Hex
- An output file is a CSV with 3 columns
    - Contrail_Age_hours: Age of the contrail in hours
    - LW_Radiative_Forcing_W_m2: Net longwave radiative forcing in W/m2. Equal to CONTRAIL-CLEAR.
    - LW_RF_CLEAR_W_m2: Longwave radiative forcing for clear-sky conditions (no contrail)
    - LW_RF_CONTRAIL_W_m2: Longwave radiative forcing for contrail conditions (yes contrail)

### Notes
- Ensure the data symlink points to a valid libRadtran data directory and that LIBRADTRAN inputs are available.
- Inspect the .sh wrappers for required module/conda setup and command‑line arguments.
- Drivers produce input files and results in the configured output paths, make sure to check these!
