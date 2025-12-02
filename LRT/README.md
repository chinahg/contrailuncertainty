# libRadtran — README

## Overview
This directory contains code, scripts, notebooks and small input files for libRadtran-based radiative‑transfer (LRT) contrail computations. The libRadtran data are referenced via a symlink to Ian Ross' installation.

## Directory
/home/chinahg/GCresearch/contrailuncertainty/LRT

## Description
Implements LRT comparisons and RF‑based analyses for contrail/optical‑property experiments. Drivers call core functions in LRT_fxnlib.py and produce notebooks, CSV/figure outputs, and slurm logs.

## Files
| Filename | Purpose | Produces / notes |
|---|---:|---|
| LRT_fxnlib.py | Core functions / library | LRT computations, helper functions for pre-processing inputs |
| APCEMM_RF.py | Wrapper for running LRT using APCEMM avg IWC outputs | CSVs containing LW and SW RF per timestep |
| APCEMM_RF.sh | SLURM wrapper | Cluster submission wrapper for APCEMM_RF.py |
| APCEMM_slicing_RF.py | Wrapper for running LRT using APCEMM outputs for multiple slices of the IWC per timestep | Runs LRT on sliced subsets and saves LW and SW results as CSVs |
| micro_sweep_cocip_RF.py | Parameter sweep driver | RF sweeps over microphysical parameters (COCIP); outputs metrics and plots |
| micro_sweep_cocip_RF.sh | SLURM wrapper | Cluster submission wrapper for the sweep |
| base_inputs/ | Input snippets | Directory with small input files used by drivers |
| data -> /home/iross/misc-code/libRadtran/data | Symlink to libRadtran data | Provides libRadtran lookup tables and inputs used by .in files |
| randomseed | Seed file | Contains a seed for reproducible runs |
| slurm_outs/ | SLURM outputs | Job stdout/stderr and job-specific logs produced by .sh submissions |
| README.md | This file | High‑level map of files and usage |

Ignored directories (not listed above)
- to_clean/ (excluded per request)
- __pycache__/ (Python bytecode cache; ignore)

## Usage (examples)
- Interactive: open LRT_example.ipynb or RF_compare.ipynb (if present) in JupyterLab to reproduce analyses and plots.
- CLI: run drivers or use the SLURM wrappers; e.g.
    - ./APCEMM_RF.sh  (or python APCEMM_RF.py with appropriate args)
    - ./micro_sweep_cocip_RF.sh

### Notes
- Ensure the data symlink points to a valid libRadtran data directory and that LIBRADTRAN inputs are available.
- Inspect the .sh wrappers for required module/conda setup and command‑line arguments.
- Drivers produce results (CSV/figures) in the working directory or configured output paths.
