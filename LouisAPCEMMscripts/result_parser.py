import datetime as dt
import glob
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

APCEMM_OUT_VARIABLES = [
    "r_e",
    "Pressure",
    "Altitude",
    "H2O",
    "Temperature",
    "Ice aerosol particle number",
    "Ice aerosol surface area",
    "Ice aerosol volume",
    "Effective radius",
    "Horizontal optical depth",
    "Vertical optical depth",
    "Overall size distribution",
    "Ice Mass",
    "Number Ice Particles",
    "Extinction",
    "IWC",
    "RHi",
    "width",
    "depth",
    "intOD",
]


class RunStatus(Enum):
    # Run finished successfully and contrail disappeared before max simulation time
    COMPLETED = 0
    # Run finished successfully but reached max simulation time before contrail disappeared
    INCOMPLETE = 1
    # Run crashed
    FAILED = -1


@dataclass
class APCEMMDataStructure:
    paths_ds_step: np.ndarray
    path_combined_ds: Union[None, str] = None

    def open_ds(self) -> xr.Dataset:
        # Prioritize reading the combined ds because it is much faster to open
        # than opening multiple files using open_mfdataset (order of x100 faster)
        if self.path_combined_ds is not None:
            return xr.open_dataset(
                self.path_combined_ds, engine="netcdf4", decode_times=False
            )
        else:
            return xr.open_mfdataset(
                self.paths_ds_step,
                concat_dim="t",
                combine="nested",
                engine="netcdf4",
                decode_times=False,
            )

    def get_variable(self, variable: str) -> xr.DataArray:
        if variable not in APCEMM_OUT_VARIABLES:
            raise ValueError(f"Variable '{variable}' not in APCEMM output variables")
        ds = self.open_ds()
        ds_var = ds[variable]
        ds.close()
        return ds_var


@dataclass
class APCEMMResult:
    path_simulation: str
    time_init_simulation: dt.datetime
    longitudes: np.ndarray
    latitudes: np.ndarray
    pressures: np.ndarray
    status: RunStatus
    combine_ds: bool = True
    data: Union[APCEMMDataStructure, None] = field(init=False)
    lifetime_mins: int = field(init=False)
    n_files: int = field(init=False)

    def __post_init__(self):
        self.parse_data()

    def parse_data(self, verbose=False) -> None:
        """
        Reads the APCEMM_out directory for this particular simulation collecting metadata such as location,
        timesteps and provides access to the underlying APCEMM timestep NetCDF outputs.

        Defaults to combining all the timestep NetCDF files into once large NetCDF file. This file
        is saved to path_simulation/combined_ts_aerosol.nc
        This means the first call to parse_data will be slow, but subsequent calls will be faster by directly
        reading the new NetCDF file. (e.g. for 11 timesteps: first call ~650ms, second call ~5ms)

        Reading a lot of small NetCDF files (APCEMM timestep NetCDFs are numerous but small files)
        incurs a lot of overhead in xarray to check that concatenation is valid.
        By combining all the timestep datasets and saving it into one dataset to disk, we do
        this operation only once and then access the new combined NetCDF dataset.

        Slow read speed is particularly problematic when iterating through 100s of APCEMM simulation
        to retrieve the evolution of a specific variable from the ts_aerosol.nc files. This solution comes
        at the cost of more disk space usage.

        Parameters
        ----------
        verbose : bool, optional
            Verbose ouputs, by default False
        """
        array_path_files = np.sort(
            glob.glob(self.path_simulation + "/ts_aerosol" + "*.nc")
        )

        self.n_files = array_path_files.size

        if self.n_files == 0:
            self.timestep = None
            self.lifetime_mins = 0
            self.data = None
            if verbose:
                print("No ts_aerosol file found => no persistent contrail")
        else:
            #### Get some metadata for the simulations ####

            # Take the last saved aerosol file corresponding to last timestep
            # Formating is ts_aerosol_case0_HHMM.nc
            lifetime_hours = int(array_path_files[-1].split(".")[0][-4:-2])
            lifetime_mins = int(array_path_files[-1].split(".")[0][-2:])
            self.lifetime_mins = lifetime_hours * 60 + lifetime_mins

            #### Access the aerosol dataset files ####
            # See doc of function for why we default to combining APCEMM output datasets
            path_combined_ds = self.path_simulation + "combined_ts_aerosol.nc"

            if os.path.exists(path_combined_ds):
                self.data = APCEMMDataStructure(
                    array_path_files, path_combined_ds=path_combined_ds
                )
            else:
                if self.combine_ds:
                    combined_ds = APCEMMDataStructure(array_path_files).open_ds()
                    combined_ds.to_netcdf(path_combined_ds)

                    if verbose:
                        print(f"Saved combined dataset to {path_combined_ds}")

                    self.data = APCEMMDataStructure(
                        array_path_files, path_combined_ds=path_combined_ds
                    )
                else:
                    # Accessing data this way is slow
                    self.data = APCEMMDataStructure(array_path_files)


@dataclass
class APCEMMResultParser:
    path_run: str
    path_apcemm: str = field(init=False)
    path_waypoints: str = field(init=False)
    n_simulations: int = field(init=False)
    df_waypoints: Union[pd.DataFrame, None] = field(init=False, default=None)
    run_statuses: dict = field(init=False)

    def __post_init__(self):
        self.path_apcemm = self.path_run + "APCEMM_out/"
        self.path_waypoints = self.path_run + "input/waypoints/"
        self.n_simulations = len(glob.glob(self.path_apcemm + "*/"))
        self._parse_simulations_status()

    def _parse_simulations_status(self):
        path_txt_file = self.path_run + "failed_jobs.txt"
        unique_waypoints = np.sort(self.get_input_waypoints().id.unique())

        # By default consider all run succeeded
        self.run_statuses = {wid: RunStatus.COMPLETED for wid in unique_waypoints}

        # If the file does not exist then no jobs failed
        if not os.path.exists(path_txt_file):
            return
        else:
            # Fetch the ids int the log from the run
            failed_runs = []
            with open(path_txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    waypoint_id = int(line.split("/")[-1].split("_")[0])
                    failed_runs.append(waypoint_id)

            # Update the relevant waypoints' status
            for wid in failed_runs:
                self.run_statuses[wid] = RunStatus.FAILED

        # TODO add some check for INCOMPLETE RUNS

    def get_input_waypoints(self) -> pd.DataFrame:
        list_path_waypoints = glob.glob(self.path_waypoints + "*.pkl")
        if len(list_path_waypoints) > 1:
            raise ValueError(
                "More than one pickle file found for waypoints, cannot guess which one is correct"
            )
        return pd.read_pickle(list_path_waypoints[0])

    def get_result(
        self, waypoint_id: int, df_in_memory=False, completed_runs_only: bool = True
    ):
        status = self.run_statuses[waypoint_id]

        # Use Python 3.10 and use match statement instead...
        if status != RunStatus.COMPLETED:
            if status == RunStatus.FAILED:
                if completed_runs_only:
                    raise ValueError(
                        f"APCEMM run for {waypoint_id = } has crashed. By default it cannot be accessed. To access it anyways, set completed_runs_only = False"
                    )
                else:
                    print(
                        f"**** APCEMM run for {waypoint_id = } has crashed, proceed with caution ****"
                    )

            elif status == RunStatus.INCOMPLETE:
                raise NotImplementedError

        # Check if the df has been saved in memory
        if self.df_waypoints is None:
            df_waypoints = self.get_input_waypoints()
            row = df_waypoints[df_waypoints.id == waypoint_id]
            if df_in_memory:
                self.df_waypoints = df_waypoints
        # df is already in memory, we skip the loading
        else:
            row = self.df_waypoints[self.df_waypoints.id == waypoint_id]

        if row.index.size == 0:
            raise ValueError(
                f"Waypoint ID {waypoint_id} is not in input waypoint dataframe"
            )

        w_id_str = str(waypoint_id).rjust(4, "0")
        list_dir_waypoint = glob.glob(self.path_apcemm + w_id_str + "*")

        if len(list_dir_waypoint) == 0:
            raise FileNotFoundError(
                f"Waypoint ID {waypoint_id} is in the input waypoint dataframe but has no corresponding simulation directory.\
                Unexpected, maybe a formatting error in the save directory name, or APCEMM crash, or APCEMM run is still ongoing"
            )
        elif len(list_dir_waypoint) > 1:
            raise ValueError(
                f"Waypoint ID {waypoint_id} has multiple candidate APCEMM out directories, cannot guess correct one"
            )

        path_sim = list_dir_waypoint[0] + "/"

        return APCEMMResult(
            path_sim,
            row.time.values.min(),
            row.longitude.values,
            row.latitude.values,
            row.pressure.values,
            status,
        )

    def iter_results(self, completed_runs_only: bool = True):
        """
        Convenience method to iterate through all the results

        Yields
        ------
        APCEMMResult
            Objects are ordered by waypoint id
        """
        # Dict keys are already sorted, but to guarantee access in correct order
        # if keys are modified, we sort just in case
        sorted_wids = sorted(self.run_statuses.keys())
        for wid in sorted_wids:
            status = self.run_statuses[wid]
            if status != RunStatus.COMPLETED and completed_runs_only:
                continue
            # Setting df_in_memory avoids reading the .pkl file at every iteration
            yield self.get_result(
                wid, completed_runs_only=completed_runs_only, df_in_memory=True
            )
