import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from . import config
from .flight.route_parser import make_input_df_routes
from .simulation.apcemm_input_file import APCEMMSimParams, make_yaml_input
from .utils import check_input_waypoint_df
from .weather.advection import advect_waypoints
from .weather.apcemm_input import make_multiple_nc_inputs
from .weather.weather_input import HRRRExtent, make_AFCA_weather_input


@dataclass
class PathConfig:
    """
    Dataclass containing all relevant paths. Some common defaults such
    as weather data location can be specified through the package config
    file, other frequently changing ones need to be initialized everytime

    Structure of output will be:
    path_simdir/
        APCEMM_out/
            0000_xxxxx/
                APCEMM out files for waypoint 0000
            0001_xxxxx/
                APCEMM out files for waypoint 0001
            ...
        input/
            met_files/
                *.nc
            waypoints/
                *.pkl
            yaml/
                *.yaml
    """

    # Path to directory to save all inputs and outputs in
    path_sim_dir: str
    # Directory containing HRRR weather data in .nc format
    path_hrrr_nc: str = field(default=config.get("Weather", "path_hrrr_nc"))
    # Directory to save the HRRR wind data in an AFCA compatible format
    path_hrrr_winds: str = field(default=config.get("Weather", "path_winds"))
    # Path to default yaml input file
    path_default_yaml_input: str = field(default=config.get("APCEMM", "default_yaml"))
    # APCEMM output save directory to create per run subdirectories in
    path_apcemm_save_results: str = field(init=False, default="")
    # Directory to save the generated yaml inputs in
    path_save_yaml_input: str = field(init=False, default="")
    # Directory to save the .pkl file containing the advected waypoints
    path_save_waypoints: str = field(init=False, default="")
    # Directory to save the .nc met inputs to APCEMM
    path_save_nc_input: str = field(init=False, default="")

    def init_paths(self) -> None:
        # First check that specified paths to met exist
        if not os.path.exists(self.path_hrrr_nc):
            raise FileNotFoundError(
                f"{self.path_hrrr_nc} not found. Invalid HRRR weather data path"
            )

        if not os.path.exists(self.path_hrrr_winds):
            raise FileNotFoundError(
                f"{self.path_hrrr_winds} not found. Invalid HRRR wind data path"
            )

        # Make the base save directory
        if not os.path.exists(self.path_sim_dir):
            os.mkdir(self.path_sim_dir)
        else:
            raise FileExistsError(f"{self.path_sim_dir} already exists.")

        # Make input subdir
        os.mkdir(self.path_sim_dir + "input/")

        self.path_apcemm_save_results = self.path_sim_dir + "APCEMM_out/"
        self.path_save_yaml_input = self.path_sim_dir + "input/yaml/"
        self.path_save_waypoints = self.path_sim_dir + "input/waypoints/"
        self.path_save_nc_input = self.path_sim_dir + "input/met_files/"

        path_subdirs = [
            self.path_apcemm_save_results,
            self.path_save_nc_input,
            self.path_save_yaml_input,
            self.path_save_waypoints,
        ]

        # Make subdirs for input/output
        for path in path_subdirs:
            os.mkdir(path)


def make_apcemm_inputs(
    path_config: PathConfig,
    df_initial_waypoints: pd.DataFrame,
    overwrite=True,
    alt_range: slice = slice(600, 100),
    apcemm_sim_params: Optional[APCEMMSimParams] = None,
    path_wind_ds: Optional[str] = None,
) -> None:
    """
    Given a pd.DataFrame of waypoints (id, times, longitudes, latitudes, pressures, heading), prepares
    all APCEMM input files. Advects and saves the waypoints, uses the advected waypoints to generate the
    .nc met input to APCEMM, and generates the .yaml input file.

    ***CURRENTLY ONLY WORKS ON HRRR DATA***

    Parameters
    ----------
    path_config : PathConfig
        Object containing all paths
    df_initial_waypoints : pd.DataFrame
        Locations at which APCEMM will be run. df columns:
        (id, times, longitudes, latitudes, pressures, heading)
        - heading (degrees) is necessary to compute the shear of the contrail
        - id is an int assigned to each waypoint which is used in
            the simulation results save name
        - pressures (hPa)
    overwrite : bool, optional
        Overwrite existing winds, advection and input files, by default True
    alt_range : slice, optional
        Altitude range (hPa) to restrict the domain too. Done for memory savings
        May need to be extented to account for plume descent, by default slice(600, 100)
    apcemm_sim_params : APCEMMSimParams, optional
        Specific APCEMM simulation parameters. Most relevant is the max simulation duration
        which is by default 5h, by default None
    """
    # Check that input df is valid
    check_input_waypoint_df(df_initial_waypoints)

    # Create all save paths
    path_config.init_paths()

    # Use default parameters if not initialized
    if apcemm_sim_params is None:
        apcemm_sim_params = APCEMMSimParams()

    # For now, only compatible with HRRR
    is_HRRR = True
    if is_HRRR:
        # This should be reimplemented into something better...
        # The idea is to be able to restrict the extent of the dataset
        # that we are loading for memory savings. Currently restricts
        # only the altitude range
        weather_extent = HRRRExtent()

    # Convert from np.datetime64 to pd.Timestamp to dt.datetime...
    min_time = pd.to_datetime(pd.Timestamp(df_initial_waypoints.times.values.min()))

    # Get minimum hour required here, this is to locate the correct weather files
    previous_hour = dt.datetime(
        min_time.year, min_time.month, min_time.day, min_time.hour
    )
    # +2 is necessary to avoid strange AFCA effects at time bounds...
    # TODO fix this issue
    n_hours_wind = apcemm_sim_params.simulation_duration + 2

    # Generates the .nc file containing the relevant HRRR wind data for advection
    path_wind_ds = make_AFCA_weather_input(
        path_config.path_hrrr_nc,
        path_config.path_hrrr_winds,
        previous_hour,
        weather_extent,
        n_hours_wind,
        overwrite=False,
    )
    
    # Save the waypoints df in the correct location with the time used as filename
    filename_waypoints = min_time.strftime("%Y%m%d_%H%M_waypoints.pkl")
    path_save_waypoints = path_config.path_save_waypoints + filename_waypoints

    # Advect all initial waypoints to get trajectories for each simulation
    df_waypoints = advect_waypoints(
        df_initial_waypoints,
        path_wind_ds,
        apcemm_sim_params.simulation_duration,
        is_HRRR,
        path_save_waypoints,
        return_output=True,
        overwrite=overwrite,
    )

    # Make the .nc weather inputs for APCEMM, generates 1 .nc file per input waypoint
    make_multiple_nc_inputs(
        path_config.path_hrrr_nc,
        df_waypoints,
        path_config.path_save_nc_input,
        overwrite,
        alt_range,
    )

    # Make the corresponding yaml inputs for APCEMM poiting to the right .nc files generated previously
    make_yaml_input(
        df_waypoints,
        path_config.path_save_nc_input,
        path_config.path_save_yaml_input,
        path_config.path_apcemm_save_results,
        path_config.path_default_yaml_input,
        apcemm_sim_params,
    )

    print("Successfully generated all inputs")


def make_inputs_for_route(
    route: str,
    path_config: PathConfig,
    time: dt.datetime,
    pressure: float,
    overwrite: bool = True,
    sub_sample: Union[None, int] = None,
    alt_range: slice = slice(600, 100),
    apcemm_sim_params: APCEMMSimParams = None,
) -> None:
    df_input = make_input_df_routes(route, time, pressure, sub_sample)
    make_apcemm_inputs(path_config, df_input, overwrite, alt_range, apcemm_sim_params)
