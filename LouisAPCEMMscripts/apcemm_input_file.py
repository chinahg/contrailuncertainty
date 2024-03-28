"""
Make the YAML input file for APCEMM
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .. import PACKAGE_LOCATION, config
from ..weather.apcemm_input import format_save

PATH_DEFAULT_YAML = config.get("APCEMM", "default_yaml")
PATH_DEFAULT_BACKGROUND = PACKAGE_LOCATION + "/asset/init.txt"
PATH_DEFAULT_ENGINE_EI = PACKAGE_LOCATION + "/asset/ENG_EI.txt"


@dataclass
class APCEMMSimParams:
    overwrite_folder: str = field(init=False, default="F")
    simulation_duration: int = field(init=False, default=5)
    path_background: str = field(init=False, default=PATH_DEFAULT_BACKGROUND)
    path_engine: str = field(init=False, default=PATH_DEFAULT_ENGINE_EI)


def load_default_dict(path: str = PATH_DEFAULT_YAML) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        input_file = yaml.safe_load(stream)
    return input_file


def make_yaml_input(
    df_waypoints: pd.DataFrame,
    path_apcemm_nc: str,
    path_save_input: str,
    path_apcemm_save: str,
    path_defaults: str = PATH_DEFAULT_YAML,
    sim_params: APCEMMSimParams = APCEMMSimParams(),
) -> None:
    waypoinds_id = np.sort(df_waypoints.id.unique())

    dict_defaults = load_default_dict(path_defaults)

    dict_defaults["SIMULATION MENU"]["OUTPUT SUBMENU"][
        "Overwrite if folder exists (T/F)"
    ] = sim_params.overwrite_folder

    dict_defaults["SIMULATION MENU"][
        "Input background condition (string)"
    ] = sim_params.path_background
    dict_defaults["SIMULATION MENU"][
        "Input engine emissions (string)"
    ] = sim_params.path_engine

    dict_defaults["PARAMETER MENU"][
        "Plume Process [hr] (double)"
    ] = sim_params.simulation_duration

    for _, w_id in enumerate(waypoinds_id):
        sub_df = df_waypoints[df_waypoints.id == w_id]
        init_time = sub_df.time.min()
        initial_df = sub_df[sub_df.time == init_time]

        ##### Update the position of the emission of the plume #####

        # Strangely enough if you don't convert to float here, the yaml input
        # contains a view of the array object and not the actual float itself
        init_alt = float(initial_df.pressure.values[0])
        init_longitude = float(initial_df.longitude.values[0])
        init_latitude = float(initial_df.latitude.values[0])

        dict_defaults["PARAMETER MENU"]["LOCATION AND TIME SUBMENU"][
            "LON [deg] (double)"
        ] = init_longitude

        dict_defaults["PARAMETER MENU"]["LOCATION AND TIME SUBMENU"][
            "LAT [deg] (double)"
        ] = init_latitude

        dict_defaults["PARAMETER MENU"]["METEOROLOGICAL PARAMETERS SUBMENU"][
            "Pressure [hPa] (double)"
        ] = init_alt

        ##### Update the time of emission of the plume #####
        day_of_year = int(init_time.strftime("%j"))
        hour_of_emission = float(init_time.hour + init_time.minute / 60)

        dict_defaults["PARAMETER MENU"]["LOCATION AND TIME SUBMENU"][
            "Emission day [1-365] (int)"
        ] = day_of_year

        dict_defaults["PARAMETER MENU"]["LOCATION AND TIME SUBMENU"][
            "Emission time [hr] (double)"
        ] = hour_of_emission

        ##### Update the path to the met file #####
        path_met_file = format_save(path_apcemm_nc, w_id, init_time, init_alt)
        dict_defaults["METEOROLOGY MENU"]["METEOROLOGICAL INPUT SUBMENU"][
            "Met input file path (string)"
        ] = path_met_file

        ##### Update the save location #####
        full_save_dir = path_apcemm_save + path_met_file.split("/")[-1].replace(
            ".nc", ""
        )
        dict_defaults["SIMULATION MENU"]["OUTPUT SUBMENU"][
            "Output folder (string)"
        ] = full_save_dir

        ##### Find the correct path to save this .yaml input #####
        path_yaml = path_save_input + path_met_file.split("/")[-1].replace(
            ".nc", ".yaml"
        )
        with open(path_yaml, "w", encoding="utf-8") as stream:
            yaml.dump(dict_defaults, stream)

        print(f"Saved input file at {path_yaml}")
