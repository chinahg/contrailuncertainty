### GRUAN FUNCTION LIBRARY ###

import numpy as np

def compute_Psat_w(T):
    """
    Returns water liquid saturation pressure in Pascal.
    Source: Sonntag (1994)

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        H2O Liquid saturation pressure in Pascal
    """
    return 100.0 * np.exp(
        -6096.9385 / T
        + 16.635794
        - 0.02711193 * T
        + 1.673952e-5 * T**2
        + 2.433502 * np.log(T)
    )

def compute_Psat_i(T):
    """
    Returns water solid saturation pressure in Pascal.
    Source: Sonntag (1990)

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        H2O solid saturation pressure in Pascal
    """
    return 100.0 * np.exp(
        -6024.5282 / T
        + 24.7219
        + 0.010613868 * T
        - 1.3198825e-5 * T**2
        - 0.49382577 * np.log(T)
    )

