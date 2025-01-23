import numpy as np

### Functions ###

# Function to construct ERA5 file path from GRUAN file path
def construct_era5_path(gruan_file_path):
    date_str = os.path.basename(gruan_file_path).split('_')[4][:8]
    era5_file_path = os.path.join(era5_base_dir, date_str[:4], f'{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}.nc')
    return era5_file_path

def fill_nan_with_next(arr):
    """Fill NaN values in the array with the next non-NaN value, if available."""
    for i in range(len(arr) - 1):
        if np.isnan(arr[i]):
            next_valid = next((x for x in arr[i + 1:] if not np.isnan(x)), np.nan)
            arr[i] = next_valid
    return arr

def press2alt(pressure):
    """
    Convert pressure to altitude.

    Parameters
    ----------
    pressure : Union[int, np.ndarray]
        Pressure in Pascal.

    Returns
    -------
    Union[float, np.ndarray]
        Altitude in meters.
    """
    L = -6.5*10**-3
    P0 = 101325
    T0 = 288.15
    R = 287.053
    g = 9.81

    altitudes = np.zeros_like(pressure)

    if type(pressure)==int:
        return (T0/L)*((pressure*100/P0)**(-R*L/g) -1)
    else:
        for i in range(len(pressure)):
            altitudes[i] = (T0/L)*((pressure[i]*100/P0)**(-R*L/g) -1)

        return altitudes

def alt2press(altitude):
    """
    Convert altitude to pressure.

    Parameters
    ----------
    altitude : Union[float, np.ndarray]
        Altitude in meters.

    Returns
    -------
    Union[int, np.ndarray]
        Pressure in Pascal.
    """
    L = -6.5*10**-3
    P0 = 101325
    T0 = 288.15
    R = 287.053
    g = 9.81

    if isinstance(altitude, (float, np.float32)):
        return P0*(1 + L*altitude/T0)**(-g/(R*L))
    else:
        pressures = np.zeros_like(altitude)
        for i in range(len(altitude)):
            pressures[i] = P0*(1 + L*altitude[i]/T0)**(-g/(R*L))

        return pressures

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

def calculate_evaporation_depth(regime_array, altitudes, RHw, T, pressures, latitude, longitude, met_type):
    # P_sat: Saturation vapor pressure (Pa)
    # RH: Relative humidity wrt ice (unitless)
    # P_atm: Atmospheric pressure (Pa)
    
    print("Initial Regime Binary:", regime_array)
    regime_array_ED = regime_array.copy()
    R = 6371*10**3 # Radius of the Earth (m)
    
    if met_type == "ERA5":
        # latitude and longitude must be floats describing the location of the GRUAN launch site
        if isinstance(latitude, float) != True:
            raise ValueError("Invalid type of", type(latitude) ,". Latitude and Longitude must be floats describing the location of the GRUAN launch site. Did you mean to pass in the GRUAN type parameter?")
        elif isinstance(longitude, float) != True:
            raise ValueError("Invalid type of", type(longitude) ,". Latitude and Longitude must be floats describing the location of the GRUAN launch site. Did you mean to pass in the GRUAN type parameter?")
        
        lat_len = 111320*0.25 # Latitude length (m). The size of a degree of latitude remains fairly constant across the Earth.
        lon_len = np.abs((2*np.pi*R*np.cos(latitude)/360)*0.25) # Longitude length (m), Haverside function
        heights = np.diff(altitudes) # Height of the gridcell (m)
        V = heights*lat_len*lon_len # volume of air in m^3
        
    elif met_type == "GRUAN":
        # latitude and longitude are altitude/pressure dependent arrays of floats describing the path of the radiosonde
        if isinstance(latitude, float) == True:
            raise ValueError("Invalid type of", type(latitude) ,". Latitude and Longitude must be arrays of floats describing the path of the GRUAN radiosonde. Did you mean to pass in the ERA5 type parameter?")
        elif isinstance(longitude, float) == True:
            raise ValueError("Invalid type of", type(longitude),". Latitude and Longitude must be arrays of floats describing the path of the GRUAN radiosonde. Did you mean to pass in the ERA5 type parameter?")
        
        lat_len = np.abs(111320*np.diff(latitude)) # Latitude length (m)
        lon_len = np.abs((2*np.pi*R*np.cos(averages_between_elements(latitude))/360)*(np.diff(longitude))) # Longitude length (m), Haverside function
        height = np.diff(altitudes) # Height of the gridcell (m)
        V = height*lat_len*lon_len # volume of air in m^3
    
    else:
        raise ValueError("Invalid type. Must be 'ERA5' or 'GRUAN'.")

    contrail_molecules_water = 0 # Initialize the amount of water molecules in the volume of air
    for i in range (len(regime_array)):
        
        # If regime_array[i] == 1, the altitude is supersaturated
        if regime_array[i] == 1:
            print("Initially Supersaturated: regime_array is 1")
            # Calculate water picked up in supersaturated zone
            P_sat = compute_Psat_w(T[i]) # [Pa]
            P_atm = pressures[i] # [Pa]
            ppmv = (P_sat/P_atm)*(RHw[i])*10**6 # ppmv
            contrail_molecules_water = contrail_molecules_water + ppmv*V[i] # molecules of water in the volume of air

        else:
            print("Initially Subsaturated: regime_array is 0")
            # Calculate water deposited in subsaturated zone
            P_sat = compute_Psat_w(T[i]) # [Pa]
            P_atm = pressures[i] # [Pa]
            contrail_molecules_water_initial = contrail_molecules_water
            sat_molecules_water = (P_sat/P_atm)*10**6*V[i] # For RH = 1, molecules of water in the volume of air required for saturation
            background_molecules_water = (P_sat/P_atm)*(RHw[i])*10**6*V[i] # For RH < 1, molecules of water actually in the volume of air
            contrail_molecules_water = contrail_molecules_water - (sat_molecules_water - background_molecules_water) # Amount of water molecules deposited in the volume of air by the contrail
            print("Available water:", f"{contrail_molecules_water_initial :.2e}", 
                  "Saturation Req:", f"{sat_molecules_water - background_molecules_water:.2e}", 
                  "Remaining ppm:", f"{contrail_molecules_water:.2e}")

        # When contrail_molecules_water = 0, the contrail has evaporated
        if contrail_molecules_water > 0:
            regime_array_ED[i] = 1 # Mark subsaturated binary as supersaturated (evaporation depth)
        else:
            contrail_molecules_water = 0 # Reset the amount of water molecules in the volume of air
            print("Contrail Death, ppm reset: ", contrail_molecules_water, " ppm")
        
        print("Molecules of contrail water at altitude ", altitudes[i], "are:", f"{contrail_molecules_water:.2e}\n")
    print("Evaporation Depth Regime Binary:", regime_array_ED)
    return regime_array_ED

def calculate_MLD(altitudes, pressures, humidities, temperatures, met_type):
    # Convert arrays to read as floats
    humidities = np.array(humidities, dtype=float)
    altitudes = np.array(altitudes, dtype=float)
    
    # Initialize the binary array to store the regime classification
    regime_array = [0] * len(altitudes)  # Start by assuming all are subsaturated (0)

    # Step 1: Find the first altitude with RH >= 1 (supersaturated)
    found_supersaturated = False
    first_supersaturated_index = -1
    
    for i in range(len(humidities)):
        if humidities[i] >= 1:
            found_supersaturated = True
            first_supersaturated_index = i
            regime_array[i] = 1  # Mark this altitude as supersaturated
            break  # Stop once we find the first supersaturated altitude

    if not found_supersaturated:
        print("No supersaturated altitudes found.")
    else:
        # Step 2: Now classify the rest of the altitudes
        for i in range(first_supersaturated_index + 1, len(humidities)):
            if humidities[i] >= 1:
                regime_array[i] = 1  # Mark as supersaturated (since RH >= 1)

        # Step 3: For altitudes below the first supersaturated, calculate evaporation depth
        regime_array_ED = regime_array # Initialize the evaporation depth array with the same values as the regime array, which will then be modified
        for i in range(first_supersaturated_index - 1, -1, -1):
            if humidities[i] < 1:
                # Calculate evaporation depth between current altitude and first supersaturated altitude
                altitude_diff = altitudes[first_supersaturated_index] - altitudes[i]
                evaporation_depth = calculate_evaporation_depth(regime_array, altitudes, humidities, temperatures, pressures, met_type)

                # If the evaporation depth is sufficient, mark as supersaturated
                if evaporation_depth >= altitude_diff:
                    regime_array_ED[i] = 1

    return regime_array, regime_array_ED

def averages_between_elements(arr):
    return [(arr[i] + arr[i + 1]) / 2 for i in range(len(arr) - 1)]
