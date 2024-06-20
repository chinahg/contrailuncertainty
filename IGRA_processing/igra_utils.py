"""
An amalgamation of functions written for processing and analyzing 
data obtained from the Integrated Global Radiosonde Archive (IGRA).

See also: 
https://www.ncdc.noaa.gov/data-access/weather-balloon/integrated-global-radiosonde-archive

Author: Vincent Meijer
""" 

"""
Additional comments/notes:
- IGRA python module available here: https://pypi.org/project/igra/

"""


import igra
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os 
import glob 
import pandas as pd
import datetime
import scipy.optimize 
import pysolar.solar as pys
import datetime as dt
import sonde_reanalysis_comparison as src
import scipy.interpolate as spy_interp

DATA_DIR = 'data'
EXTENT = [-135, -65, 10, 50] 

def extract_data_station(station_id, start_date=datetime.datetime(2018, 1, 1)):
    """Downloads .txt.zip file containing measurements for a particular station"""
    
    try:
        # Find file name
        file_name = glob.glob(DATA_DIR +'/'+station_id+'*.zip')[0]
        data, stat = igra.read.igra(station_id, file_name)
    except:
        # Download data
        print(f"Downloading data for station {station_id}")
        igra.download.station(station_id, DATA_DIR)
    
         # Find file name 
        file_name = glob.glob(DATA_DIR + '/'+station_id+'*.zip')[0]
        print(f"Found data at {file_name}")
    
        # Get data from file
        data, stat = igra.read.igra(station_id, file_name)
    
    # Filter based on start date
    filtered = data.where(data.date > np.datetime64(start_date), drop=True)
    
    df = filtered.to_dataframe()
    save_name = DATA_DIR + f'/{station_id}.pkh'
    df.to_pickle(save_name)
    print(f"Saved extracted data at {save_name}")



def download_station_data():
    """Download station meta data""" 
    
    stations = igra.download.stationlist(DATA_DIR)
    return stations
    
    

def download_data_for_region(extent, savename=None, parallel=False):
    """
    Downloads data for all stations in region bounded by extent
    
    Inputs:
    
    extent: [minimum longitude, maximum longitude, minimum latitude, maximum latitude]
    savename: where to store results
    parallel: to use parallel processing or not.
    """ 
    
    stations = igra.download.stationlist(DATA_DIR)
    
    # Filter stations based on extent
    stations = stations[(stations['lon'] > extent[0])*(stations['lon'] < extent[1])*(stations['lat'] > extent[2])*(stations['lat'] <extent[3] )]
    
    # Only keep stations that still give data 
    stations = stations[stations['end'] == datetime.datetime.now().year]
    
    print(f"Found {len(stations['id'].values)} stations")
    
    # Download station data 
    if parallel:
        from multiprocessing import Pool
        ex = Pool(8)
        ex.map(extract_dta_station, station_list)
    else:
        for station_id in station_list:
            extract_data_station(station_id)
    
    
    file_list = glob.glob('data/igra/*.pkh')
    
    df = combine_dataframes(file_list)
    
    if not savename:
        savename = f"igra_data_{datetime.datetime.now().strf('%Y%m%d_%H_%M')}.pkh"
    
    df.to_pickle(savename)
    print(f"Saved results at {savename}") 
    

def combine_dataframes(file_list):
    """ Given paths to measurement data for individual stations, 
    combines the dataframes""" 
    
    lst = []
    stations = download_station_data() 
    for file in file_list:
        df = pd.read_pickle(file)
        station_id = os.path.basename(file).replace('.pkh','')
        lon = stations[stations.index == station_id]['lon'].values[0]
        lat = stations[stations.index == station_id]['lat'].values[0]
        df['lon'] = lon
        df['lat'] = lat
        df['station_id'] = station_id
        lst.append(df)
        
    return pd.concat(lst).reset_index(level=[1,0])


def plot_sonde_locations(dataframe, extent, projection):
    unique = dataframe.groupby(['lon', 'lat']).size().reset_index().rename(columns={0:'count'})
    
    fig = plt.figure(figsize = (10, 5))
    
#     EXTENT = [-2373970.2054802035,3638399.906890716, -2388159.0372323524, 1620933.528091551]  
    
#     proj = ccrs.Orthographic(central_latitude=39.8283, central_longitude=-98.5795)

    ax= fig.add_axes([0, 0, 1, 1], projection=projection)
    ax.set_extent(extent,proj)
    ax.scatter(unique['lon'].values, unique['lat'].values,
               marker='o', color='blue', alpha = 1.0, transform=ccrs.Geodetic())
   
    ax.gridlines()
    ax.coastlines()
    ax.legend()


def hyland_wexler_ice(T):
    """Calculate saturation vapor pressure over ice.
    Source: Hyland and Wexler (1983)
    
    Inputs:
    
    T: Temperature in Kelvin
    
    Outputs:
    
    ei: Saturation vapor pressure over ice in Pascal
    """
    exponent = (-0.56745359*10**4 / T                                                           
              + 0.63925247*10**1 
              - 0.96778430*10**-2 * T 
              + 0.62215701*10**-6 * T**2
              + 0.20747825*10**-8 * T**3
              - 0.94840240*10**-12 * T**4
              + 0.41635019*10**1 * np.log(T)
               )
        
    return np.exp(exponent)

def hyland_wexler_water(T):
    """Calculate saturation vapor pressure over water.
    Source: Hyland and Wexler (1983)
    
    Inputs:
    
    T: Temperature in Kelvin
    
    Outputs:
    
    ew: Saturation vapor pressure over water in Pascal
    """
    exponent = (-0.58002206*10**4 / T     
              + 0.13914993*10**1 
              - 0.48640239*10**-1 * T 
              + 0.41764768*10**-4 * T**2 
              - 0.14452093*10**-7 * T**3 
              + 0.65459673*10**1  * np.log(T) 
               )
        
    return np.exp(exponent)

def augment_dataframe(dataframe):
    """
    Given an IGRA dataframe, computes 
    saturation pressure w.r.t. water and ice,
    relative humidity w.r.t. ice, ISS and SAC satisfaction. 
    """
    dataframe['ew'] = hyland_wexler_water(dataframe['temp'].values)
    dataframe['ei'] = hyland_wexler_ice(dataframe['temp'].values)
    dataframe['RHi'] = dataframe['rhumi'].values*(dataframe['ew'].values/dataframe['ei'].values)
    dataframe['ISS'] = dataframe['RHi'].values > 1 
    dataframe['SAC'] = [check_SAC(p, rh, temp) for p, rh, temp in zip(dataframe['pres'], dataframe['rhumi'], dataframe['temp'])]
    return dataframe

def mixing_line_gradient(p, epsilon=0.622, EI_H20=1.23, LHV=42e6, cp=1005, eta=0.4):
    """ 
    Computes mixing line gradient 
    
    G = (cp*p/epsilon)*(EI_H20/(LHV*(1-eta)))
    
    Inputs:
    
    p: pressure in Pascal
    eps: ratio of molar mass of water to air 
    EI_H2O: emissions index water of fuel used 
    LHV: Lower heating value of fuel in J/kg
    cp: specific heat of air in J/(Kg*Kelvin)
    eta: efficiency of aircraft 
    
    Output:
    
    G: mixing line gradient in Pascal/Kelvin
    """
    
    return (cp*p/epsilon)*(EI_H20/(LHV*(1-eta)))






def check_SAC(p, RH, T, plot_result=False):
    """
    Checks whether the Schmidt-Appleman Criterion (SAC) is satisfied
    
    Inputs:
    
    p: Pressure in Pascal
    RH: Relative humidity w.r.t water in range [0,1]
    T: Temperature in Kelvin
    
    Output:
    
    boolean indicating whether SAC is satisfied. 
    """
    
    G = mixing_line_gradient(p)
    p_sat_w = hyland_wexler_water(T)
    
    mixing_line = lambda x: RH*p_sat_w + G*(x-T+273.15)
    
    derivative = lambda x: ((0.58002206*10**4 *x**-2  
              - 0.48640239*10**-1 
              + 2*0.41764768*10**-4 * x
              - 3*0.14452093*10**-7 * x**2 
              + 0.65459673*10**1  * x**-1 
                            
               )*hyland_wexler_water(x))
    
    # See Schumann (1996)
    initial_estimate = 273.15+-46.46 + 9.43*np.log(G-0.053) + 0.720*(np.log(G-0.053))**2
    T_LM = scipy.optimize.newton(lambda x: G - derivative(x), initial_estimate)
    
    SAC = (T < T_LM + (RH*p_sat_w-hyland_wexler_water(T_LM))/G)*(RH*p_sat_w < hyland_wexler_water(T))*(T < T_LM)
    if plot_result:
        fig, ax = plt.subplots(figsize=(15,10)) 
        temps = np.linspace(-60, -20, 100) 
        
        limiting_condition = lambda x: hyland_wexler_water(T_LM) + (x+273.15-T_LM)*G
        filtered_temps = temps[temps > T-273.15]
        
        # Set axis limits 
        ax.set_ylim([0,40])
        ax.set_xlim([-60, -25
                    ])
        # Plot saturation lines, mixing line, limiting condition 
        ax.plot(temps, hyland_wexler_water(temps + 273.15), 'b', label="Saturation line w.r.t. water")
        ax.plot(temps, hyland_wexler_ice(temps + 273.15), 'r', label="Saturation line w.r.t. ice")
        ax.plot(temps, limiting_condition(temps), 'k--', label="Limiting condition")
        ax.plot(filtered_temps, mixing_line(filtered_temps), 'g-.', label='Mixing line')
       
        
        
        
        
        # Mark regions of persistence and non-persistence
        ax.fill_between(temps, np.maximum(hyland_wexler_ice(temps + 273.15), limiting_condition(temps)), 
                        y2=hyland_wexler_water(temps+ 273.15), 
                        where=(temps < T_LM -273.15), 
                        alpha = 0.5, label="Persistent contrails")
        
        ax.fill_between(temps, limiting_condition(temps), 
                        y2=hyland_wexler_ice(temps+ 273.15), 
                        where=(limiting_condition(temps)< hyland_wexler_ice(temps+ 273.15))*(temps < T_LM -273.15), 
                        alpha = 0.5, label="Non-persistent contrails")
        
        # Plot environment point 
        ax.scatter([T-273.15],[ RH*p_sat_w], marker='x',c='g',s=200)
        ax.annotate("Environment", (T-273.15 , RH*p_sat_w-1.5))
        
        ax.set(xlabel=r"Temperature [${}^\circ$C]", ylabel="Saturation vapor pressure [Pa]")
        ax.legend()        
    return  SAC
    
#########################################################
## Correction algorithm following Dirksen et al (2014) ##
#########################################################

# Radiation correction
def radcorr( cur_lat, cur_lon, cur_time, cur_p, cur_T, cur_RH ):
    
    # Get the temperature change
    dT = delT( cur_lat, cur_lon, cur_time, cur_p, cur_T )
    Tcorr = cur_T - dT
    
    # Get the RH change
    RHcorr = delRH( cur_RH, Tcorr, dT )
    
    return Tcorr, RHcorr

# Temperature change
def delT( cur_lat, cur_lon, cur_time, cur_p, cur_T ):
    
    # Assumptions
    v = 5 # Ventilation speed [m/s]
    a = 0.18 # Constant a for Eq. 1, Dirksen et al (2014)
    b = 0.55 # Constant b for Eq. 1, Dirksen et al (2014)
    
    # Estimate solar elevation angle
    if np.size(cur_time)>1:
        for ii, cur_time_val in enumerate(cur_time):
            solrelev_angle = pys.get_altitude_fast( cur_lat, cur_lon, pd.Timestamp(cur_time_val, tzinfo=dt.timezone.utc) )
    else:
        solrelev_angle = pys.get_altitude_fast( cur_lat, cur_lon, pd.Timestamp(cur_time, tzinfo=dt.timezone.utc) )
    
    # Calculate actinic flux by linear interpolation
    if solrelev_angle > 5:
        actFlux_interp = actFlux_lookup()
        Ia = actFlux_interp( cur_p, solrelev_angle )
    else:
        Ia = 0
    
    # Calculate temperature change using Eq. 1, Dirksen et al (2014)
    x = Ia / ( cur_p*v )
    dT = a*x**b
    
    return dT

# RH change
def delRH( cur_RH, Tcorr, dT ):
    
    # Assumptions
    f = 6.5 # Sensitivity of humidity sensor vs T sensor (Dirksen et al (2014))
    
    # Calculate corrected RH
    RHcorr = cur_RH * src.Psat_h2ol( Tcorr + f*dT ) / src.Psat_h2ol( Tcorr )
    
    return RHcorr

# Actinic flux estimate
def actFlux_lookup( ):
    # Generate the lookup function
    # Note: values have been read off Fig. 5, Dirksen et al (2014)
    # These should be improved before publication...
    
    # Lookup variables
    # altitude = np.array( [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30 ] ) # km
    pressure = np.array( [ 1013, 899, 795, 701, 617, 541, 471, 411, 357, 308, 265, \
                           225, 192, 165, 141, 121, 55, 25, 12 ] ) # hPa
    solrelev = np.array( [ 30, 72, 90 ] ) # degrees

    # Average values
    actFlux_vals = np.array( [ [ 280, 300, 400, 525, 535, 550, 565, 580, 600, 625,       \
                                 650, 850, 855, 860, 865, 870, 880, 890, 895 ],          \
                               [ 450, 455, 460, 465, 470, 600, 1275, 1280, 1350, 1450,   \
                                 1490, 1490, 1490, 1485, 1480, 1480, 1475, 1475, 1475 ], \
                               [ 490, 495, 500, 505, 510, 950, 1500, 1500, 1600, 1700,   \
                                 1300, 1295, 1290, 1285, 1280, 1275, 1270, 1265, 1265 ] ] )
    
    # Interpolation function
    actFlux_interp = spy_interp.interp2d( pressure, solrelev, actFlux_vals, kind='linear' )
    
    return actFlux_interp


    
    
    
    
    
