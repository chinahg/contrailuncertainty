## Library of functions to run for NOAA and/or GRUAN sonde + compare to GEOS-FP and/or MERRA-2 reanalysis data

# Imports
import numpy as np
import os as os
from glob import glob
import re
import datetime as dt
import ftplib as ftp
import netCDF4 as nc
from io import BytesIO
import pandas as pd
import datetime as dt
import itertools as itl
import cdsapi
import urllib
import scipy

## Meterology conversion functions

# Pressure to altitude (https://www.mide.com/air-pressure-at-altitude-calculator)
def P2Alt ( P, T ):
    alt0 = 0 # [m] Initial elevation
    T0 = 288 # [K] Temperature at initial elevation
    L = -0.0065 # [K/m] Lapse rate
    P0 = 101325 # [Pa] Pressure at initial elevation
    R =  8.31432 # [N m / mol K] Universal gas constant
    g = 9.80665 # [m/s^2] Gravitational constant
    M = 0.0289644 # [kg/mol] Molar mass of Earth's air
    
    alt = 100 * (alt0 + T0/L * ((P/100/P0)**(-R*L/(g*M)) -1)) /1000
    return alt

# Liquid saturation pressure
def Psat_h2ol ( T ):
    return 100 * np.exp( -6096.9385/T + 16.635794 - 0.02711193*T  + 1.673952E-5*T**2 + 2.433502  *np.log(T))

# Ice saturation pressure
def Psat_h2os ( T ):
    return 100 * np.exp( -6024.5282/T + 24.7219   + 0.010613868*T - 1.319883E-5*T**2 - 0.49382577*np.log(T));

# RHi from mixing ratio and temperature
def QT2RHi( QV, TEMP, P ): # QV = kg/kg; T = K; P = hPa
    QVsat = 0.622 * Psat_h2os(TEMP) / ( P*100 )
    RHi = 100 * QV / QVsat
    return RHi

# RHw to RHi
def RHw2RHi( RH, T ):
    RHi = RH * Psat_h2ol(T) / Psat_h2os(T)
    return RHi

# RHi to RHw
def RHi2RHw( RHi, T):
    RH = RHi * Psat_h2os(T) / Psat_h2ol(T)
    return RH

# SAC formulation
def SAC_general( RH, T, P, EIH2O=1.23, cp=1004, LHV=43.13, eta0=0.4, RHic=100, printout=0 ):
    # For given scalar or vector values of RH, T and P, does a contrail form?
    
    # Calculate SAC constraints
    G = ( EIH2O*cp*P*100 ) / ( 0.622*LHV*1E6*(1-eta0) )
    Tc = 226.69 + 9.43*np.log(G-0.053) + 0.72*(np.log(G-0.053))**2
    RHc = 100 * ( G*(T-Tc) + Psat_h2ol( Tc ) ) / Psat_h2ol( T )
    
    # Calculate RHw
    RHi = RHw2RHi( RH, T )
    # RH = RHi2RHw( RHi, T )
    
    # Identify if pass
    SAC = ((T<Tc) & (RH>RHc) & (RHi>RHic) & (RH<100)) * 1
    
    return SAC

# Uncertainty of PCC
def unc_PCC( RH, T, P, u_RH, u_T, EIH2O=1.23, cp=1004, LHV=43.13, eta0=0.4 ):
    
    # Calculate SAC constraints
    G = ( EIH2O*cp*P*100 ) / ( 0.622*LHV*1E6*(1-eta0) )
    Tc = 226.69 + 9.43*np.log(G-0.053) + 0.72*(np.log(G-0.053))**2
    RHc = 100 * ( G*(T-Tc) + Psat_h2ol( Tc ) ) / Psat_h2ol( T )
    
    # Calculate RHi
    RHi = RHw2RHi( RH, T )
    
    # Calculate each aux variable
    X1 = Tc - T
    X2 = ( RH - RHc )/100
    X3 = 1 - RH/100
    X4 = RHi/100 - 1
    
    # Calculate individual uncertainties
    dRHc_dT = calc_dRHc_dT( T, P, EIH2O=EIH2O, cp=cp, LHV=LHV, eta0=eta0 )
    dX4_dT  = calc_dX4_dT( T )
    dX4_dRH = Psat_h2ol(T)/Psat_h2os(T)
    
    # Combine uncertainties
    dX1 = u_T
    dX2 = ( u_RH**2 + (dRHc_dT*u_T)**2 )**0.5
    dX3 = 0
    dX4 = ( (dX4_dT*u_T)**2 + (dX4_dRH*u_RH)**2 )**0.5
    
    # Calculate probability PCC for each Xi
    P_X1 = 1 - scipy.stats.norm.cdf( -X1/dX1 )
    P_X2 = 1 - scipy.stats.norm.cdf( -X2/dX2 )
    P_X3 = 1 - scipy.stats.norm.cdf( -X3/dX3 )
    P_X4 = 1 - scipy.stats.norm.cdf( -X4/dX4 )
    
    # Combine probabilities together
    P_PCC = P_X1 * P_X2 * P_X3 * P_X4
    
    return P_PCC

# Calculate derivatives of critical RH wrt T
def calc_dRHc_dT( T, P, EIH2O=1.23, cp=1004, LHV=43.13, eta0=0.4 ):
    
    # Calculate input values
    G = ( EIH2O*cp*P*100 ) / ( 0.622*LHV*1E6*( 1 - eta0 ) )
    Tc = 226.69 + 9.43*np.log( G - 0.053 ) + 0.72*( np.log( G - 0.053 ) )**2
    
    # Resulting derivative
    dRHc_dT = G/Psat_h2ol(T) + ( G*( T - Tc ) + Psat_h2ol(Tc) )/( Psat_h2ol(Tc)**2 )*dPsat_h2ol_dT(T)
    
    return dRHc_dT

# Calculate derivative of auxiliary variable 4 (X4)
def calc_dX4_dT( T ):
    
    dX4_dT = 1/Psat_h2os(T)*dPsat_h2ol_dT(T) + Psat_h2ol(T)/Psat_h2os(T)**2*dPsat_h2os_dT(T)
    
    return dX4_dT

def dPsat_h2ol_dT( T ):
        
    # Constants
    A1 = 6024.5282
    A2 = 24.7219
    A3 = 0.010613868
    A4 = 1.319883E-5
    A5 = 0.49382577
    
    # Calculate gradient
    return 100*( A1/T**2 + A3 - 2*A4*T - A5/T )*np.exp( -A1/T + A2 + A3*T  - \
                                                        A4*T**2 - A5*np.log(T) )

def dPsat_h2os_dT( T ):
    
    # Constants
    A1 = 6096.9385
    A2 = 16.635794
    A3 = 0.02711193
    A4 = 1.673952E-5
    A5 = 2.433502
    
    # Calculate gradient
    return 100*( A1/T**2 - A3 + 2*A4*T + A5/T )*np.exp( -A1/T + A2 - A3*T  + \
                                                        A4*T**2 + A5*np.log(T) )

# SAC between altLo and altHi
def SAC( df_sonde, EIH2O=1.23, cp=1004, LHV=43.13, eta0=0.4, RHic=100, altLo=8, altHi=12,xCont=100 ):
    # Extract from dataframe
    T = df_sonde['T']
    RH = df_sonde['RH']
    P = df_sonde['P']
    altitude = df_sonde['altitude']
    # Calculate SAC constraints
    G = ( EIH2O*cp*P*100 ) / ( 0.622*LHV*1E6*(1-eta0) )
    Tc = 226.69 + 9.43*np.log(G-0.053) + 0.72*(np.log(G-0.053))**2
    RHc = 100 * ( G*(T-Tc) + Psat_h2ol( Tc ) ) / Psat_h2ol( T )
    # Calculate RHi
    RHi = df_sonde['RHi'] # RHw2RHi( RH, T )
    # Identify if pass SAC within altitude range
    SACall = ((T<Tc) & (RH>RHc) & (RHi>RHic)) * 1
    altIdx = df_sonde["altitude"].between(altLo, altHi, inclusive = True) 
    SAC_cur = SACall[altIdx]
    SACcounter = cont_ISS( df_sonde, SACall*100, altLo, altHi, xCont )
    SACpass = ( SACcounter>=1 ) * 1
    return SACpass, SACall

# Convert dewpoint depression to RH using Bolton (1980)
def dpd2RH( dpd, T ):
    
    # Calculate dewpoint depression
    Td = T - dpd
    
    # Use Bolton's equation to convert dewpoint depression to vapor pressure
    a = 611.2
    b = 17.67
    c = 243.5
    p_wv = a*np.exp( b*Td / ( c + Td ) )
    
    # Use saturation vapor pressure to get RH
    psat_wv = Psat_h2ol( T + 273.15 )
    RH = 100 * p_wv / psat_wv
    
    return RH

# Identify if any x m vertical layer is continouosly ISS
def cont_ISS( df_sonde, RHi, altLow, altHig, xCont ):
    # Identify ISS regions
    alt_idx = df_sonde["altitude"].between(altLow, altHig, inclusive = True) 
    # alt_idx = np.argwhere( np.logical_and(altitude>=altLow, altitude<=altHig) )
    alt_ = df_sonde["altitude"].values[alt_idx]*1000
    RHi_ = RHi[alt_idx]
    ISS = (RHi_>=100)*1
    # Loop over each group of true vs false values
    ISS_counter = 0
    cur_idx = 0
    for k, g in itl.groupby(ISS):
        g = list(g)
        # Identify distance between first and last member of group
        nxt_idx = cur_idx + len(g)
        if nxt_idx >= len(alt_):
            nxt_idx = len(alt_)-1
        alt_cur = alt_[cur_idx]
        alt_nxt = alt_[nxt_idx]
        alt_range = alt_nxt - alt_cur
        if alt_range >= xCont and k==1:
            ISS_counter = ISS_counter+1
        # Iterate idx
        if nxt_idx == len(alt_)-1:
            break
        cur_idx = nxt_idx
    return ISS_counter

# Split sonde data into super and sub-saturated by wide altitude widths
def split_ISS( RHi, altitude, altitude_binedges ):
    
    # Initialize
    nbins = len(altitude_binedges)-1
    total_counts = np.zeros(nbins)
    ISS_counts = np.zeros(nbins)
    
    # Loop over each bin
    for iedge in range(nbins-2):
        
        # Define bin edges
        altbinlow = altitude_binedges[iedge]
        altbinhig = altitude_binedges[iedge+1]
        
        # Find RHi's within this region
        idx = (altitude>=altbinlow) * (altitude<altbinhig)
        RHi_cur = RHi[idx]
        
        # Use RHi to get counts
        ISS_counts[iedge] = ISS_counts[iedge] + np.sum( RHi_cur>100 )
        total_counts[iedge] = total_counts[iedge] + len(RHi_cur)
        
    pISS = ISS_counts / total_counts
    
    return pISS, ISS_counts, total_counts

# Split sonde data into super and sub-saturated by wide altitude widths
def split_ISS_pres( RHi, pressure, pressure_binedges ):
    
    # Initialize
    nbins = len(pressure_binedges)-1
    total_counts = np.zeros(nbins)
    ISS_counts = np.zeros(nbins)
    
    # Loop over each bin
    for iedge in range(nbins-2):
        
        # Define bin edges
        prebinlow = pressure_binedges[iedge]
        prebinhig = pressure_binedges[iedge+1]
        
        # Find RHi's within this region
        idx = (pressure>=prebinlow) * (pressure<prebinhig)
        RHi_cur = RHi[idx]
        
        # Use RHi to get counts
        ISS_counts[iedge] = ISS_counts[iedge] + np.sum( RHi_cur>100 )
        total_counts[iedge] = total_counts[iedge] + len(RHi_cur)
        
    pISS = ISS_counts / total_counts
    
    return pISS, ISS_counts, total_counts

# Define bin edges if standard altitude bins required
def bin_edge_def( zBot=5, zTop=16, dz=0.2, zMax=100 ):
    altitude_binedges = np.array(0)
    altitude_binedges = np.append( altitude_binedges, np.arange(zBot, zTop, dz) )
    altitude_binedges = np.append( altitude_binedges, [zTop, zTop+dz*20, zMax] )
    altitude_bincent = 0.5 * ( altitude_binedges[0:-1] + altitude_binedges[1:] )
    altitude_binwidth = altitude_binedges[1:] - altitude_binedges[0:-1]
    return altitude_binedges, altitude_bincent, altitude_binwidth

# Define bin edges if standard pressure bins required
def bin_edge_def_pres( pBot=550, pTop=25, nBins=50, pMax=0 ):
    
    # Start the basic bin edges vector
    pressure_binedges = np.array(pMax)
    
    # Convert the pressures to a power for np.logspace
    pBot_pow = np.log10( pBot )
    pTop_pow = np.log10( pTop )
    
    # Fill in binedges assuming logspace until pTop
    pressure_binedges = np.append( pressure_binedges, np.logspace( pTop_pow, pBot_pow, nBins ) )
    pressure_binedges = np.append( pressure_binedges, [ 1200 ] )
    
    # Estimate centers and bin widths
    pressure_bincent = 0.5 * ( pressure_binedges[0:-1] + pressure_binedges[1:] )
    pressure_binwidth = np.abs( pressure_binedges[1:] - pressure_binedges[0:-1] )
    
    return pressure_binedges, pressure_bincent, pressure_binwidth

# Define bin edges using baseline merra2 grid
def bin_edge_def_pres_alt( ):
    
    # Standard merra2 vertical pressure distribution
    pressure_binedges, pressure_bincent = geosfp_press( 1013.25 )
    
    # Estimate bin widths
    pressure_binwidth = np.abs( pressure_binedges[1:] - pressure_binedges[0:-1] )
    
    return pressure_binedges, pressure_bincent, pressure_binwidth

## Reanalysis data related functions

# Define geos-fp grid
def grid_setup( geos_type, folderpath ):
    gfp_folderpath = folderpath + '2015/01/'
    if geos_type == 'GEOS_FP':
        gfp_filename = 'GEOSFP.20150101.I3.025x03125.nc'
    else:
        gfp_filename = 'MERRA2.20150101.I3.05x0625.nc4'
    nc_geosfp = nc.Dataset( gfp_folderpath + gfp_filename )
    ntime = len(nc_geosfp.dimensions['time'])
    nlat = len(nc_geosfp.dimensions['lat'])
    nlon = len(nc_geosfp.dimensions['lon'])
    nlev = len(nc_geosfp.dimensions['lev'])
    time0 = nc_geosfp.variables['time'][0]
    lat0 = nc_geosfp.variables['lat'][0]
    lon0 = nc_geosfp.variables['lon'][0]
    lev0 = nc_geosfp.variables['lev'][0]
    dtime = nc_geosfp.variables['time'][1] - time0
    dlat = nc_geosfp.variables['lat'][10] - nc_geosfp.variables['lat'][9]
    dlon = nc_geosfp.variables['lon'][10] - nc_geosfp.variables['lon'][9]
    dlev = nc_geosfp.variables['lev'][10] - nc_geosfp.variables['lev'][9]
    if geos_type == 'GEOS_FP':
        lat0 = lat0 - dlat/2
    nc_geosfp.close()
    return ntime, nlat, nlon, nlev, time0, lat0, lon0, lev0, dtime, dlat, dlon, dlev

# Get pressure given surface level pressure (scalar only)
def geosfp_press( PS, dims=None ): # Returns geos pressure edge and center at the lat/lon location
    Ap = np.zeros((72))
    Bp = np.zeros((72))
    Ap = np.array( [ 0.000000E+00, 4.804826E-02, 6.593752E+00, 1.313480E+01, 1.961311E+01, 2.609201E+01,
    3.257081E+01, 3.898201E+01, 4.533901E+01, 5.169611E+01, 5.805321E+01, 6.436264E+01,
    7.062198E+01, 7.883422E+01, 8.909992E+01, 9.936521E+01, 1.091817E+02, 1.189586E+02,
    1.286959E+02, 1.429100E+02, 1.562600E+02, 1.696090E+02, 1.816190E+02, 1.930970E+02,
    2.032590E+02, 2.121500E+02, 2.187760E+02, 2.238980E+02, 2.243630E+02, 2.168650E+02,
    2.011920E+02, 1.769300E+02, 1.503930E+02, 1.278370E+02, 1.086630E+02, 9.236572E+01,
    7.851231E+01, 6.660341E+01, 5.638791E+01, 4.764391E+01, 4.017541E+01, 3.381001E+01,
    2.836781E+01, 2.373041E+01, 1.979160E+01, 1.645710E+01, 1.364340E+01, 1.127690E+01,
    9.292942E+00, 7.619842E+00, 6.216801E+00, 5.046801E+00, 4.076571E+00, 3.276431E+00,
    2.620211E+00, 2.084970E+00, 1.650790E+00, 1.300510E+00, 1.019440E+00, 7.951341E-01,
    6.167791E-01, 4.758061E-01, 3.650411E-01, 2.785261E-01, 2.113490E-01, 1.594950E-01,
    1.197030E-01, 8.934502E-02, 6.600001E-02, 4.758501E-02, 3.270000E-02, 2.000000E-02,
    1.000000E-02  ] )
    Bp = np.array( [ 1.000000E+00, 9.849520E-01, 9.634060E-01, 9.418650E-01, 9.203870E-01, 8.989080E-01,
    8.774290E-01, 8.560180E-01, 8.346609E-01, 8.133039E-01, 7.919469E-01, 7.706375E-01,
    7.493782E-01, 7.211660E-01, 6.858999E-01, 6.506349E-01, 6.158184E-01, 5.810415E-01,
    5.463042E-01, 4.945902E-01, 4.437402E-01, 3.928911E-01, 3.433811E-01, 2.944031E-01,
    2.467411E-01, 2.003501E-01, 1.562241E-01, 1.136021E-01, 6.372006E-02, 2.801004E-02,
    6.960025E-03, 8.175413E-09, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00,
    0.000000E+00  ] )
    if np.ndim( PS )<=1:
        Pedge = (Ap + [ Bp * PS ]).flatten()
        Pcent = np.zeros((72))
        for ii in range(len(Pedge)-1):
            Pcent[ii] = 0.5 * ( Pedge[ii] + Pedge[ii+1] )
    else:
        Nt, Np, Nlat, Nlon = dims
        Pedge = np.zeros( ( Nt, Np+1, Nlat, Nlon ) )
        PS = np.expand_dims(np.squeeze(PS), axis=1)
        for ii, Api in enumerate( Ap ):
            Bpi = Bp[ii]
            Pedge[:,ii,:,:] = np.squeeze( Api + Bpi * PS )
        Pcent = np.zeros( dims )
        for ii in range(len(Ap)-1):
            Pcent[:,ii,:,:] = 0.5 * ( Pedge[:,ii,:,:] + Pedge[:,ii+1,:,:] )
    return Pedge, Pcent

## GRUAN related functions

# Open and extract data from netcdf files
def nc_sonde_extract( nc_sonde ):
    df_sonde = pd.DataFrame()
    try:
        df_sonde['time'] = nc_sonde.variables['time'][:] # s
    except:
        time_gmt = 0
        skip_case = 1
        return df_sonde, time_gmt, skip_case
    time_gmt = nc_sonde.variables['time'].units
    time_gmt = dt.datetime.strptime(time_gmt,"seconds since %Y-%m-%dT%H:%M:%S")
    if 'press' not in nc_sonde.variables.keys():
        skip_case = 1
        return df_sonde, time_gmt, skip_case
    df_sonde['P'] = nc_sonde.variables['press'][:] # hPa
    df_sonde['T'] = nc_sonde.variables['temp'][:] # Kelvin
    df_sonde['longitude'] = nc_sonde.variables['lon'][:]
    df_sonde['latitude'] = nc_sonde.variables['lat'][:]
    df_sonde['altitude'] = nc_sonde.variables['alt'][:]/1000 # km
    df_sonde['RH'] = nc_sonde.variables['rh'][:]*100 # Percent
    df_sonde['Tunc'] = nc_sonde.variables['u_cor_temp'][:]
    df_sonde['Tcor'] = nc_sonde.variables['cor_temp'][:]
    df_sonde['RHunc'] = nc_sonde.variables['u_cor_rh'][:]*100
    df_sonde['RHcor'] = nc_sonde.variables['cor_rh'][:]
    df_sonde.dropna( subset=['time'], inplace=True )
    df_sonde['Time GMT'] = [time_gmt + dt.timedelta(seconds=i) for i in df_sonde['time']]
    df_sonde.reset_index( drop=True, inplace=True )
    # Check return is reasonable
    latnan = df_sonde.latitude.isnull().all()
    lonnan = df_sonde.longitude.isnull().all()
    altnan = df_sonde.altitude.isnull().all()
    if latnan or lonnan or altnan:
        print('All nans in lat, lon or alt')
        skip_case = 1
    else:
        skip_case = 0
    return df_sonde, time_gmt, skip_case

# Retrieve specific ECMWF
def retECMWF( time, savepath, load=True ):
    
    # Create cdsapi client
    c = cdsapi.Client()
    
    # Round time and extract date and time
#     print( time )
    time = hour_rounder( time )
    time_save = "T{:02d}00".format(time.hour)
    time_str = "{:02d}:00".format(time.hour)
    year = str( time.year )
    month = str(time.month).zfill(2)
    day = str(time.day).zfill(2)
    
    # Identify name to save and check if exists
    savename = savepath + 'ECMWF' + year + month + day + time_save + '.nc'
#     print(savename)
    if (os.path.exists( savename )) & (load==False): 
        # print('Data exists')
        return savename
    else:
        print( 'Loading, ' + year + month + day + time_save )
        if load:
            # Retrieve data and return path
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type':'reanalysis',
                    'format':'netcdf',
                    'pressure_level':[
                        '1','2','3',
                        '5','7','10',
                        '20','30','50',
                        '70','100','125',
                        '150','175','200',
                        '225','250','300',
                        '350','400','450',
                        '500','550','600',
                        '650','700','750',
                        '775','800','825',
                        '850','875','900',
                        '925','950','975',
                        '1000'
                    ],
                    'year':year,
                    'month':month,
                    'day':day,
                    'time':[
                        time_str
                    ],
                    'variable':[
                        'specific_humidity','temperature','specific_cloud_ice_water_content','fraction_of_cloud_cover'
                    ]
                },
                savename )
    
    return savename

# Round to nearest hour
def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +dt.timedelta(hours=t.minute//30))

# Extract data from ECMWF
def extractECMWF( filename, df_sonde ):
    
    # Open the dataset and extract basic variables (longitude, latitude, pressure)
    nc_test = nc.Dataset( filename )
    longitude = nc_test.variables['longitude'][:] - 180 # Shift so from -180 to +180
    latitude = nc_test.variables['latitude'][:]
    pressure = nc_test.variables['level'][:]
    
    # Estimate pressure edges from pressure mid points
    pressure_edge = np.zeros( len(pressure)+1 )
    pressure_edge[-1] = 1200 # Use unreasonably large value to treat as surface edge
    pressure_edge[1:-1] = 0.5*( pressure[0:-1] + pressure[1:] )
    
    # Identify matching longitude and latitude based on sonde starting position
    lon_match = np.argmin( np.abs( longitude - df_sonde.longitude[0] ) )
    lat_match = np.argmin( np.abs( latitude  - df_sonde.latitude[0]  ) )
    
    # Get variables from dataset
    SH = np.squeeze( nc_test.variables['q'][:, :, lat_match, lon_match] )
    T = np.squeeze( nc_test.variables['t'][:, :, lat_match, lon_match] )
    CC = np.squeeze( nc_test.variables['cc'][:, :, lat_match, lon_match] )
    
    # Calculate RHw and RHi
    RHi = QT2RHi( SH, T, pressure )
    RHw = RHi2RHw( RHi, T)
    
    # Calculate clear sky RHi
    RHi_clr = ( RHi - CC ) / ( 100 - CC )
    
    # Close the netcdf file
    nc_test.close()
    
    return pressure, SH, T, RHi, RHw, pressure_edge, CC


## NOAA related functions

# Open sonde data text file as dataframe
def sonde_dataframe( filename ):
    iHeader = headerline( filename )
    columns = pd.read_csv(filename, sep=',', header=iHeader, skipinitialspace=True).columns
    df_sonde = pd.read_csv(filename, sep=',', header=iHeader+2, names=columns, na_values=99999, skipinitialspace=True)
    # Add RHi to df and date/time
    df_sonde['RHi'] = df_sonde['RH'] * Psat_h2ol(df_sonde['Temp']+273.15) / Psat_h2os(df_sonde['Temp']+273.15)
    df_sonde['Time GMT'] = pd.to_datetime(df_sonde['Time GMT'])
    return df_sonde

# Identify location of header file
def headerline( filename ):
    f = open(filename,"r")
    fhead = []
    for ii, line in enumerate(f):
        fhead.append(line)
        if line == '\n':
            break
    #iHeader = ii
    return ii


## Comparing sonde data (in dataframe only) to reanalysis data

# Average sonde data onto reanalysis layers
def sonde_averaging( df_sonde, P, Pedge, T, RHi, SAC, df_store, sonde_filename ):
    
    # Get location and datetime of current sonde
    split_filename = sonde_filename.split('_')
    location = split_filename[0].split('-')[0]
    datetime_dt = dt.datetime.strptime(split_filename[4],'%Y%m%dT%H%M%S')
    
    # Define sonde pressure edges
    Psonde = df_sonde['P'].as_matrix()
    Pedge_sonde = np.zeros( len(Psonde)+1 )
    Pedge_sonde[0] = Psonde[0] + ( Psonde[0] - Psonde[1] )/2
    Pedge_sonde[1:-1] = 0.5*( Psonde[0:-1] + Psonde[1:] )
    Pedge_sonde[-1] = Psonde[-1] + ( Psonde[-1] - Psonde[-2] )/2
    
    # Get dP of each sonde "layer"
    dP_sonde = Pedge_sonde[0:-1] - Pedge_sonde[1:]
    
    # Zero out some values
    Psonde_ave = np.zeros( len(P) )
    RHisonde_ave = np.zeros( len(P) )
    RHisonde_std = np.zeros( len(P) )
    RHsonde_ave = np.zeros( len(P) )
    Tsonde_ave = np.zeros( len(P) )
    Tsonde_std = np.zeros( len(P) )
    SACsonde_ave = np.zeros( len(P) )
    SACsonde_pct = np.zeros( len(P) )
    ISSsonde_ave = np.zeros( len(P) )
    ISSsonde_pct = np.zeros( len(P) )
    
        # Loop over each merra2 layer
    for layer, Pcent in enumerate(P):

        # Get sonde points between in this layer
        sonde_idx = (Psonde>=Pedge[layer]) & (Psonde<Pedge[layer+1])

        # Get current values in sonde
        Psonde_cur = Psonde[sonde_idx]
        Tsonde_cur = df_sonde['T'][sonde_idx]
        RHisonde_cur = df_sonde['RHi'][sonde_idx]
        SACsonde_cur = df_sonde['SAC'][sonde_idx]

        # Calculate averages for P and T
        Psonde_ave[layer] = np.mean( Psonde_cur )
        Tsonde_ave[layer] = np.mean( Tsonde_cur )
        Tsonde_std[layer] = np.std( Tsonde_cur )

        # Get the average vapor pressure and saturation vapor pressure
        p_wv = RHisonde_cur * Psat_h2os(  Tsonde_cur )
        psat_ice = Psat_h2os(  Tsonde_cur )
        psat_wv = Psat_h2ol(  Tsonde_cur )

        # Pressure weight average the vapor pressures and use to calculate average RHi
        p_wv_ave = np.sum( dP_sonde[sonde_idx]*p_wv ) / np.sum( dP_sonde[sonde_idx] )
        psat_ice_ave = np.sum( dP_sonde[sonde_idx]*psat_ice ) / np.sum( dP_sonde[sonde_idx] )
        psat_wv_ave = np.sum( dP_sonde[sonde_idx]*psat_wv ) / np.sum( dP_sonde[sonde_idx] )
        RHisonde_ave[layer] = p_wv_ave/psat_ice_ave
        RHsonde_ave[layer] = p_wv_ave/psat_wv_ave

        # Proportion of vertical grid satisfying SAC
        SACsonde_ave[layer] = SAC_general( RHsonde_ave[layer], Tsonde_ave[layer], Psonde_ave[layer] )
        SACsonde_pct[layer] = np.mean( SACsonde_cur )

        # Proportion of vertical grid ISS
        ISSsonde_ave[layer] = ( (RHsonde_ave[layer]>=100) & (Tsonde_ave[layer]<233.15) )*1
        ISSsonde_pct[layer] = np.mean( ( (RHisonde_cur>=100) & (Tsonde_cur<233.15) )*1 )
        
        # Store information in dataframe
        # Pass a series in append() to append a row in dataframee
        df_store = df_store.append( pd.Series( [ location, datetime_dt, Pcent, Pedge[layer], Pedge[layer+1], \
                                                 T[layer], RHi[layer], SAC[layer],            \
                                                 Tsonde_ave[layer], np.std( Tsonde_cur ),     \
                                                 RHisonde_ave[layer], np.std( RHisonde_cur ), \
                                                 SACsonde_ave[layer], SACsonde_pct[layer],    \
                                                 ISSsonde_ave[layer], ISSsonde_pct[layer], ]  \
                                              , index=df_store.columns ) \
                                    , ignore_index=True)
        
    return df_store

# Load in MERRA-2 data for a given sonde's starting position
def loadMERRA2_base( geos_type, geos_path, df_sonde, geos_html ):
    
    # Datetime for extracting MERRA2 data
    datetime_merra2 = round_dt(df_sonde['Time GMT'][0], 3)
    
    # File path
    year = str( datetime_merra2.year )
    month = str( datetime_merra2.month ).zfill(2)
    day = str( datetime_merra2.day ).zfill(2)
    geosfp_fullpath = geos_path + year + '/' + month + '/'
    
    # Identify grid information
    ntime, nlat, nlon, nlev, time0, lat0, lon0, lev0, dtime, dlat, dlon, dlev = grid_setup( geos_type, geos_path )
    
    # Identify filename
    if geos_type == 'GEOS_FP':
        gfp_filename = 'GEOSFP.' + year + month + day + '.I3.025x03125.nc'
        gfp_filenamedyn = 'GEOSFP.' + year + month + day + '.A3dyn.025x03125.nc'
    else:
        gfp_filename = 'MERRA2.' + year + month + day + '.I3.05x0625.nc4'
        gfp_filenamedyn = 'MERRA2.' + year + month + day + '.A3dyn.05x0625.nc4'
    
    # Pointers to open netcdf files
    geos_html = 'http://geoschemdata.computecanada.ca/ExtData/GEOS_0.5x0.625/MERRA2/'
    geosfp_fullpath2 = retMERRA2( datetime_merra2, geos_type, geos_html, geos_path )
    nc_geosfp = nc.Dataset( geosfp_fullpath + gfp_filename )
    
    # Get cloud data
    geoscld_savepath = retMERRA2cld( datetime_merra2, geos_type, geos_html, geos_path )
    nc_geosfpcld = nc.Dataset( geoscld_savepath )
    
    # Convert first lat/lon to geos-fp grid indices
    lat_first = df_sonde.latitude.loc[~df_sonde.latitude.isnull()].iloc[0]
    lon_first = df_sonde.longitude.loc[~df_sonde.longitude.isnull()].iloc[0]
    lat_index = np.ceil( (lat_first-lat0)/dlat )
    lon_index = np.ceil( (lon_first-lon0)/dlon )
    
    # Calculate time to geos-fp grid index
    now = datetime_merra2
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = (now-midnight).seconds/60
    time_index = np.ceil( (minutes-time0)/dtime )
    if time_index == 8.0:
        time_index = 7.0
    
    # Get pressure information for a column
    PS = nc_geosfp.variables['PS'][ time_index, int(lat_index), int(lon_index) ] * 1/100
    Pedge, P = geosfp_press( PS )
    
    # Get column data and calculate RHi
    QV = nc_geosfp.variables['QV'][ time_index, :, lat_index, lon_index ]
    T  = nc_geosfp.variables['T'][ time_index, :, lat_index, lon_index ]
    CC = nc_geosfpcld.variables[ 'CLOUD' ][ time_index, :, lat_index, lon_index ]
    QI = nc_geosfpcld.variables[ 'QI' ][ time_index, :, lat_index, lon_index ]

    # Convert to RHi
    RHi = QT2RHi( QV, T, P )
    RH = RHi2RHw( RHi, T )
    
    # RHi if ice cloud was water
    RHi_cld = QT2RHi( QV+QI, T, P )
    
    # Close the nc pointers
    nc_geosfp.close()
    
    return RHi, RH, T, P, QV, Pedge, RHi_cld

def retMERRA2cld( time, geos_type, geos_html, geos_path ):
    
    # Get the current date
    time = hour_rounder( time )
    time_str = "{:02d}:00".format(time.hour)
    year = str( time.year )
    month = str(time.month).zfill(2)
    day = str(time.day).zfill(2)
    
    # Define file name and paths
    geoscld_name = geos_type + '.' + year + month + day + '.A3cld.05x0625.nc4'
    geoscld_htmlpath = geos_html + year + '/' + month + '/' + geoscld_name
    geoscld_savepath = geos_path + year + '/' + month + '/' + geoscld_name
    
    # Check existence of file
    if os.path.exists( geoscld_savepath ): 
        print('Data exists')
        return geoscld_savepath
    
    # Use wget to download
    if not os.path.exists( geos_path + year + '/' + month + '/' ):
        print('Creating folder')
        os.mkdir( geos_path + year + '/' + month + '/' )
    os.system( "wget -O " + geoscld_savepath  + " " + geoscld_htmlpath )
    
    return geoscld_savepath

def retMERRA2( time, geos_type, geos_html, geos_path, load=True ):
    # Inputs:
    #   - load = 'True': Do load data if not exists
    
    # Get the current date
    time = hour_rounder( time )
    time_str = "{:02d}:00".format(time.hour)
    year = str( time.year )
    month = str(time.month).zfill(2)
    day = str(time.day).zfill(2)
    
    # Define file name and paths
    geoscld_name = geos_type + '.' + year + month + day + '.I3.05x0625.nc4'
    geoscld_htmlpath = geos_html + year + '/' + month + '/' + geoscld_name
    geoscld_savepath = geos_path + year + '/' + month + '/' + geoscld_name
    
    # Check existence of file
    if load:
        
        if os.path.exists( geoscld_savepath ):

            # Get local and remote filesize
            site = urllib.request.urlopen( geoscld_htmlpath )
            meta = site.info()
            filesize_remote = int( meta.get("Content-Length") )
            filesize_local = os.path.getsize( geoscld_savepath )

            # If filesize too small then reload, else exit!
            if filesize_local >= filesize_remote*0.9:
                # print('Data exists')
                return geoscld_savepath
            else:
                os.remove( geoscld_savepath )
                print('Re-loading file: ', filesize_remote, filesize_local )

        # Check existence of folder
        if not os.path.exists( geos_path + year + '/' + month + '/' ): 
            # print('Creating folder')
            os.mkdir( geos_path + year + '/' + month + '/' )

        # Use wget to download
        print( 'Loading data...' )
        os.system( "wget -O " + geoscld_savepath  + " " + geoscld_htmlpath )
    
    return geoscld_savepath
    

# Find closest 3 hours to current datetime object
def round_dt(dt_obj, n):
    # Round to the nearest n hours (must be divisible into 24)
    
    # Get current time
    cur_hour = dt_obj.hour
    cur_min = dt_obj.minute
    cur_sec = dt_obj.second
    
    # Convert time to hours
    curtime_hours = cur_hour + ( cur_min + cur_sec/60 )/60
    roundtime_hours = n * round(curtime_hours/n)
    
    # Return as new datetime object
    rounddt_obj = dt_obj.replace( hour=0, minute=0, second=0) + dt.timedelta(hours=roundtime_hours)
    
    return rounddt_obj

# Identify sonde locations in reanalysis data
def sonde_indices( df_sonde, geos_type, folderpath, year, month, day, compare_type=0 ):
    # compare_type: 0 - lat/lon/alt/time compare; 1 - alt/time compare, lat/lon based on start
    
    # Identify grid information
    ntime, nlat, nlon, nlev, time0, lat0, lon0, lev0, dtime, dlat, dlon, dlev = grid_setup( geos_type, folderpath )
    gfp_folderpath = folderpath + year + '/' + month + '/'
    if geos_type == 'GEOS_FP':
        gfp_filename = 'GEOSFP.' + year + month + day + '.I3.025x03125.nc'
        gfp_filenamedyn = 'GEOSFP.' + year + month + day + '.A3dyn.025x03125.nc'
    else:
        gfp_filename = 'MERRA2.' + year + month + day + '.I3.05x0625.nc4'
        gfp_filenamedyn = 'MERRA2.' + year + month + day + '.A3dyn.05x0625.nc4'
    
    # Pointers to open netcdf files
    nc_geosfp = nc.Dataset( gfp_folderpath + gfp_filename )
    nc_geosfpdyn = nc.Dataset(gfp_folderpath + gfp_filenamedyn)
    
    # Convert lat/lon to geos-fp grid indices
    if compare_type == 0:
        df_sonde['lat_index'] = np.ceil( (df_sonde['latitude']-lat0)/dlat )
        df_sonde['lon_index'] = np.ceil( (df_sonde['longitude']-lon0)/dlon )
    elif compare_type == 1:
        lat_first = df_sonde.latitude.loc[~df_sonde.latitude.isnull()].iloc[0]
        lon_first = df_sonde.longitude.loc[~df_sonde.longitude.isnull()].iloc[0]
        df_sonde['lat_index'] = np.ceil( (lat_first-lat0)/dlat )
        df_sonde['lon_index'] = np.ceil( (lon_first-lon0)/dlon )
    else:
        error('compare_type not found!')
    
    # Calculate time to geos-fp grid index
    now = df_sonde['Time GMT'][0] # dt.datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = np.zeros((df_sonde.shape[0]))
    for ii,item in enumerate( df_sonde['Time GMT'] ):
        minutes[ii] = (item-midnight).seconds/60
    df_sonde['time_index'] = np.ceil( (minutes-time0)/dtime )
    df_sonde.loc[df_sonde.time_index==8, 'time_index'] = 7
    
    ## Calculate pressure at the lat/lon/time locations
    
    # Identify if pressure in hPa or Pa
    if geos_type == 'MERRA2':
        PS_scale = 1/100
    else:
        PS_scale = 1
    
    # Zero out some data
    Pcent = np.zeros((len(minutes)))
    press_index = np.zeros((len(minutes)))
    
    # Just extract the column (don't account for lat/lon location)
    if compare_type == 1:
        PS = nc_geosfp.variables['PS'][ df_sonde['time_index'][0], df_sonde['lat_index'][0].astype(int), df_sonde['lon_index'][0].astype(int) ] * PS_scale
        Pedge_cur, Pcent_cur = geosfp_press( PS )
    
    # Loop over each sonde value
    for ii in range( len(minutes) ):
        
        if compare_type == 0: # If compare by all indices, need to keep loading vertical pressure distribution
            
            if not(np.isnan(df_sonde['lat_index'][ii]) or np.isnan(df_sonde['lon_index'][ii])):
                
                # Get the surface pressure for this time, lat and lon and use to calculate pressure column
                PS = nc_geosfp.variables['PS'][ df_sonde['time_index'][ii], df_sonde['lat_index'][ii].astype(int), df_sonde['lon_index'][ii].astype(int) ] * PS_scale
                Pedge_cur, Pcent_cur = geosfp_press( PS )
                
                # Find closest pressure to sonde pressure
                press_index_cur = np.searchsorted(np.flip(Pcent_cur), df_sonde['P'][ii])
                if press_index_cur > 71:
                    press_index_cur = 71
                press_index_cur = 71 - press_index_cur
                press_index[ii] = press_index_cur
                Pcent[ii] = Pcent_cur[press_index_cur]
                
            else:
                
                Pcent[ii] = np.nan
                press_index[ii] = np.nan
                
        else: # If compare by only start location, just need to match the starting vertical pressure distribution
            
            # Find closest pressure to sonde pressure
            press_index_cur = np.searchsorted(np.flip(Pcent_cur), df_sonde['P'][ii])
            if press_index_cur > 71:
                press_index_cur = 71
            press_index_cur = 71 - press_index_cur
            press_index[ii] = press_index_cur
            Pcent[ii] = Pcent_cur[press_index_cur]
    
    # Convert pressure index data format to integer
    df_sonde['press_index'] = press_index.astype('int')
    
    # Extract temperature and specific humidity - use to calculate RHi
    QV, TEMP, RH = geosfp_extract( df_sonde['lat_index'], df_sonde['lon_index'],
                           df_sonde['press_index'], df_sonde['time_index'], nc_geosfp, nc_geosfpdyn )
    RHi = QT2RHi( QV, TEMP, Pcent )
    RH = RHi2RHw( RHi, TEMP )
    
    # Estimate altitude of each geos pressure
    g = 9.81
    R = 8.31
    M = 0.02896
    ALT = R*TEMP/(M*g) * np.log( Pcent[0]/Pcent )/1000 # Altitude in km
    
    # Also just get the column centers
    if compare_type == 1:
        ones_lev = np.ones( 72 )
        QV_col, TEMP_col, RH_col = geosfp_extract( ones_lev*df_sonde['lat_index'][0], ones_lev*df_sonde['lon_index'][0],\
                                                 np.linspace(0,71,72), ones_lev*df_sonde['time_index'][0], \
                                                 nc_geosfp, nc_geosfpdyn )
        Pcent_col = Pcent_cur
        Pedge_col = Pedge_cur
        RHi_col = QT2RHi( QV_col, TEMP_col, Pcent_cur )
        RH_col = RHi2RHw( RHi_col, TEMP_col )
    else:
        QV_col = 0
        TEMP_col = 0
        RH_col = 0
        Pcent_col = 0
        Pedge_col = 0
        RHi_col = 0
    
    return df_sonde, QV, TEMP, RH, RHi, Pcent, ALT, QV_col, TEMP_col, RH_col, Pcent_col, RHi_col, Pedge_col

# Extract temperature and specific humidity at these lat/lon/lev/time indices
def geosfp_extract( I, J, L, T, nc_geosfp, nc_geosfpdyn ):
    QV = np.zeros((len(I)))
    Temp  = np.zeros((len(I)))
    RH  = np.zeros((len(I)))
    # T[T>7] = 7
    for ii in range(len(I)):
        if not(np.isnan(I[ii]) or np.isnan(J[ii])):
            QV[ii] = nc_geosfp.variables['QV'][ T[ii], L[ii], I[ii].astype(int), J[ii].astype(int) ]
            Temp[ii]  = nc_geosfp.variables['T'][ T[ii], L[ii], I[ii].astype(int), J[ii].astype(int) ]
            RH[ii]  = nc_geosfpdyn.variables['RH'][ T[ii], L[ii], I[ii].astype(int), J[ii].astype(int) ]
        else:
            QV[ii] = np.nan
            Temp[ii] = np.nan
            RH[ii] = np.nan
    return QV, Temp, RH

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Calculate RMSE between profiles at particular heights
def RMSE_alt( df_sonde, RHi, P_cent, altLo, altHi ):
    # Identify altitude range and extract values
    alt_idx = df_sonde["altitude"].between(altLo, altHi, inclusive = True)
    alt_ = df_sonde["altitude"].values[alt_idx]
    P_sonde = df_sonde["P"].values[alt_idx]
    P_remet = P_cent[alt_idx]
    RHi_sonde = df_sonde["RHi"].values[alt_idx]
    RHi_remet = RHi[alt_idx]
    # Check for nans and replace by interpolation
    remet_nans = np.isnan(RHi_remet)
    if np.sum(remet_nans)>0:
        RHi_remet[remet_nans]= np.interp(alt_[remet_nans], alt_[~remet_nans], RHi_remet[~remet_nans])
    # Calculate the RMSE
    RMSerr = rmse( RHi_sonde, RHi_remet )
    return RHi_sonde, RHi_remet, alt_, P_sonde, P_remet, RMSerr

# "False positives" analysis
def binary_test( df_sonde, RH, Pcent, T, Pedge, edge=1 ):
    # GOAL: At each reanalysis pressure, would I get any contrails?
    # Compare reanalysis with sonde!
    # edge == 1 --> use edges and get average RHi_sonde in layer; edge == 0 --> just use center point to match
    
    # Flip pressure if ascending
    if Pcent[0] < Pcent[-1]:
        Pcent = np.flipud( Pcent )
        Pedge = np.flipud( Pedge )
        T = np.flipud( T )
        RH = np.flipud( RH )
        flipped = 1
    else:
        flipped = 0
    
    # Extract sonde data
    RH_sonde = df_sonde["RH"].values
    P_sonde = df_sonde["P"].values
    T_sonde = df_sonde["T"].values
    
    # Zero out some storage space
    SAC_reanalysis = np.zeros(len(Pcent))
    SAC_sonde = np.zeros(len(Pcent))
    
    # Loop over each reanalysis layer
    for layer, P_cur in enumerate( Pcent ):
        
        # Get current layer's RHi, T and pressure edges
        RH_cur = RH[layer]
        T_cur = T[layer]
        Pedge_lowcur = Pedge[layer]
        Pedge_uppcur = Pedge[layer+1]
        
        # Find cases between upper and lower pressure edges and extract conditions
        if edge == 1:
            match_idx = np.where( (P_sonde<Pedge_lowcur)*(P_sonde>Pedge_uppcur) )
        else:
            match_idx = find_nearest( P_sonde, P_cur )
        P_sonde_cur = P_sonde[match_idx]
        RH_sonde_cur = RH_sonde[match_idx]
        T_sonde_cur = T_sonde[match_idx]
        
        # Identify if contrail forms
        SAC_reanalysis_cur = SAC_general( RH_cur, T_cur, P_cur, printout=1 )
        SAC_sonde_cur_temp = SAC_general( RH_sonde_cur, T_sonde_cur, P_sonde_cur )
        # SAC_sonde_cur_alt = SAC_general( np.mean(RH_sonde_cur), np.mean(T_sonde_cur), np.mean(P_sonde_cur) )
        
        # If taking every point, need to calculate averages
        if edge == 1:
            SAC_sonde_ave = np.mean( SAC_sonde_cur_temp )
            SAC_sonde_cur = 1*(SAC_sonde_ave>0.2)
        
#         if SAC_reanalysis_cur == 1:
#             print( 'Reanalysis case satisfies SAC' )
#             print(SAC_sonde_ave, SAC_reanalysis_cur)
#         if np.any(SAC_sonde_cur_temp):
#             print( 'Some sonde cases satisfy SAC')
#             print(SAC_sonde_ave, SAC_sonde_cur, SAC_reanalysis_cur)
        
        # if len(RHi_sonde_cur) > 0:
            # print( RHi_cur, np.max(RHi_sonde_cur) )
            # print(T_cur, np.mean(T_sonde_cur), np.ptp(T_sonde_cur), np.max(T_sonde_cur))
            # print(P_cur, Pedge_lowcur, Pedge_uppcur, np.mean(P_sonde_cur), np.ptp(P_sonde_cur), np.max(P_sonde_cur))
            # print(np.mean(RHi_sonde_cur), np.ptp(RHi_sonde_cur), np.max(RHi_sonde_cur), SAC_sonde_ave, RHi_cur, SAC_reanalysis_cur)
        
        # Store this data
        SAC_reanalysis[layer] = SAC_reanalysis_cur
        SAC_sonde[layer] = SAC_sonde_cur
        
    # If pressure was ascending, reverse the flipped data
    if flipped:
        SAC_reanalysis = np.flipud( SAC_reanalysis )
        SAC_sonde = np.flipud( SAC_sonde )
    
    return SAC_reanalysis, SAC_sonde
    

## Useful plotting functions

# Function to plot ascent and descent separately
def plot_ascdes( xvar, yvar, alt, line_style, line_color1, line_color2, plot_label, ax ):
    max_index = np.nanargmax(alt,axis=0)
    ax.plot( xvar[1:max_index], yvar[1:max_index], line_style, color=line_color1, label=plot_label+' ascent' )
    ax.plot( xvar[max_index:] , yvar[max_index:] , line_style, color=line_color2, label=plot_label+' decent' )
    return max_index

# Function to plot generally
def plot_gen( xvar, yvar, alt, line_style, line_color1, line_color2, plot_label, ax ):
    ax.plot( xvar, yvar, line_style, color=line_color1, label=plot_label )
    
    
## Other useful functions

# Nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx






