"""
Convert .grib to .nc files
Parameters
----------
path_grib : str
    Path to grib file
save_dir : str
    Path to save directory
overwrite : bool, optional
    Flag to overwriting existing files, by default False
Returns
-------
None
"""

import os
import xarray as xr
import numpy as np
import glob
import csv

have_csv = True

if have_csv == True:
    csv_path = '/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/files2convert.csv'
    csv_files = []

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            csv_files.extend(row)
    num_files = len(csv_files)

    print("Number of files in CSV: ", len(csv_files))
    
    for k in range(num_files):
        print("Loading GRIB file: ", csv_files[k])

        for root, dirs, files in os.walk('/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'):
            for file in files:
                if file == csv_files[k]: 
                    path_grib = os.path.join(root, file)
                    break

        if path_grib != '':
            ds = xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath":""})
        else:
            print("File not found: ", csv_files[k])
            continue
        
        nc_path = path_grib.replace(".grib", ".nc") 
        ds.to_netcdf(nc_path, format="NETCDF4")
        print("Saved NetCDF file: ", os.path.basename(nc_path))

        ds.close()
        print("Closed GRIB file: ", path_grib)

        #Delete old grib file
        os.remove(path_grib)
        print(f"File '{path_grib}' deleted successfully.")

else:
    # Make array of .nc filenames (grib)
    file_names = []
    years = np.linspace(2020,2021,num=2,dtype=int)

    # Need to loop as glob is not capturing all file names
    for i in range(len(years)):
        file_names = file_names + glob.glob('/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'+str(years[i])+'/*.grib', recursive=True)

    num_files = len(file_names)
    file_names.sort()

    print("Number of files: ", num_files)

    for j in range(num_files):
        path_grib = file_names[j] # Original file path
        filename = os.path.basename(path_grib) # GRIB filename
        year = filename.split('_')[0] # Extract year from filename
        nc_path = path_grib.replace(".grib", ".nc") # Make path to save NetCDF file
        print("file number: ", j)
        # print("nc file path: ", nc_path)
        # print("Current file: ", filename)

        if os.path.exists(nc_path): # If the .nc file already exists, skip conversion and delete the .grib file
            if os.path.exists(path_grib):
                #Delete old grib file
                os.remove(path_grib)
            print(f'Found {nc_path}, skipping conversion to NetCDF')

        else: 
            print("Loading GRIB file: ", path_grib)
            try: 
                ds = xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath":""})
                ds.to_netcdf(nc_path, format="NETCDF4")
                print("Saved NetCDF file: ", os.path.basename(nc_path))
                ds.close()
                print("Closed GRIB file: ", path_grib)

                #Delete old grib file
                os.remove(path_grib)
                print(f"File '{path_grib}' deleted successfully.")
            except:
                print("Error saving NetCDF file: ", os.path.basename(nc_path))

                if os.path.exists(path_grib) == False:
                    print("File does not exist: ", path_grib)
                    ds.close()

                else:
                    ds.close()
                    print("Closed GRIB file: ", path_grib)

                    #Delete old grib file
                    os.remove(path_grib)
                    print(f"File '{path_grib}' deleted successfully.")