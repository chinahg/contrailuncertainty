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

# Make array of .nc filenames (grib)
file_names = []
years = np.linspace(2016,2020,num=5,dtype=int)
# Need to loop as glob is not capturing all file names
for i in range(len(years)):
    file_names = file_names + glob.glob('/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'+str(years[i])+'/*.grib', recursive=True)
num_files = len(file_names)
file_names.sort()

save_dir = '/home/chinahg/GCresearch/contrailuncertainty/ERA5_processing/ERA5_downloads/ERA5_downloads/'

for j in range(num_files):
    path_grib = file_names[j]
    filename = os.path.basename(path_grib)
    year = filename.split('_')[0]
    save_path = save_dir + year +"/"+ filename.replace(".grib", ".nc")
    if os.path.exists(save_path):
        if os.path.exists(path_grib):
            #Delete old grib file
            os.remove(path_grib)
        print(f'Found {save_path}, skipping conversion to NetCDF')
        
    else: 
        ds = xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath":""})
        print(save_path)
        ds.to_netcdf(save_path)
        ds.close()
        #Delete old grib file
        os.remove(path_grib)
        print(f"File '{path_grib}' deleted successfully.")