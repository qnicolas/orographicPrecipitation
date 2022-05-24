#!/global/homes/q/qnicolas/.conda/envs/era5/bin/python

#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=qnicolas@berkeley.edu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=00:10:00
#SBATCH --constraint=haswell
#SBATCH --qos=debug
#SBATCH --output=R-%x.out

import os
import glob
import time

import pandas as pd
import numpy as np
import xarray as xr
import datetime

def extract_wrfpro(simulation_path,nfiles=False):
    files=sorted(glob.glob(simulation_path+'wrfpout_d01_197*'))
    if nfiles:
        files=files[:nfiles]
    return xr.open_mfdataset(files,combine='nested',concat_dim='Time',parallel=True)

deepchannel_p = extract_wrfpro('/global/cscratch1/sd/qnicolas/wrfdata/saved/channel.wrf.100x2.mountain.60lev.3km/')

pressure=deepchannel_p.P_PL[-1].load()/100
plevs=pressure.assign_attrs({'units':'hPa'})

deepchannel_p.Q_PL[2900:,:,:,-1000:].assign_coords(lev=plevs).swap_dims({'num_press_levels_stag':'lev'}).rename(south_north='lat',west_east='lon',Time='time').to_netcdf("/global/cscratch1/sd/qnicolas/temp/deepchannel.Q_PL.days100-250.hourly_ups.nc")

