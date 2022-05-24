import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.ticker import NullFormatter
matplotlib.rcParams.update({'font.size': 16})
import os
import sys
sys.path.append(os.path.abspath('../..'))
from orographicPrecipitation.wrfProcessing.wrf_hr_utils import *

movietype = 'tprime'#'isentropes'#

# Get data

ghats_ideal = xr.open_dataset("/global/cscratch1/sd/qnicolas/wrfdata/saved/gw.wrf.2D.60lev.500m.3km/wrfout_d01_1970-01-01_00_00_00")

x_kwargs={'center':314,'flip_x':False,'dx':10}
ghats_topo = change_coords_sfc(ghats_ideal.HGT[0,0],**x_kwargs)

# Define nb frames
print("Total nb of times: ",len(ghats_ideal.Time))
nframes=len(ghats_ideal.Time)
time_disc=24 # frames per day
fig,ax=plt.subplots(1,1,figsize=(15,5))

# Animate
if movietype == 'isentropes':
    ghats_ideal_theta_z = xr.open_dataarray("/global/cscratch1/sd/qnicolas/wrfdata/saved/gw.wrf.2D.60lev.500m.3km/diags/wrf.THETA.zinterp.days0-10.nc")[:,2:]
    levels=np.array(ghats_ideal_theta_z.isel(Time=0,distance_from_mtn=0).sel(z=np.arange(1000.,20000.,1000.)))
    
    
    def update(i):
        print(i)
        ax.cla()
        ghats_topo.plot(ax=ax,color='k',linewidth=2.)
        ghats_ideal_theta_z.isel(Time=i).plot.contour(ax=ax,levels=levels)
        ax.set_ylim(0.,20000.)
        ax.set_xlabel("Distance from mountain peak (km)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("Isentropes, Time = %i days %02i h"%(i//time_disc,((24*i)//time_disc)%24))

elif movietype == 'tprime':
    ghats_ideal_temp_z= xr.open_dataarray("/global/cscratch1/sd/qnicolas/wrfdata/saved/gw.wrf.2D.60lev.500m.3km/diags/wrf.TEMP.zinterp.days0-10.nc")
    ghats_ideal_temp_z = ghats_ideal_temp_z - ghats_ideal_temp_z.isel(distance_from_mtn=0)
    levels=np.linspace(-5,5,41)
    def update(i):
        print(i)
        ax.cla()
        ghats_topo.plot(ax=ax,color='k',linewidth=2.)
        ghats_ideal_temp_z.isel(Time=i).plot.contourf(ax=ax,levels=levels,add_colorbar=False)
        ax.set_ylim(0.,20000.)
        ax.set_xlabel("Distance from mountain peak (km)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("T', Time = %i days %02i h - colorscale from -5 to +5 K"%(i//time_disc,((24*i)//time_disc)%24))
    
im_ani = animation.FuncAnimation(fig, update, frames=nframes,interval=100)
#im_ani.save("isentropes_10days.mp4")
im_ani.save("tprime_10days.mp4")

