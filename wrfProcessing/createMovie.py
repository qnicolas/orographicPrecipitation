import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.ticker import NullFormatter
matplotlib.rcParams.update({'font.size': 16})
import glob

def extract_wrfpro(simulation_path,nfiles=30): #default extracts 300 days
    files=sorted(glob.glob(simulation_path+'wrfpout_d01_197*'))
    files = files[:min(len(files),nfiles)]
    return xr.open_mfdataset(files,combine='nested',concat_dim='Time',parallel=True)
hrchannel_p = extract_wrfpro('/global/cscratch1/sd/qnicolas/wrfdata/saved/channel.wrf.100x2.mountain.3km/')

def change_coords_pl(sim,w,center=1635):
    return w.assign_coords({'distance_from_mtn':(center-w.west_east)*3,'pressure':sim.P_PL[0]/100}).swap_dims({'num_press_levels_stag':'pressure','west_east':'distance_from_mtn'}).persist()

def change_coords_sfc(w,center=1635):
    return w.assign_coords({'distance_from_mtn':(center-w.west_east)*3}).swap_dims({'west_east':'distance_from_mtn'}).persist()

def change_coords_sn(var):
    return var.assign_coords({'distance_ns':3*var.south_north}).swap_dims({'south_north':'distance_ns'})

ndays=5
wmovie = change_coords_sn(change_coords_pl(hrchannel_p,hrchannel_p.W_PL.isel(Time=slice(3600,3600+ndays*24)).squeeze())).sel(distance_from_mtn=slice(500,-500))
wmovie=wmovie.where(wmovie>-100).fillna(0.)

wmovie_xy = wmovie.sel(pressure=500.).persist()
wmovie_xp = wmovie.mean('distance_ns')[:,1:-4].persist()

# Animate
fig,(ax,ax2) = plt.subplots(2,1,figsize=(20,9))

wscale=1
wscale2=0.5

def update(i):
    print(i)
    ax.axvline(   0.,color='k',linewidth=0.5)
    ax.axvline(-100.,color='k',linewidth=0.5,linestyle='--')
    ax.axvline( 100.,color='k',linewidth=0.5,linestyle='--')
    wmovie_xy.isel(Time=i).plot.contourf(ax=ax,levels=np.linspace(-wscale,wscale,41),add_colorbar=False)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance from mountain peak (km)")
    ax.set_ylabel("Along-ridge distance (km)")
    ax.set_title("W at 500 hPa, Time = %i days %02i h (color scale = [-1m/s,1m/s])"%(150+i//24,i%24))
    
    wmovie_xp.isel(Time=i).plot.contourf(ax=ax2,yincrease=False,levels=np.linspace(-wscale2,wscale2,41),add_colorbar=False)
    ax2.set_xlabel("Distance from mountain peak (km)")
    ax2.set_ylabel("pressure (hPa)")
    
im_ani = animation.FuncAnimation(fig, update, frames=ndays*24)
im_ani.save("test.mp4")