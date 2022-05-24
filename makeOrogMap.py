import numpy as np
import xarray as xr

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

orogm=xr.open_dataset("/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc").Z.isel(time=0)/9.81

width=25
fig,ax = plt.subplots(1,1,figsize=(width,width/2.5),subplot_kw={'projection' : ccrs.PlateCarree()})
orogm.plot(ax=ax,levels = np.linspace(-1000,6000,101),transform = ccrs.PlateCarree(),cmap=plt.cm.terrain,cbar_kwargs = {'ticks':range(0,6001,1000)})
ax.coastlines()


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='w', alpha=.5, linestyle='-')
gl.top_labels = False
gl.bottom_labels = False

gl.xlocator = mticker.FixedLocator(range(-180,180,10))
ax.set_xticks(range(-180,190,10))
ax.set_xticklabels(["%i°E"%lon for lon in range(-180,190,10)],fontsize=9);None
ax2 = ax.secondary_xaxis('top')
ax2.set_xticks(range(-180,190,10))
ax2.set_xticklabels(["%i°E"%lon for lon in list(range(180,360,10))+list(range(0,190,10))],fontsize=9);None

gl.ylocator = mticker.FixedLocator(range(-80,90,10))
gl.ylabel_style = {'size': 10, 'color': 'k'}

ax.set_title("World topography, ERA5")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

fig.savefig("world_topo_grid.png",dpi=300)
