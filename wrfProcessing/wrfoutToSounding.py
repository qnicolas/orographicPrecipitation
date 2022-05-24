import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
SCRATCH='/global/cscratch1/sd/qnicolas/'
import wrf
import xarray as xr


### MODIFY HERE ###
wrfinput = xr.open_dataset(SCRATCH+"wrfdata/saved/channel.wrf.100x2.mountain.60lev.3km/wrfinput_d01")
wrfout = xr.open_dataset(SCRATCH+"wrfdata/saved/channel.wrf.100x2.mountain.60lev.3km/wrfout_d01_1970-08-29_06_00_00")
outfile = "WRF/WRFV4_channelbis/test/em_beta_plane/input_sounding"
wind= -10.
###################

z_stag = (wrfinput.PHB[0,:,0,0]+wrfout.PH[-40:,:,:,2000:].mean(['Time','south_north','west_east']))/9.81
z_destag  = np.array(wrf.destagger(z_stag,0))
theta = np.array(300+wrfout.T[-40:,:,:,2000:].mean(['Time','south_north','west_east']))
q = np.array(wrfout.QVAPOR[-40:,:,:,2000:].mean(['Time','south_north','west_east']))

z = np.arange(0.,35000,50.)
thetaz = np.interp(z,z_destag,theta)
qz = np.interp(z,z_destag,q)

i=0
f = open(SCRATCH+outfile, "w")
print('{:>10.2f}{:>10.2f}{:>12.7f}'.format(1000.,thetaz[0],1000*qz[0]),file=f)
for i,zz in enumerate(z):
    print('{:>10.2f}{:>10.2f}{:>12.7f}{:>10.2f}{:>10.2f}'.format(zz,thetaz[i],1000*qz[i],wind,0.),file=f)
f.close()