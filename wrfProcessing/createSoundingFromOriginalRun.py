import numpy as np
import xarray as xr
import wrf
SCRATCH='/global/cscratch1/sd/qnicolas/'


### MODIFY HERE ###
outfile = "WRF/WRFV4_gw/test/em_beta_plane/input_sounding" #coarse
wind= -10.
###################

wrfinput = xr.open_dataset("/global/cscratch1/sd/qnicolas/wrfdata/saved/channel.wrf.100x2.mountain.60lev.dry.3km/wrfinput_d01")
z_ref = wrf.destagger((wrfinput.PHB+wrfinput.PH).isel(Time=0,west_east=0,south_north=0)/9.81,0)
theta_ref = 300+wrfinput.T.isel(Time=0,west_east=0,south_north=0)
q_ref = wrfinput.QVAPOR.isel(Time=0,west_east=0,south_north=0)

z_stratosphere     = np.linspace(z_ref[-1],35000,30)
theta_stratosphere = float(theta_ref[-1])+27*(z_stratosphere-float(z_ref[-1]))/1000
q_stratosphere     = np.array(q_ref[-1])*np.exp(-(z_stratosphere-float(z_ref[-1]))/1.5e3)

z_ref_ext = np.concatenate([np.array(z_ref),z_stratosphere])
q_ref_ext = np.concatenate([np.array(q_ref),q_stratosphere])
theta_ref_ext = np.concatenate([np.array(theta_ref),theta_stratosphere])

z = np.arange(0.,35000,50.)
thetaz = np.interp(z,z_ref_ext,theta_ref_ext)
qz = np.interp(z,z_ref_ext,q_ref_ext)

i=0
f = open(SCRATCH+outfile, "w")
print('{:>10.2f}{:>10.2f}{:>10.2f}'.format(1000.,thetaz[0],1000*qz[0]),file=f)
for i,zz in enumerate(z):
    print('{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}'.format(zz,thetaz[i],1000*qz[i],wind,0.),file=f)
f.close()