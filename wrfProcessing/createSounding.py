import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
SCRATCH='/global/cscratch1/sd/qnicolas/'


### MODIFY HERE ###
outfile = "WRF/WRFV4_channel/test/em_beta_plane/input_sounding" #coarse
wind=-10.
SST = 300
###################

z = np.arange(0.,30001,50.)

def moist_adiabat(z,SST):
    p = 1000*np.exp(-9.81*z/(287.*270.)) * units.hPa
    Tp = mpcalc.moist_lapse(p,(SST-1)*units.K)
    thetap = mpcalc.potential_temperature(p,Tp)
    thicknesses = [mpcalc.thickness_hydrostatic(p, Tp,bottom=p[i] ,depth=p[i]-p[i+1])/units.m for i in range(len(p)-1)]
    zp = np.concatenate([[0.],np.cumsum(thicknesses)])
    qp = 0.8*mpcalc.saturation_mixing_ratio(p,Tp)
        
    thetaz = np.interp(z,zp,(thetap/units.K))
    qz = np.interp(z,zp,qp)
    return [float(x) for x in thetaz],[float(x) for x in qz]

thetaz,qz = moist_adiabat(z,SST)

i=0
f = open(SCRATCH+outfile, "w")
print('{:>10.2f}{:>10.2f}{:>10.2f}'.format(1000.,thetaz[0],1000*qz[0]),file=f)
for i,zz in enumerate(z):
    print('{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}'.format(zz,thetaz[i],1000*qz[i],wind,0.),file=f)
f.close()