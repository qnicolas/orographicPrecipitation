import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
SCRATCH='/global/cscratch1/sd/qnicolas/'


### MODIFY HERE ###
outfile = "WRF/WRFV4_channel296/test/em_beta_plane/input_sounding" #coarse
wind= -10.
SST = 296.
###################

z = np.arange(0.,50001,50.)

def moist_adiabat(z,SST):
    p = 1000*np.exp(-9.81*z/(287.*270.)) * units.hPa
    Tp = mpcalc.moist_lapse(p,(SST-1)*units.K)
    qp = 0.8*mpcalc.saturation_mixing_ratio(p,Tp)
    
    ztrop1=17e3
    ztrop2=19e3
    idx1 = np.argmin((z-ztrop1)**2)
    idx2 = np.argmin((z-ztrop2)**2)
    Tp[idx1:idx2] = Tp[idx1]
    Tp[idx2:] = Tp[idx1]+2e-3*(z[idx2:]-ztrop2)*units.K
    thetap = mpcalc.potential_temperature(p,Tp)
    thicknesses = [mpcalc.thickness_hydrostatic(p, Tp,bottom=p[i] ,depth=p[i]-p[i+1])/units.m for i in range(len(p)-1)]
    zp = np.concatenate([[0.],np.cumsum(thicknesses)])
    
    thetaz = np.interp(z,zp,(thetap/units.K))
    qz = np.interp(z,zp,qp)
    idxs = z<35000
    return z[idxs], np.array([float(x) for x in thetaz])[idxs],np.array([float(x) for x in qz])[idxs]

z,thetaz,qz = moist_adiabat(z,SST)

i=0
f = open(SCRATCH+outfile, "w")
print('{:>10.2f}{:>10.2f}{:>10.2f}'.format(1000.,thetaz[0],1000*qz[0]),file=f)
for i,zz in enumerate(z):
    print('{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}'.format(zz,thetaz[i],1000*qz[i],wind,0.),file=f)
f.close()