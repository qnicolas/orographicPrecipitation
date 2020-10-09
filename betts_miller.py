import numpy as np
import xarray as xr

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

# ==================================================================
# IMPLEMENT A SIMPLIFIED BETTS-MILLER SCHEME AS DESCRIBED IN FRIERSON 2007
# ==================================================================
tau_sbm = 2*3600 #2h relaxation time
rh_sbm = 0.7

def qprofile(ps,temp_2m,plevs):
    """Compute the vertical profile of specific humidity associated with a moist adiabatic temperature
    profile, constant RH defined by rh_sbm, and reference level.temperature given by ps and temp_2m
    
    Args:
        ps : surface pressure; 3D xarray with dimensions latitude, longitude, time
        temp_2m : 2m temperature; 3D xarray with dimensions latitude, longitude, time
        plevs : array giving pressure levels to be considered, in increasing order
            
    Returns:
        qprofile : reference specific humidity; 4D xarray with dimensions latitude, longitude, level, time
    """
    pressure = plevs * units.hPa
    temperature = temp_2m*units.degK
    surfpressure = ps*units.hPa

    moistprofile = mpcalc.moist_lapse(pressure[::-1], temperature,surfpressure)[::-1]
    estar = mpcalc.saturation_vapor_pressure(moistprofile)
    qprofile = rh_sbm*mpcalc.specific_humidity_from_dewpoint(mpcalc.dewpoint(estar),pressure)
    
    return xr.DataArray(np.array(qprofile),[('level', plevs)])

def integrate(f, x):
    #INTEGRATE  Computes one-dimensional integral.
    #    INTEGRATE(f, x) computes an approximation to the integral
    #    \int f(x) dx over the range of the input vector x. Both x
    #    and f must be vectors. 

    dx1 = np.gradient(x)
    dx1[0] = 0.5 * dx1[0]
    dx1[-1] = 0.5 * dx1[-1]
    F = np.sum(f * dx1)

    return F

def sbm(ps,temp_2m,q):
    """Compute the convective precipitation thanks to a simplified betts-miller scheme as 
    described in Frierson, 2007. Relaxation to a constant RH, moist pseudoadiabatic profile.
    Args:
        ps : surface pressure; 3D xarray with dimensions latitude, longitude, time
        temp_2m : 2m temperature; 3D xarray with dimensions latitude, longitude, time
        q : specific humidity; 4D xarray with dimensions latitude, longitude, level, time 
            
    Returns:
        pr : convective precipitation; 3D xarray with dimensions latitude, longitude, time
    """
    qprofile2 = lambda p,t2m : qprofile(p,t2m,np.array(q.level))
    integrate2 = lambda f : integrate(f,np.array(q.level))
    
    qref = xr.apply_ufunc(qprofile2,ps/100,temp_2m,output_core_dims=[['level']],vectorize=True)
    
    pr = np.maximum(xr.apply_ufunc(integrate2,(q-qref)/tau_sbm,input_core_dims=[['level']],vectorize=True),0.)
    
    return pr