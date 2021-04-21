import numpy as np
import pandas as pd
import xarray as xr
import glob

import os
import sys
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from orographicPrecipitation.precip_extremes_scaling import saturation_thermodynamics, moist_adiabatic_lapse_rate, pars
from orographicPrecipitation.precip_model_functions import retrieve_era5_pl,retrieve_era5_sfc
from tools.e5tools import e5_monthly_file


EPS = np.finfo(float).eps
orog1 = xr.open_dataset("/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc")
orog=orog1.Z/9.80665
m_per_degreelat = 6370*1e3*np.pi/180
orog_precise=xr.open_dataset("/global/cscratch1/sd/qnicolas/precipmodel/GMTED2010_15n015_00625deg.nc").elevation
orog_precise=orog_precise.assign_coords({'latitude':np.arange(-90.+0.0625/2,90.,0.0625)*orog_precise.nlat**0,'longitude':np.arange(-180.,180.,0.0625)*orog_precise.nlon**0}).swap_dims({'nlat':'latitude','nlon':'longitude'})
orog_precise.coords['longitude'] = orog_precise.coords['longitude'] % 360 
orog_precise = orog_precise.reindex(latitude=list(reversed(orog_precise.latitude))).sortby(orog_precise.longitude)
timedisc = 6 #take data every 6 hours



def retrieve_era5_monthly(month,lonlat,varid):
    path = "/global/project/projectdirs/m3310/wboos/era5monthlyQuentin/"
    era5var = xr.open_dataset(e5_monthly_file(varid,month[:4]))
    varname = list(era5var.data_vars)[0] #get name of the main variable, eg 'W' for omega
    era5var1 = era5var[varname].sel(time=pd.to_datetime(month,format='%Y%m')).sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]))
    return era5var1

def full_linear_model_saturated_monthly(month,lonlat,fine_scale=False,lr_param=0.99,p0=0.):
    """compute total precip in a box given by lonlat, for a given month
        with Smith's linear model
        
        month : month for which compute the precip; format "YYYYMM"       
        lonlat : list, [lon1, lon2, lat1, lat2] specifying the box on which to perform calculation. 
            NOTE THAT 0 <= lon1 < lon2 <= 360 and -90 <= lat1 < lat2 <= 90        
    """
    
    lonlat2 = np.array(lonlat)+np.array([-1,1,-1,1]) #Do this to get rid of the border effects (high values of precip where the box crosses topography)

    #Start by retriving surface temperature, pressure, and 900 hpa winds
    temp_surface = np.array([retrieve_era5_monthly(month,lonlat2,'128_167_2t').mean(["latitude","longitude"])])
    ps =           np.array([retrieve_era5_monthly(month,lonlat2,'128_134_sp').mean(["latitude","longitude"])])
    #u900 = np.array([retrieve_era5_monthly(month,lonlat2,'128_131_u').sel(level=900.).mean(["latitude","longitude"])])
    #v900 = np.array([retrieve_era5_monthly(month,lonlat2,'128_132_v').sel(level=900.).mean(["latitude","longitude"])])
    u900 = np.array([retrieve_era5_monthly(month,lonlat2,'*_10u').mean(["latitude","longitude"])])
    v900 = np.array([retrieve_era5_monthly(month,lonlat2,'*_10v').mean(["latitude","longitude"])])
        
    #compute elevation and dx, dy
    if fine_scale :
        elevation = orog_precise.sel(longitude=slice(lonlat2[0],lonlat2[1]),latitude=slice(lonlat2[3],lonlat2[2]))
    else :
        elevation = orog.sel(longitude=slice(lonlat2[0],lonlat2[1]),latitude=slice(lonlat2[3],lonlat2[2])).isel(time=0)
    coslat = np.cos(np.array(elevation.latitude[0])*np.pi/180.)
    dx = coslat*m_per_degreelat*np.abs(np.array(elevation.latitude[1]-elevation.latitude[0]))
    dy = m_per_degreelat*np.abs(np.array(elevation.longitude[1]-elevation.longitude[0]))

    pr = linear_model_saturated(temp_surface,ps,u900,v900,(lonlat[2]+lonlat[3])/2.,elevation,dx,dy,lr_param,p0)
    return pr.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]))


def full_linear_model_saturated(ds,lonlat,fine_scale=False):
    """compute total precip in a box given by lonlat, for days in ds
        with Smith's linear model
        
        ds : list of days for which to average and compute the precip; format "YYYYMMDD" (ie list of strings)        
        lonlat : list, [lon1, lon2, lat1, lat2] specifying the box on which to perform calculation. 
            NOTE THAT 0 <= lon1 < lon2 <= 360 and -90 <= lat1 < lat2 <= 90        
    """
    
    lonlat2 = np.array(lonlat)+np.array([-1,1,-1,1]) #Do this to get rid of the border effects (high values of precip where the box crosses topography)
    
    #Start by retriving surface temperature, pressure, and 900 hpa winds
    temp_surface = np.array(retrieve_era5_sfc(ds,lonlat2,'128_167_2t',tdisc=timedisc).mean(["latitude","longitude"]))
    ps =           np.array(retrieve_era5_sfc(ds,lonlat2,'128_134_sp',tdisc=timedisc).mean(["latitude","longitude"]))
    u900 = np.array(retrieve_era5_pl(ds,lonlat2, '128_131_u', firstlev=10,levdisc=1,tdisc=timedisc).sel(level=900.).mean(["latitude","longitude"]))
    v900 = np.array(retrieve_era5_pl(ds,lonlat2, '128_132_v', firstlev=10,levdisc=1,tdisc=timedisc).sel(level=900.).mean(["latitude","longitude"]))
    
    #compute elevation and dx, dy
    if fine_scale :
        elevation = orog_precise.sel(longitude=slice(lonlat2[0],lonlat2[1]),latitude=slice(lonlat2[3],lonlat2[2]))
    else :
        elevation = orog.sel(longitude=slice(lonlat2[0],lonlat2[1]),latitude=slice(lonlat2[3],lonlat2[2])).isel(time=0)
    coslat = np.cos(np.array(elevation.latitude[0])*np.pi/180.)
    dx = coslat*m_per_degreelat*np.abs(np.array(elevation.latitude[1]-elevation.latitude[0]))
    dy = m_per_degreelat*np.abs(np.array(elevation.longitude[1]-elevation.longitude[0]))
    
    pr = 0
    for i in range(len(temp_surface)):
        pr+= linear_model_saturated(temp_surface[i:i+1],ps[i:i+1],u900[i:i+1],v900[i:i+1],(lonlat[2]+lonlat[3])/2.,elevation,dx,dy)
    return pr.sel(longitude=slice(lonlat[0],lonlat[1]),latitude=slice(lonlat[3],lonlat[2]))/len(temp_surface)


def HwCw(temp_surface,ps,gamma):
    """Compute water vapor scale height and coefficient Cw assuming a moist adiabatic atmosphere
     - temp_surface = surface temperature (np.array)
     - ps = surface pressure (np.array)
     - gamma = environment lapse rate
    """
    gamma_m = moist_adiabatic_lapse_rate(temp_surface,ps,'era')
    es,_,_,L = saturation_thermodynamics(temp_surface,ps,'era')
    Hw = pars('gas_constant_v')*temp_surface**2/(L*gamma)
    Cw = es/(pars('gas_constant')*temp_surface)*gamma_m/gamma
    return Hw,Cw

def linear_model_saturated(temp_surface,ps,u900,v900,latitude,elevation,dx,dy,lr_param=0.99,p0=0.):
    """Compute orographic precipitation using the Linear Model, assuming a lapse rate = 0.99*moist adiabatic lapse rate
     - temp_surface = surface temperature [K] (np.array)
     - ps = surface pressure [Pa] (np.array)
     - u900 = 900 hPa zonal wind [m/s] (np.array)
     - v900 = 900 hPa meridional wind, [m/s] (np.array)
     - elevation : 2D xarray (latitude, longitude) of orography in [m]
     - dx : float, horizontal resolution in [m]
     - dy : float, vertical resolution in [m]
    """    
    gamma_m = moist_adiabatic_lapse_rate(temp_surface,ps,'era')
    gamma = lr_param*gamma_m
    Hw,Cw = HwCw(temp_surface,ps,gamma)
    param = {"latitude":latitude,
              "p0":p0,
              "windspeed":np.sqrt(u900[0]**2+v900[0]**2),
              "winddir":np.mod(270 - np.angle(u900[0]+v900[0]*1j,deg=True),360),
              "tau_c":2000.,
              "tau_f":2000.,
              "nm":np.sqrt(pars('gravity')/temp_surface*(gamma_m[0]-gamma)), #approximate formula, see Smith 2004
              "hw":Hw[0],
              "cw":Cw[0]
             }
    
    pra = compute_orographic_precip(np.array(elevation), dx, dy, **param)*24 #convert mm/hr into mm/day
    pr = xr.DataArray(pra,[('latitude', elevation.latitude),('longitude', elevation.longitude)])
    return pr


#From https://github.com/rlange2/orographic-precipitation/tree/master/orographic_precipitation
#----------------------------------------------------------------------------------------------
def compute_orographic_precip(elevation, dx, dy, **param):
    """Compute orographic precipitation.

    Parameters
    ----------
    elevation : array_like
        2D input array of a given elevation
    dx, dy : int
        Horizontal and vertical resolution in [m]
    **param
        A dictionary used to store relevant parameters for computation.

    param kwargs
    ----------------
    latitude (float) : Coriolis effect decreases as latitude decreases
    p0 (float) : uniform precipitation rate [mm hr-1], usually [0, 10]
    windspeed (float) : [m s-1]
    winddir (float) : wind direction [0: north, 270: west]
    tau_c (float) : conversion time delay [s]
    tau_f (float) : fallout time delay [s]
    nm (float) : moist stability frequency [s-1]
    hw (float) : water vapor scale height [m]
    cw (float) : uplift sensitivity [kg m-3], product of saturation water vapor sensitivity rhosref [kg m-3] and environmental lapse rate (gamma/gamma_n)

    Returns
    -------
    array_like
        2D array structure the same size as elevation with precipitation [mm hr-1]
    """

    # --- wind components
    u0 = -np.sin(param['winddir'] * 2 * np.pi / 360) * param['windspeed']
    v0 = np.cos(param['winddir'] * 2 * np.pi / 360) * param['windspeed']

    # --- other factors
    f_coriolis = 2 * 7.2921e-5 * np.sin(param['latitude'] * np.pi / 180)

    # --- pad raster boundaries prior to FFT
    calc_pad = int(np.ceil(((sum(elevation.shape))) / 2) / 100 * 100)
    pad = min([calc_pad, 200])

    h = np.pad(elevation, pad, 'constant')
    nx, ny = h.shape

    # --- FFT
    hhat = np.fft.fft2(h)

    x_n_value = np.fft.fftfreq(ny, (1. / ny))
    y_n_value = np.fft.fftfreq(nx, (1. / nx))

    x_len = nx * dx
    y_len = ny * dy
    kx_line = 2 * np.pi * x_n_value / x_len
    ky_line = 2 * np.pi * y_n_value / y_len
    kx = np.tile(kx_line, (nx, 1))
    ky = np.tile(ky_line[:, None], (1, ny))

    # --- vertical wave number (m)
    sigma = kx * u0 + ky * v0

    mf_num = param['nm']**2 - sigma**2
    mf_den = sigma**2 - f_coriolis**2

    # numerical stability
    mf_num[mf_num < 0] = 0.
    mf_den[(mf_den < EPS) & (mf_den >= 0)] = EPS
    mf_den[(mf_den > -EPS) & (mf_den < 0)] = -EPS
    sign = np.where(sigma >= 0, 1, -1)

    m = sign * np.sqrt(np.abs(mf_num / mf_den * (kx**2 + ky**2)))

    # --- transfer function
    P_karot = ((param['cw'] * 1j * sigma * hhat) /
               ((1 - (param['hw'] * m * 1j)) *
                (1 + (sigma * param['tau_c'] * 1j)) *
                (1 + (sigma * param['tau_f'] * 1j))))

    # --- inverse FFT, de-pad, convert units, add uniform rate
    P = np.fft.ifft2(P_karot)
    P = np.real(P[pad:-pad, pad:-pad])
    P *= 3600   # mm hr-1
    P += param['p0']
    P[P < 0] = 0

    return P
