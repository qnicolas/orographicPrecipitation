import os
import glob
import time

import pandas as pd
import numpy as np
import xarray as xr
import datetime

import sys
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

from orographicPrecipitation.precip_model_functions import qsat
from orographicPrecipitation.precip_extremes_scaling import *

import warnings


########################################################################################
#################################### FUNCTIONS ########################################
########################################################################################

def sel_tropicmountain(ds):
    return ds.sel(west_east=slice(38,40),south_north=slice(0,5))
def sel_midlatmountain(ds):
    return ds.sel(west_east=slice(37,40),south_north=slice(40,59))


def open_wrfds(path,shifttimes):
    ds = xr.open_mfdataset(path,combine='by_coords',use_cftime=True,concat_dim="Time")
    if shifttimes :
        ds['time'] = (ds.indexes['time'].shift(2+(datetime.datetime(1970,1,1)-datetime.datetime(1,1,1)).days,'D')
                      ).to_datetimeindex()
    else:
        ds['time'] = ds.indexes['time'].to_datetimeindex()
    if 'pressure' in ds.coords:
        ds=ds.rename({'pressure':'level'})
    return ds

def spatial_mean(da):
    """Compute spatial mean of a DataArray"""
    if type(da)==int:
        return da
    coslat = xr.ufuncs.cos(xr.ufuncs.deg2rad(da.south_north))
    weight_factor = coslat / coslat.mean('south_north')
    return (da*weight_factor).mean(['south_north','west_east'])

def extreme_vars(pr,w,t,ps,quantile):
    quantile_inf = quantile-0.01
    quantile_sup = quantile+0.01
    pr2=pr.resample(time='6H').ffill()
    if quantile_sup<1:
        return (var.where((pr2 > pr2.chunk({'time': -1}).quantile(quantile_inf,"time")) &
                          (pr2 < pr2.chunk({'time': -1}).quantile(quantile_sup,"time"))
                          ,drop=True).mean('time',skipna=True).compute()  for var in (w,t,ps))
    else :
        return (var.where((pr2 > pr2.chunk({'time': -1}).quantile(quantile_inf,"time"))
                          ,drop=True).mean('time',skipna=True).compute()  for var in (w,t,ps))
    
def scaling3(omega, temp, ps, levels):
    """Same as "scaling", but the arguments are sorted by increasing pressure and plevs are already input.
    scaling3 is to be vectorized by xarray"""
    return scaling(omega[::-1], temp[::-1], levels[::-1], ps)

def ogscaling(w,t,ps):
    ref_density = t.level*100/(287*t.mean(["west_east","south_north"]))
    pr = 86400*xr.apply_ufunc(scaling3,
                              -ref_density*9.81*w,
                              t,
                              100*ps,
                              100*w.level,
                              input_core_dims=[['level'], ['level'],[],['level']],vectorize=True,dask='parallelized',output_dtypes=[float]).compute()
    return pr



########################################################################################
#################################### MAIN CLASS ########################################
########################################################################################

class WrfDataset100km :
    def __init__(self, path, simulation_name, lbl, sel_mountain, color, open_ds=1, shifttimes=1):
        self.path=path
        self.simulation_name=simulation_name
        self.lbl = lbl
        self.sel_mountain=sel_mountain
        self.color = color
        
        t=time.time()
        if open_ds :
            if not os.path.isdir(os.path.join(path,simulation_name)):
                raise FileNotFoundError("No such directory : "+(os.path.join(path,simulation_name)))
            if len(glob.glob(os.path.join(path,simulation_name,simulation_name+'.*sfc.nc'))) > 0: #outputs separated in sfc vars and pressure vars
                ds_sfc = open_wrfds(os.path.join(path,simulation_name,simulation_name+'.*sfc.nc'),shifttimes)
                ds_pl = open_wrfds(os.path.join(path,simulation_name,simulation_name+'.*pl.nc'),shifttimes)
                ds = xr.merge([ds_pl,ds_sfc])
            else :
                ds = open_wrfds(os.path.join(path,simulation_name,simulation_name+'.*.nc'),shifttimes)
                
            self.vars = ds.chunk({'south_north': 35,'west_east': 40,'time':800})
            #number of outputs per day
            self.nhours = len(self.vars.time.sel(time = pd.to_datetime(np.array(self.vars.time.isel(time=0))).strftime("%Y-%m-%d")))
        print("loading time : %.1f s"%(time.time()-t))
        
        t=time.time()
        if os.path.isfile(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_g_daily.nc')) and os.path.isfile(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_c_daily.nc')):
            print("Daily precips already computed")
            self.precip_g_daily = xr.open_dataset(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_g_daily.nc')).precip_g
            self.precip_c_daily = xr.open_dataset(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_c_daily.nc')).precip_c
        else :
            if not open_ds :
                raise ValueError("daily precips haven't been computed previously, open_ds must be set to 1")
            self.precip_g_daily = self.nhours*self.vars.precip_g.isel(time=range(50*self.nhours,len(self.vars.time))).diff('time').assign_coords(time = self.vars.precip_g.time.isel(time=range(50*self.nhours,len(self.vars.time)-1))).resample(time='1D').mean().compute()
            self.precip_c_daily = self.nhours*self.vars.precip_c.isel(time=range(50*self.nhours,len(self.vars.time))).diff('time').assign_coords(time = self.vars.precip_g.time.isel(time=range(50*self.nhours,len(self.vars.time)-1))).resample(time='1D').mean().compute()
            os.makedirs(os.path.join(path,simulation_name,'diags'), exist_ok=True)
            self.precip_g_daily.to_netcdf(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_g_daily.nc'))
            self.precip_c_daily.to_netcdf(os.path.join(path,simulation_name,'diags',simulation_name+'.precip_c_daily.nc'))
        self.precip_daily = self.precip_g_daily + self.precip_c_daily
        print("precips time : %.1f s"%(time.time()-t))
        
        
    def set_orig_vars(self, ds):
        self.orig_vars = ds#.chunk({'south_north': 2,'west_east': 2,'Time':10})
    def set_evap(self):
        self.evap_tmean = self.orig_vars.QFX.isel(Time=range(50*self.nhours,len(self.orig_vars.Time))).mean("Time").compute()*86400

    def set_extreme_precip(self,quantile):
        self.ex_pr = self.precip_daily.chunk({'time': -1}).quantile(quantile,"time")
        self.ex_pr_zonmean = self.ex_pr.mean("west_east")
    def set_extreme_vars(self,quantile):
        if (    os.path.isfile(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.w_p_%i_quantile.nc'%(quantile*100)))
            and os.path.isfile(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.T_p_%i_quantile.nc'%(quantile*100)))
            and os.path.isfile(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.p_sfc_%i_quantile.nc'%(quantile*100)))
           ):
            self.ex_w = xr.open_dataset(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.w_p_%i_quantile.nc'%(quantile*100))).w_p
            self.ex_t = xr.open_dataset(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.T_p_%i_quantile.nc'%(quantile*100))).T_p
            self.ex_ps= xr.open_dataset(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.p_sfc_%i_quantile.nc'%(quantile*100))).p_sfc
        else :
            self.ex_w,self.ex_t,self.ex_ps = extreme_vars(self.precip_daily,
                                                          self.vars.w_p,
                                                          self.vars.T_p,
                                                          self.vars.p_sfc,
                                                          quantile
                                                         )
            self.ex_w.to_netcdf(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.w_p_%i_quantile.nc'%(quantile*100)))
            self.ex_t.to_netcdf(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.T_p_%i_quantile.nc'%(quantile*100)))
            self.ex_ps.to_netcdf(os.path.join(self.path,self.simulation_name,'diags',self.simulation_name+'.p_sfc_%i_quantile.nc'%(quantile*100)))
    def compute_ogscaling(self):
        self.ex_pr_og=ogscaling(self.ex_w,self.ex_t,self.ex_ps)
            
    def set_extreme_precip_mountain(self,quantile):
        mountain_precip = self.sel_mountain(self.precip_daily).mean(["south_north","west_east"])
        self.ex_pr_mountain = mountain_precip.chunk({'time': -1}).quantile(quantile,"time")
        self.ex_pr_mountain_all = mountain_precip.where(mountain_precip>self.ex_pr_mountain,drop=True)
    
    def set_extreme_vars_mountain(self,quantile):
        self.ex_w_mountain,self.ex_t_mountain,self.ex_ps_mountain = extreme_vars(self.sel_mountain(self.precip_daily).mean(["south_north","west_east"]),
                                                                                 self.sel_mountain(self.vars.w_p  ),
                                                                                 self.sel_mountain(self.vars.T_p  ),
                                                                                 self.sel_mountain(self.vars.p_sfc),
                                                                                 quantile
                                                                                )   
    def compute_ogscaling_mountain(self):
        self.ex_pr_mountain_og=ogscaling(self.ex_w_mountain,self.ex_t_mountain,self.ex_ps_mountain).mean(["west_east","south_north"])
    
    def compute_ogscaling_mountain_allevents(self):
        #compute 6-hourly times corresponding to extreme events
        ex_times = [pd.to_datetime(d+' {:0>2}'.format(h)) for d in list(pd.to_datetime(np.array(self.ex_pr_mountain_all.time)).strftime("%Y-%m-%d")) for h in range(0,24,24//self.nhours)]
        ex_w_mountain_all  = self.sel_mountain(self.vars.w_p.sel(time=ex_times)  )
        ex_t_mountain_all  = self.sel_mountain(self.vars.T_p.sel(time=ex_times)  )
        ex_ps_mountain_all = self.sel_mountain(self.vars.p_sfc.sel(time=ex_times))        
        self.ex_pr_mountain_og_all = ogscaling(ex_w_mountain_all,ex_t_mountain_all,ex_ps_mountain_all)
        self.ex_pr_mountain_og_all = self.ex_pr_mountain_og_all.resample(time='1D').mean(["time","west_east","south_north"] ).sel(time=self.ex_pr_mountain_all.time)
        
########################################################################################
####################  SENSITIVITY BREAKDOWN - FIRST TECHNIQUE ##########################
########################################################################################

def dqsat_dp(temp, plev, ps):
    temp = temp[::-1]
    plev = plev[::-1]
    try:
        if plev[0]<plev[1]:
            print(plev)
            raise ValueError('unexpected ordering of pressure levels')
    except ValueError:
        raise
        
    # criterion for identifying tropopause
    crit_lapse_rate = 0.002 # (k/m) for tropopause
    plev_mask = 0.05e5 # (Pa) exclude levels above this as a fail-safe

    dqsat_dp, dqsat_dT,_ = sat_deriv(plev, temp)
    es, qsat, rsat, latent_heat = saturation_thermodynamics(temp, plev, 'era')
    lapse_rate = moist_adiabatic_lapse_rate(temp, plev, 'era')

    # virtual temperature
    temp_virtual = temp*(1.0+qsat*(pars('gas_constant_v')/pars('gas_constant')-1.0))

    # density
    rho = plev/pars('gas_constant')/temp_virtual

    dT_dp = lapse_rate/pars('gravity')/rho

    # find derivative of saturation specific humidity with respect to pressure along 
    # a moist adiabat at the given temperature and pressure for each level
    dqsat_dp_total = dqsat_dp+dqsat_dT*dT_dp

    # mask above tropopause using simple lapse rate criterion
    dT_dp_env = np.gradient(temp, plev)
    lapse_rate_env = dT_dp_env*rho*pars('gravity')

    itrop = np.where(lapse_rate_env>crit_lapse_rate)[0]
    if itrop.size!=0:
        if np.max(itrop)+1<len(plev):
            dqsat_dp_total[np.max(itrop)+1:]=0

    # mask above certain level as fail safe
    dqsat_dp_total[plev<plev_mask]=0
    return dqsat_dp_total[::-1]

def gamma(t,ps):
    return xr.apply_ufunc(dqsat_dp,
                          t,
                          t.level*100.,
                          ps,
                          input_core_dims=[['level'], ['level'], []],output_core_dims=[['level']],vectorize=True,dask='parallelized',output_dtypes=[float]).compute()

def mu(w,t):
    ref_density = t.level*100/(287*t.mean(["west_east","south_north"]))
    return -ref_density*9.81*w / vinteg(-ref_density*9.81*w)

def vinteg(ds):
    """Compute vertical integral of ds in pressure coordinates (integ(variable*dp/g))"""
    return ds.fillna(0.).integrate("level").compute()*100/9.81

def PR(cont,warm,DeltaT):
    return (warm.ex_pr /cont.ex_pr -1.)/DeltaT

def PR_mountain(cont,warm,DeltaT):
    return (warm.ex_pr_mountain /cont.ex_pr_mountain -1.)/DeltaT

def E(cont,warm,DeltaT):
    return ((warm.ex_pr/warm.ex_pr_og)/(cont.ex_pr/cont.ex_pr_og) -1.)/DeltaT

def E_mountain(cont,warm,DeltaT):
    return ((warm.ex_pr_mountain/warm.ex_pr_mountain_og)/(cont.ex_pr_mountain/cont.ex_pr_mountain_og) -1.)/DeltaT

def D1(cont,warm,DeltaT):
    return (vinteg(warm.ex_w) / vinteg(cont.ex_w) -1.)/DeltaT

def D1_mountain(cont,warm,DeltaT):
    return (vinteg(warm.ex_w_mountain).mean(['west_east','south_north'])
            /vinteg(cont.ex_w_mountain).mean(['west_east','south_north']) -1.)/DeltaT

def D2(cont,warm,DeltaT):
    control_gamma = gamma(cont.ex_t,cont.ex_ps)
    warm_gamma    = gamma(warm.ex_t,warm.ex_ps)
    control_mu = mu(cont.ex_w,cont.ex_t)
    warm_mu    = mu(warm.ex_w,warm.ex_t)
    return (vinteg(control_gamma*(warm_mu-control_mu))
            /vinteg(control_gamma*control_mu)            
           )/DeltaT

def D2_mountain(cont,warm,DeltaT):
    control_gamma = gamma(cont.ex_t_mountain,cont.ex_ps_mountain)
    warm_gamma    = gamma(warm.ex_t_mountain,warm.ex_ps_mountain)
    control_mu = mu(cont.ex_w_mountain,cont.ex_t_mountain)
    warm_mu    = mu(warm.ex_w_mountain,warm.ex_t_mountain)
    return (vinteg(control_gamma*(warm_mu-control_mu)).mean(['west_east','south_north'])
           /vinteg(control_gamma*control_mu).mean(['west_east','south_north'])
           /DeltaT)

def T(cont,warm,DeltaT):
    control_gamma = gamma(cont.ex_t,cont.ex_ps)
    warm_gamma    = gamma(warm.ex_t,warm.ex_ps)
    control_mu = mu(cont.ex_w,cont.ex_t)
    warm_mu    = mu(warm.ex_w,warm.ex_t)
    return (vinteg(control_mu*(warm_gamma-control_gamma))
            /vinteg(control_gamma*control_mu)            
           )/DeltaT


def T_mountain(cont,warm,DeltaT):
    control_gamma = gamma(cont.ex_t_mountain,cont.ex_ps_mountain)
    warm_gamma    = gamma(warm.ex_t_mountain,warm.ex_ps_mountain)
    control_mu = mu(cont.ex_w_mountain,cont.ex_t_mountain)
    warm_mu    = mu(warm.ex_w_mountain,warm.ex_t_mountain)
    return (vinteg(control_mu*(warm_gamma-control_gamma)).mean(['west_east','south_north'])
           /vinteg(control_gamma*control_mu).mean(['west_east','south_north'])
           /DeltaT)



########################################################################################
####################  SENSITIVITY BREAKDOWN - SECOND TECHNIQUE #########################
########################################################################################

def bPR(cont,warm,DeltaT,meanfunc):
    return (meanfunc(warm.ex_pr) /meanfunc(cont.ex_pr) -1.)/DeltaT

def bE(cont,warm,DeltaT,meanfunc):
    return (meanfunc((warm.ex_pr/warm.ex_pr_og))/meanfunc((cont.ex_pr/cont.ex_pr_og)) -1.)/DeltaT

def bD1(cont,warm,DeltaT,meanfunc):
    return (meanfunc(vinteg(warm.ex_w)) / meanfunc(vinteg(cont.ex_w)) -1.)/DeltaT
  
def bD2(cont,warm,DeltaT,meanfunc):
    control_gamma = gamma(cont.ex_t,cont.ex_ps)
    warm_gamma    = gamma(warm.ex_t,warm.ex_ps)
    control_mu = mu(cont.ex_w,cont.ex_t)
    warm_mu    = mu(warm.ex_w,warm.ex_t)
    return (meanfunc(vinteg(control_gamma*(warm_mu-control_mu)))
            /meanfunc(vinteg(control_gamma*control_mu))  
           )/DeltaT

def bT(cont,warm,DeltaT,meanfunc):
    control_gamma = gamma(cont.ex_t,cont.ex_ps)
    warm_gamma    = gamma(warm.ex_t,warm.ex_ps)
    control_mu = mu(cont.ex_w,cont.ex_t)
    warm_mu    = mu(warm.ex_w,warm.ex_t)
    return (meanfunc(vinteg(control_mu*(warm_gamma-control_gamma)))
            /meanfunc(vinteg(control_gamma*control_mu))            
           )/DeltaT

