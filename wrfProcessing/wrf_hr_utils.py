import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def extract_wrfout(simulation_path,nfiles=False):
    files=sorted(glob.glob(simulation_path+'wrfout_d01_197*'))
    if nfiles:
        files=files[:nfiles]
    return xr.open_mfdataset(files,combine='nested',concat_dim='Time',parallel=True)
def extract_wrfsub(simulation_path,prefix='wrfout',nfiles=False):
    files=sorted(glob.glob(simulation_path+'%s_d01_197*subset.nc'%prefix))
    if nfiles:
        files=files[:nfiles]
    return xr.open_mfdataset(files,combine='nested',concat_dim='Time',parallel=True)
def extract_wrfpro(simulation_path,nfiles=False):
    files=sorted(glob.glob(simulation_path+'wrfpout_d01_197*'))
    if nfiles:
        files=files[:nfiles]
    return xr.open_mfdataset(files,combine='nested',concat_dim='Time',parallel=True)

def plotsection(hgt,figsize=(15,4)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.set_xlabel("distance west of mountain top (km)")
    ax.set_ylabel("height(m)")
    ax.plot((1635-hgt.west_east)*3,hgt,color='k')
    return ax,ax.twinx()

def f(x):
    return np.sign(x)*x**2

def change_coords_sfc(w,center=1635,dx=3,flip_x=True):
    return w.assign_coords({'distance_from_mtn':(-1)**(1-flip_x)*(center-w.west_east)*dx}).swap_dims({'west_east':'distance_from_mtn'})

def change_coords_press(sim,w):
    return w.assign_coords({'pressure':sim.P_PL[0]/100}).swap_dims({'num_press_levels_stag':'pressure'})

def change_coords_etav(sim,w,staggered=0):
    if staggered:
        return w.assign_coords({'eta_level':sim.ZNW.isel(Time=-1)}).swap_dims({'bottom_top_stag':'eta_level'}).persist()
    else:
        return w.assign_coords({'eta_level':sim.ZNU.isel(Time=-1)}).swap_dims({'bottom_top':'eta_level'}).persist()

def change_coords_pl(sim,w,**x_kwargs):
    return change_coords_sfc(change_coords_press(sim,w),**x_kwargs)

def change_coords_eta(sim,w,staggered=0,**x_kwargs):
    return change_coords_sfc(change_coords_etav(sim,w,staggered),**x_kwargs)



def interp_eta_to_pressure(ds,var,staggered=1,plevs=None,dsinput=None):
    """Make sure eta_level is the first dimension"""
    if plevs is None:
        plevs= np.arange(125.,990.,20.)
    if dsinput is None:
        dsinput=ds
    rep=np.zeros((len(plevs),len(var[0])))
    PB_rev = np.array(dsinput.PB[0,:,0]+ds.P[-20:].mean(['Time','south_north']))[::-1]
    var_rev=np.array(var)[::-1]
    if staggered:
        i=0
        ZNW = np.array(ds.ZNW[0])[::-1]
        ZNU=  np.array(ds.ZNU[0])[::-1]
        for i in range(len(var[0])):
            PB_rev_stag = np.interp(ZNW,ZNU,PB_rev[:,i])
            rep[:,i]=np.interp(plevs*100,PB_rev_stag,var_rev[:,i])
            rep[plevs*100>PB_rev_stag[-1],i]=np.nan
    else:
        for i in range(len(var[0])):
            rep[:,i]=np.interp(plevs*100,PB_rev[:,i],var_rev[:,i])
            rep[plevs*100>PB_rev[-1,i],i]=np.nan
    return xr.DataArray(rep[::-1],coords={'pressure':plevs[::-1],'distance_from_mtn':var.distance_from_mtn},dims=['pressure','distance_from_mtn'])

def interp_eta_to_z(ds,var,staggered=1,zlevs=None,dsinput=None):
    """Make sure eta_level is the first dimension"""
    if zlevs is None:
        zlevs= np.arange(0.,16000.,200.)
    if dsinput is None:
        dsinput=ds
    rep = np.zeros((len(zlevs),len(var[0])))
    PHB = np.array(dsinput.PHB[0,:,0]+ds.PH[-100:].mean(['Time','south_north']))/9.81
    var_ar=np.array(var)
    if not staggered:
        ZNW_rev = np.array(ds.ZNW[0])[::-1]
        ZNU=  np.array(ds.ZNU[0])
        for i in range(len(var[0])):
            PHB_unstag = np.interp(ZNU,ZNW_rev,PHB[::-1,i])
            rep[:,i]=np.interp(zlevs,PHB_unstag,var_ar[:,i],left=np.nan,right=np.nan)
    else:
        for i in range(len(var[0])):
            rep[:,i]=np.interp(zlevs,PHB[:,i],var_ar[:,i],left=np.nan,right=np.nan)
    return xr.DataArray(rep,coords={'z':zlevs,'distance_from_mtn':var.distance_from_mtn},dims=['z','distance_from_mtn'])