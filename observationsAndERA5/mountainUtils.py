import os
import numpy as np
import xarray as xr
from scipy.ndimage import rotate
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size':15})
import time
import cartopy.crs as ccrs
import cartopy
import sys
p = os.path.abspath('/global/homes/q/qnicolas/')
if p not in sys.path:
    sys.path.append(p)
from tools.e5tools import *

def timesel(times,dayslist):
    return [d in dayslist for d in pd.to_datetime(np.array(times)).strftime("%Y%m%d")]

def crossslope_avg(var,center=0.5,halfwidth=0.25):
    m=len(var.y)
    return var.isel(y=slice(int(np.floor(m*(center-halfwidth))),int(np.ceil(m*(center+halfwidth))))).mean('y')

def sel_box_month(var,box,month,lon='longitude',lat='latitude'):
    window = var.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
    if "month" in var.dims:
        window=window.sel(month=month)
    return window

def rotate_var(var,angle,mountaintop=0,two_dim=True,**rotate_args):
    dx = 2.*np.pi*6400/360*(var.longitude[1]-var.longitude[0])# approximate everything as at the equator
    n=len(var.longitude)
    m=len(var.latitude)
    x = np.arange(0,dx*(n+1),dx)-mountaintop
    y = np.arange(0,dx*m,dx)
    
    if two_dim:
        var_rot_ar = rotate(np.array(var)[::-1],90-angle,reshape=False,cval=np.nan,**rotate_args)
        return xr.DataArray(var_rot_ar,coords={'x':x[:var_rot_ar.shape[-1]],'y':y},dims=['y','x'])
    else:
        var_rot_ar = rotate(np.array(var)[...,::-1,:],90-angle,axes=(-1,-2),reshape=False,cval=np.nan,**rotate_args)
        coords = dict(var.coords).copy()
        if "latitude" in coords: 
            del coords['latitude']
        if "longitude" in coords: 
            del coords['longitude']
        return xr.DataArray(var_rot_ar,coords={'x':x[:var_rot_ar.shape[-1]],'y':y,**coords},dims=[*var.dims[:-2],'y','x'])

def crossslopeflow(u,v,angle):
    return (u*np.sin(angle*np.pi/180)+v*np.cos(angle*np.pi/180))

class MountainRange :
    def __init__(self, name, box, Lname, angle, month, mountaintop):
        self.name=name
        self.box=box
        self.Lname=Lname
        self.angle = angle
        self.pr_month=month
        self.vars={}
        self.vars_rot={}
        self.x_mountaintop = mountaintop
        
    def set_boxes(self,box_upstream,box_above,box_tilted):
        self.box_upstream = box_upstream
        self.box_above = box_above
        self.box_tilted = box_tilted
    
    def set_2dvar(self,varname,var):
        self.vars[varname] = sel_box_month(var,self.box,self.pr_month)
        self.vars_rot[varname] = rotate_var(self.vars[varname],self.angle,self.x_mountaintop)
        
    def set_uperp(self):
        self.vars['VAR_100U_PERP'] = crossslopeflow(self.vars['VAR_100U'], self.vars['VAR_100V'],self.angle)
        self.vars_rot['uperp'] = crossslopeflow(self.vars_rot['VAR_100U'], self.vars_rot['VAR_100V'],self.angle)

    def set_3dvar(self,varname,storage_name):
        stored_file='/global/cscratch1/sd/qnicolas/regionsData/e5.climatology.%s.%s.2001-2019.nc'%(storage_name,self.name)
        if os.path.isfile(stored_file):
            self.vars[varname] = xr.open_dataarray(stored_file)
        else :
            print("Computing %s ..."%varname)
            self.vars[varname] = e5_climatology(storage_name,years=range(2001,2019),box=self.box,level=None,chunks=None,month=self.pr_month)
            self.vars[varname].to_netcdf(stored_file)
            print("Done ! and stored in SCRATCH/temp/")
        self.vars_rot[varname] = rotate_var(self.vars[varname],self.angle,self.x_mountaintop,two_dim=False)
        
    def set_4dvar(self,varname,storage_name):
        stored_file='/global/cscratch1/sd/qnicolas/regionsData/e5.monthly.%s.%s.2001-2019.nc'%(storage_name,self.name)
        if os.path.isfile(stored_file):
            self.vars[varname] = xr.open_dataarray(stored_file)
        else :
            print("Computing %s ..."%varname)
            self.vars[varname] = e5_monthly_timeseries(storage_name,years=range(2001,2019),box=self.box)
            self.vars[varname].to_netcdf(stored_file)
            print("Done ! and stored in SCRATCH/regionsData/")
        self.vars_rot[varname] = rotate_var(self.vars[varname],self.angle,self.x_mountaintop,two_dim=False)
        
    def set_othervar(self,varname,var):
        self.vars[varname] = var
        self.vars_rot[varname] = rotate_var(var,self.angle,self.x_mountaintop,two_dim=False)
    
    def set_uperp3d(self):
        self.vars['U_PERP'] = crossslopeflow(self.vars['U'], self.vars['V'],self.angle)
        self.vars_rot['U_PERP'] = crossslopeflow(self.vars_rot['U'], self.vars_rot['V'],self.angle)
        
def compute_N(T,pfactor,pname='pressure'):
    """T in K, p in Pa and N in s"""
    g = 9.81; R=287.
    rho = pfactor*T[pname]/R/T
    theta = T*(pfactor*T[pname]/1e5)**(-0.287)
    return np.sqrt(-rho*g*g/theta*theta.differentiate(pname)/pfactor)

        
def plot_xz(ax,region,varname,pert=False,fact=1,**plot_kwargs):
    if pert:
        cv=crossslope_avg(fact*region.vars_rot[varname])
        (cv-cv.sel(x=slice(cv.x[0],cv.x[0]+200)).mean('x')).plot.contourf(ax=ax,yincrease=False,**plot_kwargs)
    else:
        crossslope_avg(fact*region.vars_rot[varname]).plot.contourf(ax=ax,yincrease=False,**plot_kwargs)
    p_sfc = crossslope_avg(999-region.vars_rot['Z']/(1.1*9.81))
    p_sfc.plot(ax=ax,color='k')
    ax.fill_between(p_sfc.x,0*p_sfc+1000.,p_sfc,color='w')
    
def plot_z(ax,region,varname,x0,pert=False,**plot_kwargs):
    p_sfc = crossslope_avg(999-region.vars_rot['Z']/(1.1*9.81)).sel(x=x0,method='nearest')
    if pert:
        cv=crossslope_avg(region.vars_rot[varname])
        (cv-cv.sel(x=slice(cv.x[0],cv.x[0]+200)).mean('x')).sel(x=x0,method='nearest').where(cv.level<p_sfc).plot(ax=ax,y='level',yincrease=False,**plot_kwargs)
    else:
        crossslope_avg(region.vars_rot[varname]).sel(x=x0,method='nearest').where(cv.level<p_sfc).plot(ax=ax,y='level',yincrease=False,**plot_kwargs)

def plot_z_diff(ax,var,box1,box2,**plot_kwargs):
        (sel_box_month(var,box2,0).mean(['latitude','longitude'])-sel_box_month(var,box1,0).mean(['latitude','longitude'])).plot(ax=ax,y='level',yincrease=False,**plot_kwargs)
        
def topography_pr_wind_plot(ax,box,z,pr,u,v,prlevs=None,**plot_kwargs):
    ax.coastlines()
    z.plot.contour(ax=ax,transform=ccrs.PlateCarree(),cmap=plt.cm.Oranges,levels=np.arange(250.,1500.,250.),linewidths=0.8)

    if prlevs is None:
        prlevs = np.linspace(0,np.ceil(pr.max()/10)*10,21)
    pr.plot.contourf(ax=ax,levels=prlevs,**plot_kwargs)
    
    X = u.latitude.expand_dims({"longitude":u.longitude}).transpose()
    Y = u.longitude.expand_dims({"latitude":u.latitude})
    n=len(u.latitude)//10
    m=len(u.longitude)//10
    q=ax.quiver(np.array(Y)[::n,::m],np.array(X)[::n,::m], np.array(u)[::n,::m], np.array(v)[::n,::m], transform=ccrs.PlateCarree(),color="w",width=0.003,scale=90)
    qk = ax.quiverkey(q, 0.87, 1.03, 5, r'5 m s$^{-1}$', labelpos='E',
                           coordinates='axes',color='k')
    dl=5
    
    lats=range(dl*(1+(int(box[2])-1)//dl),dl*(int(box[3])//dl)+1,dl)
    lons=range(dl*(1+(int(box[0])-1)//dl),dl*(int(box[1])//dl)+1,dl)
    ax.set_xticks(lons)
    ax.set_xticklabels(lons)
    ax.set_yticks(lats)
    ax.set_yticklabels(lats)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim((box[2],box[3]))
    

def tilted_rect(grid,x1,y1,x2,y2,width,reverse=False):
    x = grid.longitude
    y = grid.latitude
    if reverse:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) <=0
    else:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) >=0
    sc_prod = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    halfplane_perp_up = sc_prod >= 0
    halfplane_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2) >= 0
    distance_across = np.sqrt((x-x1)**2+(y-y1)**2 - sc_prod**2/((x2-x1)**2+(y2-y1)**2))
    return (halfplane_para*halfplane_perp_up*halfplane_perp_dn*(distance_across<width)).transpose('latitude','longitude')