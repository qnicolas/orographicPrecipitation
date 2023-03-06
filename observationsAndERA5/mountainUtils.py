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
import metpy.calc as mpcalc
from metpy.units import units

def timesel(times,dayslist):
    return [d in dayslist for d in pd.to_datetime(np.array(times)).strftime("%Y%m%d")]

def crossslope_avg(var,center=0.5,halfwidth=0.25):
    m=len(var.y)
    return var.isel(y=slice(int(np.floor(m*(center-halfwidth))),int(np.ceil(m*(center+halfwidth))))).mean('y')

def sel_box(var,box,lon='longitude',lat='latitude'):
    return var.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})

def sel_months(ds,months):
    try: 
        return ds.sel(month = np.in1d( ds['month'], months))
    except KeyError:
        return ds.sel(time = np.in1d( ds['time.month'], months))


def sel_box_months(var,box,months,lon='longitude',lat='latitude'):
    window = sel_box(var,box,lon,lat)
    if "month" in var.dims:
        window=sel_months(window,months).mean('month')
    return window


def rotate_var(var,angle,mountaintop=0,two_dim=True,**rotate_args):
    dx = 2.*np.pi*6400/360*(var.longitude[1]-var.longitude[0])# approximate everything as at the equator
    n=len(var.longitude)
    m=len(var.latitude)
    x = np.arange(0,dx*(n+1),dx)-mountaintop
    y = np.arange(0,dx*m,dx)
    
    if two_dim:
        var_rot_ar = rotate(np.array(var)[::-1],90-angle,reshape=False,cval=np.nan,**rotate_args)
        return xr.DataArray(var_rot_ar,coords={'x':x[:var_rot_ar.shape[-1]],'y':y[:var_rot_ar.shape[0]]},dims=['y','x'])
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
    def __init__(self, name, box, Lname, angle, months, mountaintop, path = '/global/cscratch1/sd/qnicolas/regionsDataBig/'):
        self.name=name
        self.box=box
        self.Lname=Lname
        self.angle = angle
        self.months=months
        self.vars={}
        self.vars_rot={}
        self.x_mountaintop = mountaintop
        self.path=path
        
    def set_boxes(self,box_upstream,box_above,box_tilted):
        self.box_upstream = box_upstream
        self.box_above = box_above
        self.box_tilted = box_tilted
    
    def set_2dvar(self,varname,var):
        self.vars[varname] = sel_box_months(var,self.box,self.months)
        self.vars_rot[varname] = rotate_var(self.vars[varname],self.angle,self.x_mountaintop)
        
    def set_uperp(self):
        self.vars['VAR_100U_PERP'] = crossslopeflow(self.vars['VAR_100U'], self.vars['VAR_100V'],self.angle)
        self.vars_rot['uperp'] = crossslopeflow(self.vars_rot['VAR_100U'], self.vars_rot['VAR_100V'],self.angle)

    def set_3dvar(self,varname,storage_name):
        stored_file=self.path+'e5.monthly.%s.%s.2001-2020.nc'%(storage_name,self.name)
        if os.path.isfile(stored_file):
            self.vars[varname] = sel_months(xr.open_dataarray(stored_file).groupby('time.month').mean(),self.months).mean('month')
        else :
            print("Computing %s ..."%varname)
            monthlyvar = e5_monthly_timeseries(storage_name,years=range(2001,2021),box=self.box)
            self.vars[varname] = sel_months(monthlyvar.groupby('time.month').mean(),self.months).mean('month')
            monthlyvar.to_netcdf(stored_file)
            print("Done ! and stored in %s"%self.path)
        self.vars_rot[varname] = rotate_var(self.vars[varname],self.angle,self.x_mountaintop,two_dim=False)
        
    def set_4dvar(self,varname,storage_name):
        stored_file=self.path+'e5.monthly.%s.%s.2001-2020.nc'%(storage_name,self.name)
        if os.path.isfile(stored_file):
            self.vars[varname] = xr.open_dataarray(stored_file)
        else :
            print("Computing %s ..."%varname)
            self.vars[varname] = e5_monthly_timeseries(storage_name,years=range(2001,2021),box=self.box)
            self.vars[varname].to_netcdf(stored_file)
            print("Done ! and stored in %s"%self.path)
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

        
def plot_xz(ax,region,varname,pert=False,fact=1,center=0.5,halfwidth=0.25,**plot_kwargs):
    if pert:
        cv=crossslope_avg(fact*region.vars_rot[varname],center=center,halfwidth=halfwidth)
        (cv-cv.sel(x=slice(cv.x[0],cv.x[0]+200)).mean('x')).plot.contourf(ax=ax,yincrease=False,**plot_kwargs)
    else:
        crossslope_avg(fact*region.vars_rot[varname],center=center,halfwidth=halfwidth).plot.contourf(ax=ax,yincrease=False,**plot_kwargs)
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
        
def topography_pr_wind_plot(ax,box,z,pr,u,v,prlevs=None,windvect_density=0.1,**plot_kwargs):
    ax.coastlines()
    z.plot.contour(ax=ax,transform=ccrs.PlateCarree(),cmap=plt.cm.Oranges,levels=np.arange(250.,1500.,250.),linewidths=0.8)

    if prlevs is None:
        prlevs = np.linspace(0,np.ceil(pr.max()/10)*10,21)
    pr.plot.contourf(ax=ax,levels=prlevs,**plot_kwargs)
    
    X = u.latitude.expand_dims({"longitude":u.longitude}).transpose()
    Y = u.longitude.expand_dims({"latitude":u.latitude})
    n=int(windvect_density*len(u.latitude))
    m=int(windvect_density*len(u.longitude))
    q=ax.quiver(np.array(Y)[::n,::m],np.array(X)[::n,::m], np.array(u)[::n,::m], np.array(v)[::n,::m], transform=ccrs.PlateCarree(),color="w",width=0.003,scale=90)
    qk = ax.quiverkey(q, 0.87, 1.03, 5, r'5 m s$^{-1}$', labelpos='E',
                           coordinates='axes',color='k')
    dl=5
    
    lats=range(dl*(1+(int(box[2])-1)//dl),dl*(int(box[3])//dl)+1,dl)
    lons=range(dl*(1+(int(box[0])-1)//dl),dl*(int(box[1])//dl)+1,dl)
    ax.set_xticks([(lon+180)%360-180 for lon in lons])
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
    distance_across = np.sqrt(np.maximum(0.,(x-x1)**2+(y-y1)**2 - sc_prod**2/((x2-x1)**2+(y2-y1)**2)))
    return (halfplane_para*halfplane_perp_up*halfplane_perp_dn*(distance_across<width)).transpose('latitude','longitude')

def BL_plevs(t,q,lft_top_pressure,pname='level',fixed_thetaeb=False):
    p = t**0*t[pname]
    t_dew = mpcalc.dewpoint_from_specific_humidity(np.array(q), np.array(t)*units.K, np.array(p)*units.hPa)
    zero_array = 0.*p
    thetae = zero_array+np.array(mpcalc.equivalent_potential_temperature(np.array(p)*units.hPa,
                                                                         np.array(t)*units.K,
                                                                         t_dew
                                                                        )/units.K)
    thetaestar = zero_array+np.array(mpcalc.saturation_equivalent_potential_temperature(np.array(p)*units.hPa,
                                                                                        np.array(t)*units.K
                                                                                       )/units.K)
    kappaL=3
    g=9.81
    wB = 0.52
    wL = 1-wB
    thetae0 = 340
    if fixed_thetaeb:
        thetaeB = 350
    else:
        thetaeB     = thetae    .where( p>=900                         ).mean(pname)
    thetaeL     = thetae    .where((p< 900.) & (p>lft_top_pressure)).mean(pname)
    thetaeLstar = thetaestar.where((p< 900.) & (p>lft_top_pressure)).mean(pname)
    
    capeL   = (thetaeB/thetaeLstar - 1)*thetae0
    subsatL = (1 - thetaeL/thetaeLstar)*thetae0
    BL = g/kappaL/thetae0*(wB*capeL-wL*subsatL)
    return BL

def thetae_perso(t,q,es,qs,pname='level'):
    """t in K, t[pname] and es in hPa, q and qs in kg/kg. returns thetae in K"""
    r  = q/(1-q)
    rh = q/qs
    # Magnus formula for dew point
    b=18.678;c=257.14
    gamma = np.log(rh)+b*(t-275.15)/(c+(t-273.15))
    tdew = 273.15+c*gamma/(b-gamma)
    tL = 1/(1/(tdew-56)+np.log(t/tdew)/800)+56
    e = rh*es
    
    kappad=0.2854
    thetaL = t*(1e3/(t[pname]-e))**kappad*(t/tL)**(0.28*r)
    thetae = thetaL*np.exp((3036/tL-1.78)*r*(1+0.448*r))
    return thetae

def thetaestar_perso(t,es,qs,pname='level'):
    """t in K, t[pname] and es in hPa, qs in kg/kg. returns thetae^* in K"""
    rs=qs/(1-qs)
    kappad=0.2854
    thetaL = t*(1e3/(t[pname]-es))**kappad
    thetae = thetaL*np.exp((3036/t-1.78)*rs*(1+0.448*rs))
    return thetae

from orographicPrecipitation.precip_model_functions import humidsat
def BL_plevs_vectorized(t,q,lft_top_pressure,lft_bot_pressure,pname='level',fixed_thetaeb=False):
    t = t.sel({pname:slice(lft_top_pressure,1100.)})
    q = q.sel({pname:slice(lft_top_pressure,1100.)})
    
    es,qs,_=humidsat(t,t[pname])
    thetae =     thetae_perso(t,q,es,qs,pname=pname)
    thetaestar = thetaestar_perso(t,es,qs,pname=pname)  
        
    if fixed_thetaeb:
        thetaeB = 350
    else:
        thetaeB = thetae    .sel({pname:slice(lft_bot_pressure,1100)            }).mean(pname)
    thetaeL     = thetae    .sel({pname:slice(lft_top_pressure,lft_bot_pressure-1)}).mean(pname)
    thetaeLstar = thetaestar.sel({pname:slice(lft_top_pressure,lft_bot_pressure-1)}).mean(pname)
    return compute_BL(thetaeB,thetaeL,thetaeLstar)

def BLsimple_plevs_vectorized(t,q,kind='shallow',pname='level',fixed_thetaeb=False):
    if kind=='shallow':
        eb = t.sel({pname:slice(900,1100.)}) + 2.5e6/1004*q.sel({pname:slice(900,1100.)})
        tL = t.sel({pname:slice(700,899.)})
        eL = tL + 2.5e6/1004*q.sel({pname:slice(700,899.)})
        _,qsL,_=humidsat(tL,tL[pname])
        eLstar = tL + 2.5e6/1004*qsL
    elif kind=='deep':
        eb = t.sel({pname:slice(850,1100.)}) + 2.5e6/1004*q.sel({pname:slice(850,1100.)})
        tL = t.sel({pname:slice(500,849.)})
        eL = tL + 2.5e6/1004*q.sel({pname:slice(500,849.)})
        _,qsL,_=humidsat(tL,tL[pname])
        eLstar = tL + 2.5e6/1004*qsL
    return compute_BLsimple(eb,eL,eLstar,kind='shallow')

###################### SECOND PART ########################

def compute_BL(thetaeB,thetaeL,thetaeLstar,wB=0.52):
    g=9.81
    wL = 1-wB
    thetae0 = 340
    capeL   = (thetaeB/thetaeLstar - 1)*thetae0
    subsatL = (1 - thetaeL/thetaeLstar)*thetae0
    BL = g/thetae0*(wB*capeL-wL*subsatL) # Did not include the KappaL term to conform with AAN2020
    return BL

def compute_BLsimple(eb,eL,eLstar,kind='shallow'):
    kappaL=3
    g=9.81
    wB = 0.52
    wL = 1-wB
    if kind=='shallow':
        Pib = 0.985
        PiL = 0.938
    elif kind=='deep':
        Pib = 0.97
        PiL = 0.88
    elif kind=='semi':
        Pib = 0.985
        PiL = 0.91
    BLsimple = g/eLstar*PiL*(wB*eb/Pib+(wL*eL-eLstar)/PiL)
    return BLsimple

def tilted_rect_distance(grid,x1,y1,x2,y2,distups1,distups2,reverse=False):
    x = grid.longitude
    y = grid.latitude
    scalarprod_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1)
    scalarprod_perp_up = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    scalarprod_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2)
    if reverse:
        halfplane_para = scalarprod_para <=0
    else:
        halfplane_para = scalarprod_para >=0
    halfplane_perp_up = scalarprod_perp_up >= 0
    halfplane_perp_dn = scalarprod_perp_dn >= 0
    distance_across = np.sqrt((x-x1)**2+(y-y1)**2 - scalarprod_perp_up**2/((x2-x1)**2+(y2-y1)**2))*np.sign(scalarprod_para)
    return (halfplane_perp_up*halfplane_perp_dn*(distance_across<distups1)*(distance_across>distups2)).transpose('latitude','longitude')


class MountainRangeCustom(MountainRange):
    def __init__(self, name, box, Lname, angle, months, box_tilted, path= '/global/cscratch1/sd/qnicolas/regionsDataBig/'):
        super().__init__(name, box, Lname, angle, months, 0.,path)
        self.box_tilted = box_tilted
        self.years = range(2001,2021)
        self._monthstr = '-'.join(["{:02}".format(m) for m in self.months])
        
    def set_daily_sfc_var(self,varcode,varname):
        filepath = self.path+"e5.oper.an.sfc.{}.ll025sc.{}-{}.{}.{}.nc".format(varcode,self.years[0],self.years[-1],self._monthstr,self.name)
        self.vars[varname+'_DAILY'] = xr.open_dataarray(filepath)
        
    def set_daily_spavg_var(self,varcode,varname,group='sfc',where='above'):
        filepath = self.path+"e5.oper.an.{}.{}.ll025sc.{}-{}.{}.{}.{}mean.nc".format(group,varcode,self.years[0],self.years[-1],self._monthstr,self.name,where)
        self.vars[varname+'_'+ where.upper()+'_DAILY'] = xr.open_dataarray(filepath)
                    
    def set_daily_Bl_vars(self,kind='shallow'):
        if kind=='shallow':
            suffix=''
            wB = 0.55
        elif kind=='deep':
            suffix='deep'
            wB = 0.52
        elif kind=='semi':
            suffix='semi'
            wB=0.52
        filepaths = [self.path+"e5.diagnostic.{}{}.{}-{}.{}.{}.nc".format(varcode,suffix,self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("thetaeb", "thetaeL", "thetaeLstar", "tL", "qL")]
        thetaeb = xr.open_dataarray(filepaths[0])
        thetaeL = xr.open_dataarray(filepaths[1])
        thetaeLstar = xr.open_dataarray(filepaths[2])
        tL = xr.open_dataarray(filepaths[3])
        qL = xr.open_dataarray(filepaths[4])
        self.vars['THETAEB{}_DAILY'.format(suffix.upper())] = thetaeb
        self.vars['TL{}_DAILY'.format(suffix.upper())] = tL
        self.vars['QL{}_DAILY'.format(suffix.upper())] = qL
        BL = compute_BL(thetaeb,thetaeL,thetaeLstar,wB)
        self.vars['BL{}_DAILY'.format(suffix.upper())] = BL
        
    def set_daily_DBl_vars(self):
        suffix='semi'
        wB=0.52
        filepaths = [self.path+"e5.diagnostic.{}{}.{}-{}.{}.{}.nc".format(varcode,'dbl',self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("thetaeb", "thetaeL", "thetaeLstar", "tL", "qL")]
        thetaeb = xr.open_dataarray(filepaths[0])
        thetaeL = xr.open_dataarray(filepaths[1])
        thetaeLstar = xr.open_dataarray(filepaths[2])
        tL = xr.open_dataarray(filepaths[3])
        qL = xr.open_dataarray(filepaths[4])
        self.vars['THETAEB{}_DAILY'.format(suffix.upper())] = thetaeb
        self.vars['TL{}_DAILY'.format(suffix.upper())] = tL
        self.vars['QL{}_DAILY'.format(suffix.upper())] = qL
        BL = compute_BL(thetaeb,thetaeL,thetaeLstar,wB)
        self.vars['BL{}_DAILY'.format(suffix.upper())] = BL
                
#    def set_daily_Blsimple_vars(self,kind='shallow'):
#        if kind=='shallow':
#            suffix=''
#        else:
#            suffix=kind
#        filepaths = [self.path+"e5.diagnostic.{}{}.{}-{}.{}.{}.nc".format(varcode,suffix,self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("eb","eL","eLstar","tL","qL")]
#        eb = xr.open_dataarray(filepaths[0])#.mean('level')
#        eL = xr.open_dataarray(filepaths[1])#.mean('level')
#        eLstar = xr.open_dataarray(filepaths[2])#.mean('level')
#        tL = xr.open_dataarray(filepaths[3])
#        qL = xr.open_dataarray(filepaths[4])
#        self.vars['EB{}_DAILY'.format(suffix.upper())] = eb
#        self.vars['EL{}_DAILY'.format(suffix.upper())] = eL
#        self.vars['ELSTAR{}_DAILY'.format(suffix.upper())] = eLstar
#        self.vars['TL{}_DAILY'.format(suffix.upper())] = tL
#        self.vars['QL{}_DAILY'.format(suffix.upper())] = qL
#        BLsimple = compute_BLsimple(eb,eL,eLstar,kind=kind)
#        self.vars['BL{}SIMPLE_DAILY'.format(suffix.upper())] = BLsimple
#        
#    def set_daily_uLvL(self, extra_levels=[]):
#        filepaths = [self.path+"e5.diagnostic.{}.{}-{}.{}.{}.nc".format(varcode,self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("uL","vL")]
#        uL = xr.open_dataarray(filepaths[0])
#        vL = xr.open_dataarray(filepaths[1])
#        self.vars['UL_DAILY'] = uL
#        self.vars['VL_DAILY'] = vL
#        for suffix in extra_levels:
#            filepaths = [self.path+"e5.diagnostic.{}.{}-{}.{}.{}.nc".format(varcode,self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("u"+suffix,"v"+suffix)]
#            u = xr.open_dataarray(filepaths[0])
#            v = xr.open_dataarray(filepaths[1])
#            self.vars['U{}_DAILY'.format(suffix)] = u
#            self.vars['V{}_DAILY'.format(suffix)] = v
        
    def set_spatialmean(self,varname,locname,mask,box=None):
        #mask = tilted_rect(BL,*self.box_tilted,reverse=False)
        self.vars[varname +'_'+ locname.upper()+'_DAILY'] = spatial_mean(self.vars[varname+'_DAILY'],box=box,mask=mask)

    def set_uperp_sfc(self):
        self.vars['VAR_100U_PERP_DAILY'] = crossslopeflow(self.vars['VAR_100U_DAILY'], self.vars['VAR_100V_DAILY'],self.angle)
#    def set_uperp_L(self,angle, extra_levels=[]):
#        self.vars['UL_PERP_DAILY'] = crossslopeflow(self.vars['UL_DAILY'], self.vars['VL_DAILY'],angle)
#        for suffix in extra_levels:
#            self.vars['U{}_PERP_DAILY'.format(suffix)] = crossslopeflow(self.vars['U{}_DAILY'.format(suffix)], self.vars['V{}_DAILY'.format(suffix)],angle)
    def set_viwvperp_sfc(self):
        self.vars['VIWV_PERP_UPSTREAM_DAILY'] = crossslopeflow(self.vars['VIWVE_UPSTREAM_DAILY'], self.vars['VIWVN_UPSTREAM_DAILY'],self.angle)
                
    def set_daily_imerg(self):
        filepath = self.path+"gpm_imerg_v06.{}-{}.{}.{}.nc".format(self.years[0],self.years[-1],self._monthstr,self.name)
        self.vars['GPM_PR_DAILY'] = xr.open_dataarray(filepath)
        