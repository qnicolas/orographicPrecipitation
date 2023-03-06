import numpy as np
import xarray as xr
from orographicConvectionTheory.orographicConvectionTheory import k_vector,compute_Lq,lapse_rates
from orographicConvectionTheory.orographicConvectionTheory import hw_cw

########## ########## ########## 
########## TEMPORARY  ########## 
import time
########## ########## ########## 


def m_exponent_2D(sigma,N,ksq):
    den = sigma**2
    EPS=1e-15
    den[den < EPS] = EPS
    return (den>=N**2)*1j*np.sqrt(ksq*(1-N**2/den)+0.j) + (den<N**2)*np.sign(sigma)*np.sqrt(ksq*(N**2/den-1)+0.j)

def linear_precip_theory_2D(xx,yy,hxy,U,V,N,tauT=3,tauq=11,P0=4.,switch=1,pad_factor=0.2):
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    Lc=2.5e6;g=9.81;cp=1004.
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    tauT*=3600
    tauq*=3600
    
    # Pad boundaries
    calc_pad = int(pad_factor*np.max(hxy.shape))
    pad=calc_pad#pad = min([calc_pad, 200])
    pad_topo = int(100e3/dx)
    hxy_pad_topo = np.pad(hxy, pad_topo, 'linear_ramp',end_values = [0,0])
    hxy_pad = np.pad(hxy_pad_topo,pad-pad_topo,'constant')
    xx_pad = np.pad(xx, pad, 'linear_ramp',end_values = [xx[0]-pad*dx,xx[-1]+pad*dx])
    yy_pad = np.pad(yy, pad, 'linear_ramp',end_values = [yy[0]-pad*dy,yy[-1]+pad*dy])
    
    z=np.arange(0,10000,100)
    kx=k_vector(len(xx_pad),dx)
    ky=k_vector(len(yy_pad),dy)
    sigma = U*kx[:,None]+V*ky[None,:]
    ksq = kx[:,None]**2+ky[None,:]**2
    
    LqovU=compute_Lq(5,1,tauq)

    ds0dz = cp*300.*N**2/g
    _,dq0dz,_ = lapse_rates()
    chi = pT_ov_g * (ds0dz/tauT - dq0dz/tauq)/ Lc * 86400
    
    zbot=1000
    ztop=3000    
    z_slice = z[np.where((z>=zbot) & (z<=ztop))]
    
    
    m1 = m_exponent_2D(sigma,N,ksq)
    mm = np.copy(m1)
    mm[mm==0]=1e-8
    
    Pprimehat = (1j*sigma/(1j*sigma + switch*1/LqovU)) * chi * np.fft.fft2(hxy_pad) * ((m1!=0)*(np.exp( 1j* mm * ztop )-np.exp( 1j* mm * zbot ))/(1j*mm*(ztop-zbot)) + (m1==0)*1) 
    # equivalently np.exp( 1j* m_exponent_2D(sigma,N,ksq)[:,:,None] *  z_slice[None,None,:] ).mean(axis=-1) ?

    P = P0 + np.real(np.fft.ifft2(Pprimehat))
    P = np.maximum(0.,P)[pad:-pad, pad:-pad]
    return xr.DataArray(P,coords={'x':xx,'y':yy},dims=['x','y'])

    
def smith_theory_2D(xx,yy,hxy,U,V,N,gamma_m,ts=300.,ps=100000.,tau=2000, P0=4.,pad_factor=0.2):
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    g=9.81;cp=1004.
    gamma = g/cp - ts*N**2/g
    Hw,Cw = hw_cw(ts,ps,gamma,gamma_m)
    
    tau_c=tau
    tau_f=tau
    
    # Pad boundaries
    calc_pad = int(pad_factor*np.max(hxy.shape))
    pad=calc_pad#pad = min([calc_pad, 200])
    pad_topo = int(100e3/dx)
    hxy_pad_topo = np.pad(hxy, pad_topo, 'linear_ramp',end_values = [0,0])
    hxy_pad = np.pad(hxy_pad_topo,pad-pad_topo,'constant')
    xx_pad = np.pad(xx, pad, 'linear_ramp',end_values = [xx[0]-pad*dx,xx[-1]+pad*dx])
    yy_pad = np.pad(yy, pad, 'linear_ramp',end_values = [yy[0]-pad*dy,yy[-1]+pad*dy])
    
    z=np.arange(0,10000,100)
    kx=k_vector(len(xx_pad),dx)
    ky=k_vector(len(yy_pad),dy)
    sigma = U*kx[:,None]+V*ky[None,:]
    ksq = kx[:,None]**2+ky[None,:]**2
    
    m1 = m_exponent_2D(sigma,N,ksq)
    mm = np.copy(m1)
    mm[mm==0]=1e-8
    Pprimehat= 86400*Cw*np.fft.fft2(hxy_pad)*1j*sigma/(1-1j*Hw*mm)/(1+1j*sigma*tau_c)/(1+1j*sigma*tau_f)/2.5

    P = P0 + np.real(np.fft.ifft2(Pprimehat))
    P = np.maximum(0.,P)[pad:-pad, pad:-pad]
    return xr.DataArray(P,coords={'x':xx,'y':yy},dims=['x','y'])
    
def p_lineartheory_region(MR,topo='ETOPO',N=0.01,tauT=7.5,tauq=27.5,pad='small',switch=1):
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    hxy = np.array(z).T[:,::-1]
    lon = z.longitude
    lat = z.latitude[::-1]
    
    xx = np.array(lon)*100e3
    yy = np.array(lat)*100e3
    
    if pad=='small':
        pf=0.2
    elif pad=='big':
        pf=2
    P = linear_precip_theory_2D(xx,yy,hxy,MR.U0,MR.V0,N,tauT=tauT,tauq=tauq,P0=MR.P0,pad_factor=pf,switch=switch)
    return P.assign_coords({'longitude':P.x/100e3,'latitude':P.y/100e3}).swap_dims({'x':'longitude','y':'latitude'})[:,::-1].transpose()

def p_smiththeory_region(MR,topo='ETOPO',N=0.01,gamma_m=4.32e-3,pad='small'):
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    hxy = np.array(z).T[:,::-1]
    lon = z.longitude
    lat = z.latitude[::-1]
    
    xx = np.array(lon)*100e3
    yy = np.array(lat)*100e3
    
    if pad=='small':
        pf=0.2
    elif pad=='big':
        pf=2
    
    P = smith_theory_2D(xx,yy,hxy,MR.U0,MR.V0,N,gamma_m,P0=MR.P0,pad_factor=pf)
    return P.assign_coords({'longitude':P.x/100e3,'latitude':P.y/100e3}).swap_dims({'x':'longitude','y':'latitude'})[:,::-1].transpose()

def p_upslope_region(MR,topo='ERA5',version='IVT'):
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    m_per_degreelat = 6370*1e3*np.pi/180
    rho0=1.2 # surface air density in kg/m^3
    Hm=2500. # moisture scale height in m
    if version=='surf':
        p = 86400*rho0 * MR.vars['Q_SFC'] * (MR.vars['VAR_100U']*z.differentiate('longitude')/m_per_degreelat
                                        +MR.vars['VAR_100V']*z.differentiate('latitude')/m_per_degreelat)
    elif version=='IVT':
        p = 86400* (MR.vars['VIWVE']*z.differentiate('longitude')/m_per_degreelat
                   +MR.vars['VIWVN']*z.differentiate('latitude')/m_per_degreelat) / Hm 
    
    return np.maximum(p,0.)