import os
import glob
import numpy as np
import xarray as xr
from scipy.integrate import cumtrapz
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import sys
p = os.path.abspath('/global/homes/q/qnicolas/')
if p not in sys.path:
    sys.path.append(p)
    
from orographicConvectionTheory.orographicConvectionTheory import k_vector,m_exponent

###################################################################################################
############################################ UTILS ################################################
###################################################################################################

def z_to_p_standard(z):
    "z in m, returns pressure in hPa corresponding to an atmosphere w/ Tsfc=300K and lapse rate = 6.5K/km"
    p0=1000.; T0=300.; Gamma = 6.5e-3; R=287.;g=9.81
    return p0*np.exp(g/R/Gamma*np.log(1-Gamma*z/T0))

def p_to_z_standard(p):
    p0=1000.; T0=300.; Gamma = 6.5e-3; R=287.;g=9.81
    return T0/Gamma*(1-np.exp(R*Gamma/g*np.log(p/p0)))


def w_to_zeta(w,Uz):
    """From w on a x-z grid and U on a z-grid, return the linear streamline vertical displacement zeta
    If U is a scalar, assume it's constant through the column"""
    if not hasattr(Uz, "__len__"): #check if Uz is a scalar
        Uz = Uz*np.ones(len(w.altitude))
    return xr.DataArray(cumtrapz(w,w.distance_from_mtn,axis=0,initial=0)/Uz[None,:],
                        coords=w.coords,
                        dims=w.dims
                       )
def w_to_Tprime(w,Uz,Nz):
    if not hasattr(Nz, "__len__"): #check if Nz is a scalar
        Nz = Nz*np.ones(len(w.altitude))
    if not hasattr(Uz, "__len__"): #check if Uz is a scalar
        Uz = Uz*np.ones(len(w.altitude))
        
    cp = 1004.;g=9.81
    ds0dz = cp*300/g*Nz**2
    ds0dz = ds0dz*w.altitude**0 #convert to xarray for broadcasting
    
    return  - ds0dz * w_to_zeta(w,Uz)

def w_to_qprime(w,Uz,dq0dz = 8.1/2.5e3):
    """dqdz in kg/kg/m"""
    if not hasattr(Uz, "__len__"): #check if Uz is a scalar
        Uz = Uz*np.ones(len(w.altitude))
        
    dq0dz = dq0dz*w.altitude**0 #convert to xarray for broadcasting
    return  dq0dz * w_to_zeta(w,Uz)


###################################################################################################
################################ LINEAR 1-LAYER & 2-LAYER MODELS ##################################
###################################################################################################

def linear_w_1layer(xx,hx,z,U,N):
    k=k_vector(len(xx),xx[1]-xx[0])
    h_hat = np.fft.fft(hx)
    
    w_hat = 1j*k[:,None]*U*h_hat[:,None]*np.exp( m_exponent(k[:,None],N,U)  *  z[None,:]) 
    w=xr.DataArray(np.real(np.fft.ifft(w_hat,axis=0)),coords={'distance_from_mtn':xx/1000,'altitude':z/1000},dims=['distance_from_mtn','altitude']).assign_coords({'pressure':('altitude',z_to_p_standard(z))})
    return w

def linear_w_2layer(xx,hx,z,H,Ul,Uu,Nl,Nu):
    kk=k_vector(len(xx),xx[1]-xx[0])
    h_hat = np.fft.fft(hx)
    w_hat =np.zeros((len(kk),len(z)))*1j
    
    for i,k in enumerate(kk):
        ml = np.sqrt(Nl**2/Ul**2 - k**2 +0.j)
        mu = np.sqrt(Nu**2/Uu**2 - k**2 +0.j)
        if k<0 and k**2 < Nu**2/Uu**2:
            mu = -np.sqrt(Nu**2/Uu**2 - k**2)
        R = (ml-mu)/(ml+mu)
        Al = 1j*k*Ul*h_hat[i]/(np.exp(-1j*ml*H)+R*np.exp(1j*ml*H))
        Au = Al*(1+R)
        w_hat[i,z<=H] = Al*(np.exp(1j*ml*(z[z<=H]-H))+R*np.exp(-1j*ml*(z[z<=H]-H)))
        w_hat[i,z>H] = Au*np.exp(1j*mu*(z[z>H]-H))
        
    w=xr.DataArray(np.real(np.fft.ifft(w_hat,axis=0)),coords={'distance_from_mtn':xx/1000,'altitude':z/1000},dims=['distance_from_mtn','altitude']).assign_coords({'pressure':('altitude',z_to_p_standard(z))})
    return w

###################################################################################################
####################### LINEAR MODEL, ARBITRARY U(z) AND N(z) PROFILES ############################
###################################################################################################

def second_derivative_matrix(n,dz):
    """Second-order second derivative matrix on a uniform grid"""
    mat = -2*np.eye(n)
    for k in range(n-1):
        mat[k,k+1] = 1
        mat[k+1,k] = 1
    mat[0]=np.zeros(n);mat[-1] =np.zeros(n)
    mat[0,:4] = [2,-5,4,-1]
    mat[-1,-4:] = [-1,4,-5,2]
    return mat/dz**2

def gw_mode(z,lz2,k,hhatk,U0):
    """Computes one wave mode by solving the linear wave equation:
    d2/dz2(w_hat) + (l(z)^2-k^2)w_hat = 0, subject to BCs
    w_hat(k,z=0) = ikU(z=0)h_hat(k) 
    & d w_hat(k,ztop) = i m(ztop) w_hat(k,ztop), where m(ztop) is defined to satisfy a radiation BC or an evanescent BC at the top
    """
    n = len(z)
    dz = z[1]-z[0]
    
    sgnk = np.sign(k)
    if k==0:
        sgnk=1
    if lz2[-1] < k**2:
        mtop = 1j*np.sqrt(k**2-lz2[-1])
    else:
        mtop = sgnk*np.sqrt(lz2[-1]-k**2)
    
    D2 = second_derivative_matrix(n,1) #Matrix of second differentiation
    A = D2 + dz**2*np.diag(lz2-k**2)
    A = A.astype('complex')
    
    A[0]   = np.zeros(n)
    A[0,0] = 1
    A[-1]  = np.zeros(n)
    #A[-1,-1] = 1;A[-1,-2] = -1
    A[-1,-3:] = np.array([1,-4,3])/2
    A[-1,-1] -= dz * 1j* mtop
    
    b = 1j*np.zeros(n)
    b[0] = 1j*k*U0*hhatk
    
    return np.linalg.solve(A,b)
    A = csc_matrix(A)
    return spsolve(A,b)
    
def linear_w_generalized(xx,hx,z,Uz,Nz):
    "z must be evenly spaced"
    kk=k_vector(len(xx),xx[1]-xx[0])
    h_hat = np.fft.fft(hx)
    lz2 = Nz**2/Uz**2 - 1/Uz * np.dot(second_derivative_matrix(len(z),z[1]-z[0]),Uz)
    
    w_hat =np.zeros((len(kk),len(z)))*1j
    for i,k in enumerate(kk):
        if i%500==0:
            print(i,end=' ')
        w_hat[i] = gw_mode(z,lz2,k,h_hat[i],Uz[0])
        
    w=xr.DataArray(np.real(np.fft.ifft(w_hat,axis=0)),coords={'distance_from_mtn':xx/1000,'altitude':z/1000},dims=['distance_from_mtn','altitude']).assign_coords({'pressure':('altitude',z_to_p_standard(z))})
    return w

