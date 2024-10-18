#!/usr/bin/env python

import h5py
import numpy as np
from scipy import integrate
from colossus.halo import concentration, mass_so
from colossus.cosmology import cosmology
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from typing import Callable
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from scipy.optimize import curve_fit
from matplotlib import font_manager
from matplotlib.axes import Axes
from typing import Any
import os.path as path
from astropy.cosmology import FlatLambdaCDM
from typing import Iterable
import pandas as pd
from colossus.lss import mass_function
from numpy.dtypes import StrDType

from plotting_util import *
from summary import HDF5Wrapper


def scaling_fit_mhz(data, key_mh, key_z, key_tofit, scale = 1, mscale = 1E13, zshift = 0.5,filter=None):
    filter = np.ones(data[key_mh].shape, dtype=bool) if filter is None else filter
    mh,z,tofit = data[key_mh][filter], data[key_z][filter], data[key_tofit][filter] * scale
    fitX = np.log10(np.transpose(np.array((mh / mscale,z + zshift))))
    return LinearRegression().fit(fitX,np.log10(tofit)) 

def scaling_fit_mhz_def(data, rannulus = None, key_n_proj = None, key_mass = None, key_z = None, filter=None):
    # Hard coded values for .csv currently used
    key_n_proj = PARAM_KEY_N_PROJ_BOUND if key_n_proj is None else key_n_proj
    key_mass = KEY_DEF_HALOMASS if key_mass is None else key_mass
    key_z = KEY_DEF_Z if key_z is None else  key_z
    rannulus = 1E-2,2E-2 if rannulus is None else rannulus

    r0, r1 = rannulus 
    area = np.pi * (r1**2 - r0**2)

    return scaling_fit_mhz(data, key_mass, key_z, key_n_proj, 1/area, filter=filter)
 
def fits_to_df(fits:dict[str,LinearRegression])->pd.DataFrame:
    l = len(fits)

    out_dict = {}

    for n, coef in enumerate(fits[fits.__iter__().__next__()].coef_):
        out_dict[f"k_{n}"] = np.zeros(l)

    out_dict["x_0"] = np.zeros(l)
    out_dict["label"] = ["" for i in range(l)]
    
    for n,(label, fit) in enumerate(fits.items()): 
        for i, coef in enumerate(fit.coef_):
            out_dict[f"k_{i}"][n] = coef 

        out_dict["x_0"][n] = fit.intercept_
        out_dict["label"][n] = label

    
    return pd.DataFrame(out_dict)

def nproj_han_model(rho:float, c:float, rv:float, mh:float, z:float, gamma:float=1, scale = 1, 
                    fstripped:Callable[[np.ndarray] , np.ndarray] = None, ymin = 0, ymax = 1,
                    quadkwargs_proj = {}, quadkwargs_norm = {}, ymax_account_curvature=True):
    """
    Projects the Han et al (2016) model. 
    fstripped: The fraction of tidal stripping compared to an nfw profile as function of (r / r3d) 
    Defaults to (r/r_v3)^\gamma  if is None
    """
    if fstripped is None:
        fstripped = lambda r: r**gamma

    rho_r = rho / rv
    if ymax_account_curvature:
        ymax = np.sqrt(1 - rho_r**2)

    nfw = lambda x: scale/x/(1+x)**2
    integrand_proj = lambda y: fstripped(np.sqrt(y**2 + rho_r**2)) * nfw(c * np.sqrt(y**2 + rho_r**2))
    proj = integrate.quad(integrand_proj,ymin,ymax, **quadkwargs_proj)[0]
    
    # Impose the condition that the integral of the spatial profile is proportional to the mass
    integrand_mass = lambda y: y**2 * fstripped(y) * nfw(c*y)
    vol =  integrate.quad(integrand_mass,0, 1, **quadkwargs_norm)[0]
    coef = mh / np.pi / rv**2
    #coef = mh**(1/3) * (cosmo.Hz(z))**(4/3)

    return coef * proj / vol

def nproj_han_model_mhz(rho:float, mh:float, z:float, cosmo, gamma:float=1, scale = 1,
                        fstripped:Callable[[np.ndarray] , np.ndarray] = None, **int_kwargs):
    c = concentration.concentration(mh * cosmo.h, 'vir', z, model = 'diemer19') / cosmo.h
    rv = mass_so.M_to_R(mh * cosmo.h,z,"vir") * 1E-3 / cosmo.h
    return nproj_han_model(rho, c, rv, mh, z, gamma, scale, fstripped, **int_kwargs)



def scaling_han_model(rannulus:tuple[float,float], mh_space:np.ndarray, z_space:np.ndarray, cosmo,
                        gamma:(float | tuple) =1, fstripped:Callable[[np.ndarray] , np.ndarray] = None,
                        mscale = 1E13, zshift=0.5, int_kwargs = {}, navg=5, rv_fraction=False, msub=1E8):
    rho0,rho1 = rannulus
    mh_mg, z_mg = np.meshgrid(mh_space, z_space)
    mh_mgf, z_mgf = mh_mg.flatten(), z_mg.flatten()
    Fscale = np.zeros(mh_mgf.shape) 

    rhospace = np.linspace(rho0,rho1, navg)
         
    if isinstance(gamma,Iterable): 
        gamma0, gamma1 = gamma
        m0, m1 = np.min(mh_space), np.max(mh_space)
        #m0l,m1l = np.log10(m0), np.log10(m1)
        # Linearly interpolate gamma if givven two values
        get_gamma = lambda m: ((gamma1 - gamma0) / (m1 - m0)) * (m - m0) + gamma0
    else:
        get_gamma = lambda m: gamma
        gammaspace = gamma * np.ones(navg)

    def average_model(mh,z, rhospace):
        dnda_rho = np.zeros(rhospace.shape)

        if rv_fraction:
            rv = mass_so.M_to_R(mh * cosmo.h,z,"vir") * 1E-3 / cosmo.h
            rhospace = np.linspace(rho0 * rv, rho1 * rv, navg)

        g = get_gamma(mh)
        #if isinstance(gamma,Iterable):
        #    print(np.log10(mh),g)
        for n,rho in enumerate(rhospace):
            dnda_rho[n] = nproj_han_model_mhz(rho,mh,z,cosmo,g,1, fstripped, **int_kwargs) 
        #scale by fraction of mass in halos a in the universe
        haloabundance = mass_function.massFunction(1E8 * cosmo.h, z, mdef="fof", model="sheth99", q_out="M2dndM")
        
        return np.mean(dnda_rho) * haloabundance

    for n, (mh, z) in enumerate(zip(mh_mgf, z_mgf)):
        Fscale[n] = average_model(mh,z, rhospace)  

    g0 = get_gamma(mscale)
    norm = 1 / average_model(mscale,zshift, rhospace)

    Fscale = Fscale.reshape(mh_mg.shape)
    Fscale *= norm

    return mh_mg, z_mg, Fscale

def scaling_fit_han_model(rannulus, mh_space:np.ndarray, z_space:np.ndarray, cosmo, gamma:float=1,
                            fstripped:Callable[[np.ndarray] , np.ndarray] = None, mscale = 1E13, zshift=0.5,
                            int_kwargs = {}, navg=5, rv_fraction=False, norm:float = 1):
    mh_mg, z_mg, Fscale = scaling_han_model(rannulus, mh_space, z_space, cosmo, gamma, fstripped, mscale, zshift, int_kwargs, navg, rv_fraction)

    KEY_SCALE = "scale"
    scaling_dict = {
                        KEY_DEF_HALOMASS: mh_mg.flatten(),
                        KEY_DEF_Z: z_mg.flatten(),
                        KEY_SCALE: Fscale.flatten()
    }

    return scaling_fit_mhz(scaling_dict, KEY_DEF_HALOMASS, KEY_DEF_Z, KEY_SCALE, norm,mscale,zshift)

def nfw_mproj(m_halo,z,r_ap, cosmo):
    rvir = mass_so.M_to_R(m_halo * cosmo.h,z,"vir") / 1E3 / cosmo.h
    c = concentration.concentration(m_halo * cosmo.h, "vir", z, model = 'diemer19')
    r = r_ap / rvir

    g = 1 / (np.log(1+c) - c/(1+c))

    def C_inv(x):
        if x < 1:
            return np.arccos(x)
        if x > 1:
            return np.arccosh(x)
        return 0

    return g * m_halo * (
        +  C_inv(1/c/r) 
            / np.abs(
                + (c*r)**2
                - 1
              )**(1/2)
        +  np.log(c*r/2) 
    ) 

def nfw_scale_annulus(rannulus, m_halo,z, cosmo, alpha=PARAM_DEF_ALPHA):
    return (nfw_mproj(m_halo,z,rannulus[1], cosmo) - nfw_mproj(m_halo,z,rannulus[0], cosmo)) * m_halo**(-alpha-2)

def scaling_nfw(rannulus, mh_space:np.ndarray, z_space:np.ndarray, cosmo,mscale=1E13,zshift=0.5, alpha=PARAM_DEF_ALPHA):
    mh_mg, z_mg = np.meshgrid(mh_space, z_space)
    mh_flat,z_flat = mh_mg.flatten(), z_mg.flatten()

    fscale = np.zeros(mh_flat.shape) 

    fscale_0 = nfw_scale_annulus(rannulus,mscale,zshift,cosmo,alpha)
    
    for n, (m,z) in enumerate(zip(mh_flat,z_flat)):
        fscale[n] = nfw_scale_annulus(rannulus, m, z, cosmo,alpha) / fscale_0

    return mh_mg, z_mg, fscale.reshape(mh_mg.shape)

def scaling_nfw_dict(rannulus, mh_space:np.ndarray, z_space:np.ndarray, cosmo,mscale=1E13,zshift=0.5, alpha=PARAM_DEF_ALPHA):
    mh_mg, z_mg, fscale = scaling_nfw(rannulus, mh_space, z_space, cosmo, mscale,zshift,alpha)
    return { 
                KEY_DEF_HALOMASS:       mh_mg.flatten(),
                KEY_DEF_Z:              z_mg.flatten(),  
                PARAM_KEY_N_PROJ_INFALL:fscale.flatten()
    }


def main():
    fname = "scaling.csv"
    path_csv = "data/output/analysis_scaling_nd_annulus_new.csv"
    path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"

    path_file_hdf5_um = "out/hdf5/scaling_um.hdf5"
    path_file_hdf5 = "out/hdf5/summary_scaling.hdf5"
    
    hdf5_scaling_um = h5py.File(path_file_hdf5_um)
    hdf5_scaling    = h5py.File(path_file_hdf5)
    
    scaling_data_hdf5_um = HDF5Wrapper(hdf5_scaling_um)
    scaling_data_hdf5 = HDF5Wrapper(hdf5_scaling)
    #scaling_data    = 
    
    key_n_proj_h = "n evolved 0.01 < r_{2d} <= 0.02, 1.00e+08 < m_e < 1.00e+10 (mean)/out0"
    key_n_proj_infall_h = "n unevolved 0.01 < r_{2d} <= 0.02, 1.00e+08 < m_e < 1.00e+10 (mean)/out0" 
    key_z_h = "z (mean)/out0"
    key_mh_h = "halo mass (mean)/out0"

    key_n_proj = PARAM_KEY_N_PROJ_BOUND
    key_n_proj_infall = PARAM_KEY_N_PROJ_INFALL

    scaling_data = pd.read_csv(path_csv)

    mh, z = scaling_data[KEY_DEF_HALOMASS], scaling_data[KEY_DEF_Z]
    mh_range = (np.min(mh), np.max(mh))
    z_range = (np.min(z), np.max(z))

    mspace = np.geomspace(*mh_range,5)

    zspace = np.linspace(*z_range,5)
    rannulus = PARAM_DEF_RANNULUS

    cosmo = cosmology.setCosmology("planck18")

    filend = h5py.File(path_file)

    interp_rrange_rvf = PARAM_DEF_RRANGE_RVF
    interp_rbins = 10
    interp_mrange = PARAM_DEF_MRANGE

    #interp_t = galacticus_interp_tstripped(filend, interp_rrange_rvf, interp_rbins, interp_mrange)
    
    han_gamma = np.linspace(0.8, 2, 13)

    nfw_dict = scaling_nfw_dict(rannulus,mspace,zspace,cosmo)
    
    _filter_m = scaling_data_hdf5[key_mh_h] > 1E13

    fits = {
                "Galacticus (Unevolved)"                :scaling_fit_mhz_def(scaling_data, key_n_proj=key_n_proj_infall),   
                "Galacticus (Evolved)"                  :scaling_fit_mhz_def(scaling_data, key_n_proj=key_n_proj),
                "Galacticus (UM, Evolved)"              :scaling_fit_mhz_def(scaling_data_hdf5_um, key_n_proj=key_n_proj_h, key_mass=key_mh_h, key_z=key_z_h),   
                "Galacticus (Evolved) (new)"            :scaling_fit_mhz_def(scaling_data_hdf5, key_n_proj=key_n_proj_h, key_mass=key_mh_h, key_z=key_z_h),   
                "Galacticus (Evolved) (new) m_h > 1E13" :scaling_fit_mhz_def(scaling_data_hdf5, key_n_proj=key_n_proj_h, key_mass=key_mh_h, key_z=key_z_h, filter=_filter_m),   
                "Han (2016) (Gamma Interp)"             :scaling_fit_han_model(rannulus,mspace,zspace,cosmo,gamma=(0.94,1.24)),
                #"Han (2016) (Galacticus Interp)"        :scaling_fit_han_model(rannulus,mspace,zspace,cosmo,fstripped=interp_t),
                "NFW"                                   :scaling_fit_mhz(nfw_dict,KEY_DEF_HALOMASS,KEY_DEF_Z,PARAM_KEY_N_PROJ_INFALL)
        } 
    
    for gamma in han_gamma:
                label = f"Han et al. (Gamma = {gamma:.1f})" 
                fits[label] = scaling_fit_han_model(rannulus,mspace,zspace,cosmo,gamma=gamma)
 
    df = fits_to_df(fits)

    savedf(df,fname)

if __name__ == "__main__":
    main()
