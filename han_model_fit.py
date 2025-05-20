#!/usr/bin/env python

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
import h5py
import os.path as path


from subscript.defaults import ParamKeys
from subscript.wrappers import gscript
from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import nfilter_most_massive_progenitor
from subscript.scripts.histograms import spatial3d_dndv, bin_avg

from plotting_util import savedf, savefig, set_plot_defaults
import pandas as pd

#from spatial_ratio import script_nfw_ratio_with_error

def select_index_closest_mh_z(mh, z, select_mh, select_z):
    vec = np.asarray(((mh - select_mh) / np.sqrt(np.mean(mh)**2), (z - select_z) / (np.sqrt(np.mean(z)**2))))
    return np.argmin(np.linalg.norm(vec, axis=0))


@gscript
def get_dndv(gout, bins=None, range=None, rvfraction=False,  **kwargs):
    _, _bins = np.histogram((1, ), bins=bins, range=range)
    
    rv = nodedata(
                  gout, 
                  ParamKeys.rvir, 
                  nfilter=nfilter_most_massive_progenitor,
                  summarize=True
                 ) 

    scale_y = rv if rvfraction else 1

    radii = _bins * scale_y        
    dndv, _ = spatial3d_dndv(gout, bins=radii, **kwargs)
    
    if rvfraction:
        return dndv, radii / rv

    return dndv, radii

def profile_nfw(r, rs, p0):
    x = r / rs
    return p0 / x / (1 + x)**2

def fit_han_model_gout(gout, rbins_rvf, nfilter)->LinearRegression:
    c, rv = nodedata(gout, key=["concentration", ParamKeys.rvir], nfilter=nfilter_most_massive_progenitor, summarize=True)

    out = get_dndv(gout, bins=rbins_rvf, rvfraction=True, nfilter=nfilter, summarize=True, statfuncs=(np.mean, np.std))
    dndv, dndv_err = out[0][0], out[1][0]
    
    return fit_han_model(
                         dndv=dndv, 
                         rbins_rvf=rbins_rvf,
                         host_c=c,
                         host_rv=rv
                        )
def get_han_model_gout(gout, fit, rbins_rvf, nfilter)->LinearRegression:
    c, rv = nodedata(gout, key=["concentration", ParamKeys.rvir], nfilter=nfilter_most_massive_progenitor, summarize=True)

    out = get_dndv(gout, bins=rbins_rvf, rvfraction=True, nfilter=nfilter, summarize=True, statfuncs=(np.mean, np.std))
    dndv, dndv_err = out[0][0], out[1][0]
    
    return get_han_model(fit, rbins_rvf, c, rv)


def ratio_halo_density(dndv, rbins_rvf, host_c, host_rv):
    rvf_binavg = bin_avg(rbins_rvf)
    rrange = rvf_binavg * host_rv
    rs = host_rv / host_c

    # Normalize the nfw profile to be 1 at r = rv
    nfw_norm = profile_nfw(rrange, rs, 1) / profile_nfw(rrange[-1], rs, 1)
    return dndv / nfw_norm / dndv[-1]
 


def fit_han_model(dndv, rbins_rvf, host_c, host_rv)->LinearRegression: 
    rvf_binavg = bin_avg(rbins_rvf)
    dndv_ratio = ratio_halo_density(dndv, rbins_rvf, host_c, host_rv)

    fitX = np.log10(rvf_binavg).reshape(-1,1)
    fitY = np.log10(dndv_ratio)
     
    #weight = dndv_ratio * (rbins_rvf[1:]**2 - rbins_rvf[:-1]**2)
    weight = np.ones(dndv_ratio.shape)

    return LinearRegression().fit(fitX, fitY, sample_weight=weight)

def get_han_model(fit:LinearRegression, rbins_rvf, host_c, host_rv):
    rvf_binavg = bin_avg(rbins_rvf)
    rrange = rvf_binavg * host_rv
    rs = host_rv / host_c 
    print(fit.coef_)

    t = fit.predict(np.log10(rvf_binavg.reshape(-1,1)))

    nfw_norm = profile_nfw(rrange, rs, 1) / profile_nfw(rrange[-1], rs, 1)

    _out = nfw_norm * 10**t

    return _out 

def main():
    fname = "han_model_fit"
    path_summary_scaling = "out/hdf5/summary_scaling.hdf5"
    scalingsum = h5py.File(path_summary_scaling)

    set_plot_defaults()

    hm_select = np.asarray((10**12, 10**13.5))
    z_select = np.asarray((0.2, 0.8)) 
    mhg, zg = np.meshgrid(hm_select, z_select) 

    mhg = np.append(mhg.flatten(), 1E13)
    zg  = np.append(zg.flatten(), 0.5)

    kfit = np.zeros((mhg.flatten().shape[0]))
    fits = []

    #dndv_group_evo = scalingsum["dNdV (evolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (mean)"]
    #dndv_group_evo_std = scalingsum["dNdV (evolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (std)"]

    dndv_group_evo = scalingsum["dNdV (evolved) [MPC^${-3}$] 3.16E+08 < m <= 1.00E+09 (mean)"]
    dndv_group_evo_std = scalingsum["dNdV (evolved) [MPC^${-3}$] 3.16E+08 < m <= 1.00E+09 (std)"]

    halo_mass = scalingsum["halo mass (mean)/out0"]
    halo_z = scalingsum["z (mean)/out0"]

    host_c, host_rv = scalingsum["concentration (host) (mean)/out0"], scalingsum["rvir (host) (mean)/out0"]

    fig, ax = plt.subplots(figsize=(9,6))

    for i, (hm_s, z_s) in enumerate(zip(mhg.flatten(), zg.flatten())): 
        n = select_index_closest_mh_z(halo_mass, halo_z, hm_s,z_s)
        mh, z = halo_mass[n], halo_z[n]
        c, rv = host_c[n], host_rv[n]

        dndv_evo, dndv_evo_rbins_rvf = dndv_group_evo["out0"][:][n], dndv_group_evo["out1"][:][n]
       
        fit_evo = fit_han_model(dndv_evo, dndv_evo_rbins_rvf, c, rv)     

        ratio_density = ratio_halo_density(dndv_evo, dndv_evo_rbins_rvf, c, rv)     

        ax.plot(dndv_evo_rbins_rvf[:-1], ratio_density, label=r"$log_{10} (M_h) = " + str(np.log10(hm_s)) + r", z = $" + str(z_s) + r", $\gamma$ = " + str(fit_evo.coef_[0]))     
        fits.append(fit_evo)
        
        kfit[i] = fit_evo.coef_[0]


    hanfitdf =  {
                    r"\log_{10} (Halo Mass [M_\odot])"  : np.log10(mhg),
                    "z"                     : zg,  
                    r"\gamma"               : kfit                
                } 

    ax.set_prop_cycle(None)

    for f in fits:
        x = np.geomspace(0.1, 1, 100)
        ax.plot(x, 10**f.intercept_ * x ** f.coef_[0], linestyle="dashed")
    
    ax.loglog()
    ax.set_xlabel("r / r$_v$")
    ax.set_ylabel("Ratio Of $dN/dV$ To Host Density")
    ax.legend(fontsize=15)
    
    savedf(pd.DataFrame.from_dict(hanfitdf),fname + ".csv")
    savefig(fig, fname)
    

if __name__ == "__main__":
    main()


