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

#from spatial_ratio import script_nfw_ratio_with_error

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
                         dndv_err=dndv_err,
                         rbins_rvf=rbins_rvf,
                         host_c=c,
                         host_rv=rv
                        )
def get_han_model_gout(gout, fit, rbins_rvf, nfilter)->LinearRegression:
    c, rv = nodedata(gout, key=["concentration", ParamKeys.rvir], nfilter=nfilter_most_massive_progenitor, summarize=True)

    out = get_dndv(gout, bins=rbins_rvf, rvfraction=True, nfilter=nfilter, summarize=True, statfuncs=(np.mean, np.std))
    dndv, dndv_err = out[0][0], out[1][0]
    
    return get_han_model(fit, rbins_rvf, c, rv)


def fit_han_model(dndv, dndv_err, rbins_rvf, host_c, host_rv)->LinearRegression: 
    rvf_binavg = bin_avg(rbins_rvf)
    rrange = rvf_binavg * host_rv
    rs = host_rv / host_c

    # Normalize the nfw profile to be 1 at r = rv
    nfw_norm = profile_nfw(rrange, rs, 1) / profile_nfw(rrange[-1], rs, 1)
    dndv_ratio = dndv / nfw_norm 
 
    fitX = np.log10(rvf_binavg).reshape(-1,1)
    fitY = np.log10(dndv_ratio)

    return LinearRegression().fit(fitX, fitY)

def get_han_model(fit:LinearRegression, rbins_rvf, host_c, host_rv):
    rvf_binavg = bin_avg(rbins_rvf)
    rrange = rvf_binavg * host_rv
    rs = host_rv / host_c 
    print(fit.coef_)

    t = fit.predict(np.log10(rvf_binavg.reshape(-1,1)))

    nfw_norm = profile_nfw(rrange, rs, 1) / profile_nfw(rrange[-1], rs, 1)

    _out = nfw_norm * 10**t

    return _out 