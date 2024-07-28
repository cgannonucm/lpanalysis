#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any
import h5py

from gstk.hdf5util import read_dataset
from gstk.common.constants import GParam, CPhys
from gstk.io import io_importgalout
from gstk.scripts.spatial import  script_dndv_with_error, script_dndv
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus
from gstk.scripts.common import script, ScriptProperties
from gstk.scripts.meta import script_eachtree
from han_model import get_profile_nfw_normalized, get_profile_han, fit_profile_han
from conversion import cfactor_shmf
from gstk.util import util_binedgeavg
from scipy import integrate


from plotting_util import *
from symutil import symphony_to_galacticus_dict

@script(**ScriptProperties.KEY_KWARG_DEF)
def script_spatial_3d_cum(data, rrange = None, rbincount = None, mrange = None, 
                          rvfraction = False, selector_function = None, normalize = False,  **kwargs):
    if rvfraction:
        rv = np.mean(script_select_nodedata(data,script_selector_halos, [GParam.RVIR])[0])
        rrange = np.asarray(rrange) * rv

    selector_function = script_selector_subhalos_valid if selector_function is None else selector_function

    dndv, dndv_r = script_eachtree(data,script_dndv,selector_function=script_selector_subhalos_valid, 
                    rrange=rrange, rbincount=rbincount, mrange=mrange, **kwargs)   

    dndv, dndv_r = np.asarray(dndv), np.asarray(dndv_r)    

    ncum = integrate.cumulative_trapezoid(4 * np.pi * dndv * dndv_r**2, x=dndv_r, axis=1)

    if normalize:
        ncum = ncum / ncum[:,-1].reshape(-1,1)

    if rvfraction:
        outr = dndv_r / rv

    return ncum, outr


@script(**ScriptProperties.KEY_KWARG_DEF)
def script_spatial_3d_cum_meanstd(data, rrange = None, rbincount = None, mrange = None, 
                          rvfraction = False, selector_function = None, normalize = False,  **kwargs):
    ncum, r = script_spatial_3d_cum(data, rrange, rbincount, mrange=mrange, 
                                    rvfraction=rvfraction, selector_function=selector_function,
                                    normalize=normalize, **kwargs) 

    cum, outr, cum_std = np.mean(ncum, axis=0), util_binedgeavg(np.mean(r, axis=0)), np.std(ncum, axis=0)

    return cum, outr, cum_std

def plot_spatial_cum(fig, ax:Axes, data, rrange, rbincount, mrange, 
                        rvfraction, normalize = False, shadeplot = False,
                        kwargs_script = None, kwargs_plot = None, 
                        kwargs_shade = None, nsigma = 1):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_shade = {} if kwargs_shade  is None else kwargs_shade

    cum, r, cum_std = script_spatial_3d_cum_meanstd(data, rrange, rbincount, mrange, 
                                            rvfraction, normalize=normalize, **kwargs_script)
    
    plt_std = nsigma * cum_std
    if shadeplot:
        ax.plot(r, cum, **(KWARGS_DEF_PLOT | kwargs_plot))
        ax.fill_between(r, cum + plt_std, cum - plt_std, **(KWARGS_DEF_FILL | kwargs_shade))
        return

    ax.errorbar(r, cum, cum_std, **(KWARGS_DEF_ERR | kwargs_plot))    

def plot_spatial_cum_ratio(fig, ax:Axes, data_1, data_2, rrange, rbincount, mrange, 
                            rvfraction, normalize = False, shadeplot = False,
                            kwargs_script = None, kwargs_plot = None, kwargs_shade = None,
                            nsigma = 1.0):

    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_shade = {} if kwargs_shade  is None else kwargs_shade

    ncum_1, r_1 = script_spatial_3d_cum(data_1,rrange=rrange,rbincount=rbincount,
                                        mrange=mrange, rvfraction=rvfraction,
                                        normalize=normalize,**kwargs_script)

    ncum_2, r_2 = script_spatial_3d_cum(data_2,rrange=rrange,rbincount=rbincount,
                                        mrange=mrange, rvfraction=rvfraction,
                                        normalize=normalize,**kwargs_script)

    ratio = ncum_1 / np.mean(ncum_2, axis=0).reshape(1,-1)
    r, y, y_std = util_binedgeavg(np.mean(r_1,axis=0)), np.mean(ratio, axis=0), np.std(ratio, axis=0)

    plt_std  = nsigma * y_std

    if shadeplot:
        ax.plot(r, y, **(KWARGS_DEF_PLOT | kwargs_plot))
        ax.fill_between(r, y + plt_std, y - plt_std, **(KWARGS_DEF_FILL | kwargs_shade))
        return

    ax.errorbar(r, y, plt_std, **(KWARGS_DEF_ERR | kwargs_plot))    



def main(): 
    figname = "spatial_3d_cum"
    path_symphony = "data/symphony/SymphonyGroup/"
    path_file = "/home/charles/research/lpanalysis/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)
    rbins = 25
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E10)
    mrange_rescale = (1E9, 1E10)

    filend = io_importgalout(path_file)[path_file] 
    # isnap 203 is snapshot at z~0.5
    sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

    set_plot_defaults()

    fig, axs = plt.subplots(ncols=2, figsize=(18,6))
    ax0, ax1 = axs

    # ax 0

    plot_spatial_cum(fig, ax0, filend, rrange_rvf, rbins, mrange, 
                        rvfraction=True, normalize=True,shadeplot=True,
                        kwargs_script=dict(logscaling=False), 
                        kwargs_plot=dict(color="black"))
    

    plot_spatial_cum(fig, ax0, filend, rrange_rvf, rbins, mrange, 
                        rvfraction=True, normalize=True,shadeplot=True,
                        kwargs_script=dict(logscaling=False), 
                        kwargs_plot=dict(color="black"), nsigma=2) 

    plot_spatial_cum(fig, ax0, sym_nodedata, rrange_rvf, rbins, mrange_sym, 
                        rvfraction=True, normalize=True,shadeplot=False,
                        kwargs_script=dict(logscaling=False))

    ax0.set_xlim(0.1,1)
    ax0.set_ylim(0,1.05)
    
    ax0.set_xlabel(r"$r / r_{v, host}$")
    ax0.set_ylabel(r"$N( < r/r_{v,host}) / N(r/r_{v,host} < 1)$")

    #ax 2
    ax1:Axes = ax1
    plot_spatial_cum_ratio(fig,ax1, filend, sym_nodedata, rrange_rvf,
                            50, mrange, rvfraction=True, normalize=True, 
                            shadeplot=True, kwargs_script=dict(logscaling=False),
                            kwargs_plot=dict(color="black"))

    plot_spatial_cum_ratio(fig,ax1, filend, sym_nodedata, rrange_rvf,
                            50, mrange, rvfraction=True, normalize=True, 
                            shadeplot=True, kwargs_script=dict(logscaling=False),
                            kwargs_plot=dict(color="black"), nsigma=2)

    ax1.hlines(1.0, 0.1, 1.0, color="tab:blue", linestyles="dashed", **KWARGS_DEF_PLOT)

    ax1.set_xlabel(r"$r / r_{v, host}$")
    ax1.set_ylabel(r"$N( < r/r_{v,host})_{ratio}$")
    
    ax1.set_xlim(0.1,1)

    savefig(fig,figname + ".pdf")
    savefig(fig,figname + ".png")
    


if __name__ == "__main__":
    main()
