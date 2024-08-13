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
from spatial_3d_cum import plot_spatial_cum, plot_spatial_cum_ratio


def main(): 
    figname = "spatial_3d_cum_cg"

    path_symphony = "data/symphony/SymphonyGroup/"
    path_file = "/home/charles/research/lpanalysis/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    path_cg1 = "data/galacticus/xiaolong_update/cg-2/lsubmodv3.1-cg-date-07.11.2024-time-00.55.15-date-07.11.2024-time-00.55.16-z-5.00000E-01-mh-1.00000E+13-mstar-1.00000E+12-dmstar-0.00000E+00-drstar-0.00000E+00.xml.hdf5"

    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)
    rbins = 25
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E10)
    mrange_rescale = (1E9, 1E10)

    filend = io_importgalout(path_file)[path_file] 
    filecg = io_importgalout(path_cg1)[path_cg1] 
    # isnap 203 is snapshot at z~0.5
    #sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

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

    plot_spatial_cum(fig, ax0, filecg, rrange_rvf, rbins, mrange_sym, 
                        rvfraction=True, normalize=True,shadeplot=False,
                        kwargs_script=dict(logscaling=False))

    ax0.set_xlim(0.1,1)
    ax0.set_ylim(0,1.05)
    
    ax0.set_xlabel(r"$r / r_{v, host}$")
    ax0.set_ylabel(r"$N( < r/r_{v,host}) / N(r/r_{v,host} < 1)$")

    #ax 2
    ax1:Axes = ax1
    plot_spatial_cum_ratio(fig,ax1, filecg, filend, rrange_rvf,
                            50, mrange, rvfraction=True, normalize=True, 
                            shadeplot=True, kwargs_script=dict(logscaling=False),
                            kwargs_plot=dict(color="black"))

    plot_spatial_cum_ratio(fig,ax1, filecg, filend, rrange_rvf,
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
