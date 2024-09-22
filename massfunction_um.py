#!/usr/bin/env python
import h5py
import numpy as np
import symlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any
import os

from gstk.common.constants import GParam, CPhys
from gstk.util import util_binedgeavg
from gstk.io import io_importgalout
from gstk.scripts.common import ScriptProperties, script, NodeSelector
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus
from gstk.scripts.massfunction import script_massfunction
from gstk.scripts.meta import script_eachtree
from gstk.util import TabulatedGalacticusData

from subscript.scripts.histograms import massfunction
from subscript.wrappers import freeze
from subscript.scripts.nfilters import nfilter_subhalos_valid, nfilter_project2d, nfilter_halos, nfand
from subscript.tabulatehdf5 import tabulate_trees
from subscript.defaults import ParamKeys
from subscript.scripts.nodes import nodedata

#from subscript.defaults import Meta

#Meta.enable_higher_order_caching = False

from massfunction import symphony_to_galacticus_dict
from plotting_util import set_plot_defaults, savefig, KWARGS_DEF_FILL, KWARGS_DEF_PLOT, KWARGS_DEF_ERR



def plot_massfunction(fig, ax:Axes, gout, mrange, bincount, nfilter=None, 
                                    kwargs_plot=None, kwargs_fill=None, kwargs_script=None, 
                                    plot_dndlnm=False, useratio = True, error_plot=False, 
                                    factor=1, nsigma = 1):

    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    
    nfilter = nfilter_subhalos_valid.freeze(mass_min=mrange[0], mass_max=mrange[1], key_mass=GParam.MASS_BASIC) if nfilter is None else nfilter
    mhost = nodedata(tabulate_trees(gout), ParamKeys.mass_basic, nfilter=nfilter_halos, summarize=True)

    mrange_scale = 1

    if useratio:
        mrange_scale = mhost

    _mrange = np.asarray(mrange) * mrange_scale

    _bins = np.geomspace(*_mrange, num=bincount + 1) 
    scripto = massfunction(gout, bins=_bins, nfilter=nfilter, 
                            summarize=True, statfuncs=[np.mean, np.std], 
                            key_mass=GParam.MASS_BOUND, **kwargs_script)
    (sub_dndm_avg, sub_m), (sub_dndm_std, sub_m_std) = scripto

    plotx = util_binedgeavg(sub_m) / mrange_scale

    ploty, ploty_std = sub_dndm_avg * factor, sub_dndm_std * factor

    if plot_dndlnm:
        ploty       = sub_dndm_avg * util_binedgeavg(sub_m) * factor
        ploty_std   = sub_dndm_std * util_binedgeavg(sub_m) * factor * nsigma

    ploty_min, ploty_max = ploty - ploty_std, ploty + ploty_std

    if error_plot:
        ax.errorbar(plotx, ploty, ploty_std, **(KWARGS_DEF_ERR | kwargs_plot))
        return 

    ax.plot(plotx, ploty, **(KWARGS_DEF_PLOT | kwargs_plot))
    ax.fill_between(plotx, ploty_min, ploty_max, **(KWARGS_DEF_FILL | kwargs_fill))

def plot_massfunction_ratio(fig, ax:Axes, gout_base, gout_compare,  mrange, bincount, nfilter=None, 
                                    kwargs_plot=None, kwargs_fill=None, kwargs_script=None, 
                                    useratio = True, error_plot=False, nsigma = 1):

    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    
    nfilter = nfilter_subhalos_valid.freeze(mass_min=mrange[0], mass_max=mrange[1], key_mass=GParam.MASS_BASIC) if nfilter is None else nfilter

    mhost = nodedata(gout_base, GParam.MASS_BASIC, nfilter=nfilter_halos, summarize=True)

    mrange_scale = 1

    if useratio:
        mrange_scale = mhost

    _mrange = np.asarray(mrange) * mrange_scale

    _bins = np.geomspace(*_mrange, num=bincount) 
    out_base    = massfunction(gout_base, bins=_bins, nfilter=nfilter, 
                                    summarize=True, key_mass=GParam.MASS_BOUND, **kwargs_script)

    out_compare = massfunction(gout_compare, bins=_bins, nfilter=nfilter, 
                                    summarize=True, key_mass=GParam.MASS_BOUND, **kwargs_script)
     
    plotx = util_binedgeavg(_bins) / mrange_scale
    ploty = out_compare[0] / out_base[0]
 
    ax.plot(plotx, ploty, **(KWARGS_DEF_PLOT | kwargs_plot))


def main():
    fname = "massfunction_um"
    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5" 
    path_symphony = "data/symphony/SymphonyGroup/"
    path_file     =  "data/galacticus/um_update/dmo.hdf5" 
    path_um       = "data/galacticus/um_update/umachine.hdf5" 
    path_um_dr0   = "data/galacticus/um_update/umachine-dr0.hdf5" 
    path_um_cg    = "data/galacticus/um_update/cg/cg.hdf5"

    filend      = h5py.File(path_file, rdcc_nbytes=10*1024**2, rdcc_w0=1.0, rdcc_nslots=10000)
    fileum      = h5py.File(path_um, rdcc_nbytes=10*1024**2, rdcc_w0=1.0, rdcc_nslots=10000)  
    
    fileum_dr0  = h5py.File(path_um_dr0)
    fileum_cg   = h5py.File(path_um_cg)


    mres = 1E9

    #script_test(filend)
    
    #sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

    plot_dndlnm = True
    useratio = True

    scale = 1E13

    if useratio:
        scale = 1
    
    mrange = np.asarray((1E8 / 1E13,1)) * scale
    mrange_sym = np.asarray((mres / 10**(13.0), 1)) * scale
    bincount = 20    

    set_plot_defaults()

    fig, axs = plt.subplots(ncols = 2, nrows=2, figsize=(18,12))
    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    nfilter = freeze(nfilter_subhalos_valid, mass_min=1E8, mass_max=1E13, key_mass=GParam.MASS_BOUND)

    plot_massfunction(fig, ax0, filend, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter,
                                    kwargs_plot=dict(zorder=1, color="tab:orange", label="Dark Matter Only (DMO)"))

    plot_massfunction(fig, ax0, fileum, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter,
                                    kwargs_plot=dict(zorder=1, color="tab:blue", label="Universe Machine"), 
                                    kwargs_fill=dict(visible=False))


    ax0.loglog()

    ax0.set_xlabel(r"$m / M_{h}$")
    ax0.set_ylabel(r"$\frac{dN}{d\ln m}$")

    ax0.legend()

    # ax 1

    kwargs_select_inner = dict(
                                    selector_function = lambda d, **k: script_selector_subhalos_valid(d, **k) & script_selector_annulus(d, **k), 
                                    script_kwargs = dict(
                                            r0 = 0,
                                            r1 = 5E-2
                                    )  
                               )

    r0, r1 = 0, 2.5E-2
    annulus_area = np.pi * (r1**2 - r0**2) * 1E6
    nfilter_proj = nfand(freeze(nfilter_project2d, rmin=r0, rmax=r1, normvector=np.array((0, 0, 1))), nfilter)

    plot_massfunction(fig, ax1, filend, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter_proj,
                                    kwargs_plot=dict(zorder=1, color="tab:orange", label="Dark Matter Only"), 
                                    factor=1/annulus_area)

    plot_massfunction(fig, ax1, fileum, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter_proj,
                                    kwargs_plot=dict(zorder=1, color="tab:blue", label="Universe Machine"), 
                                    kwargs_fill=dict(visible=False),
                                    factor=1/annulus_area)


    ax1.loglog()

    ax1.set_xlabel(r"$m / M_{h}$")
    ax1.set_ylabel(r"$\frac{d^2N}{d\ln m dA} [kpc^{-2}]$ " + f" (Inner {1E3*r1:.0f} kpc)")


    #ax2 
    ax2.hlines(1, 1E-5, 1, color="tab:orange", **KWARGS_DEF_PLOT)

    plot_massfunction_ratio(fig, ax2, filend, fileum, mrange, bincount, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter,
                                    kwargs_plot=dict(zorder=1, color="tab:blue", label="Universe Machine"))


    ax2.set_xlabel(r"$m / M_{h}$")
    ax2.set_ylabel(r"Ratio to DMO Mass Function")

    ax2.set_xscale("log")
    ax2.set_ylim(0, 2)

    # ax3
    ax3.hlines(1, 1E-5, 1, color="tab:orange", **KWARGS_DEF_PLOT)

    plot_massfunction_ratio(fig, ax3, filend, fileum, mrange, bincount, 
                                    useratio=useratio, error_plot=False, nfilter=nfilter_proj,
                                    kwargs_plot=dict(zorder=1, color="tab:blue", label="Universe Machine"))


    ax3.set_xlabel(r"$m / M_{h}$")
    ax3.set_ylabel(r"Ratio" + f"(Inner {1E3*r1:.0f} kpc)")

    ax3.set_xscale("log")
    ax3.set_ylim(0, 2)

    savefig(fig, fname + ".png")
    savefig(fig, fname + ".pdf")

if __name__ == "__main__":
    main()


