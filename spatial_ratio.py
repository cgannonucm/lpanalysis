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
import os.path as path

from gstk.util import util_tabulate
from gstk.hdf5util import read_dataset
from gstk.common.constants import GParam, CPhys
from gstk.util import util_binedgeavg
from gstk.io import csv_cacheable, io_importgalout, io_import_directory
from gstk.scripts.common import ScriptProperties, script, NodeSelector, TabulatedGalacticusData
from gstk.scripts.spatial import script_rvir_halos, script_dnda_with_error, script_dndv_with_error, script_dndv
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus
from gstk.scripts.massfunction import script_massfunction
from gstk.macros.common import MacroScalingKeys
from gstk.macros.scaling import macro_scaling_filemassz
from gstk.scripts.sigmasub import sigsub_var_M0, sigsub_var_sigma_sub, sigsub_var_N0, sigsub_var_M_extrap
from gstk.macros.common import macro_combine, macro_run
from gstk.macros.scaling import macro_scaling_projected_annulus
from gstk.common.util import ifisnone

from plotting_util import *


def nfw_profile(r, rs, p0):
    x = r / rs
    return p0 / x / (1 + x)**2

@script(**ScriptProperties.KEY_KWARG_DEF)
def script_nfw_ratio_with_error(data:TabulatedGalacticusData, rrange:tuple[float,float], rbincount:float,selector_function:NodeSelector, **kwargs):

    tree_index = np.unique(data[kwargs.get(ScriptProperties.KEY_KWARG_TREENUMBER)])

    treecount = len(tree_index)

    dndv_trees = np.empty((treecount,rbincount))  
    nfw_trees = np.empty((treecount,rbincount))
    ratio_trees = np.empty((treecount,rbincount))
    selected = selector_function(data,**kwargs) 

    for i in tree_index:
        selector_tree = lambda d,**k: script_selector_tree(d,i) & selected
        selector_host = lambda d, **k: script_selector_tree(d,i,**k) & script_selector_halos(d, **k)

        dndv_trees[i], bin_avg = script_dndv(data,rrange,rbincount,selector_tree,**kwargs)

        rs = script_select_nodedata(data,selector_host,kwargs.get(ScriptProperties.KEY_KWARG_SCALERADIUS))[0][0]
        nfw_trees[i] = nfw_profile(bin_avg, rs, 1)

        nfw_trees[i] *= dndv_trees[i,-1] / nfw_trees[i,-1]

        ratio_trees[i] = dndv_trees[i] / nfw_trees[i] 
 
    return (np.mean(ratio_trees,axis=0),np.std(ratio_trees,axis=0),np.mean(dndv_trees, axis=0), np.mean(nfw_trees, axis=0), bin_avg)

def plot_spatial_ratio(ax:Axes, nodedata:dict[str,Any], rvrange, rbincount, mrange, key_mass, 
                        plot_log=False, yerr = False, kwargs_script = None, kwargs_format = None):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_format = {} if kwargs_format is None else kwargs_format

    rv = np.mean(script_select_nodedata(nodedata,script_selector_halos,GParam.RVIR))
    rrange = rv * np.asarray(rvrange)
    ratio, err, _, _, rspace = script_nfw_ratio_with_error(nodedata, rrange, rbincount, script_selector_subhalos_valid, mrange=mrange, key_mass=key_mass, **kwargs_script) 

    plt_x = rspace / rv
    if plot_log:
        plt_x = np.log10(plt_x)

    if yerr:
        return ax.errorbar(plt_x, ratio, err, **kwargs_format)

    return ax.plot(plt_x, ratio, **(KWARGS_DEF_PLOT | kwargs_format))


def sort_gal_files_host(files, key, reverse = False):
    fileslist = [f for f in files.values()]

    def sortkey(f):
        return np.mean(script_select_nodedata(f, script_selector_halos, key))

    fileslist.sort(key=sortkey, reverse=reverse)     

    return fileslist

def get_mhz_label(nodedata):
    mh = np.mean(script_select_nodedata(nodedata, script_selector_halos,GParam.MASS_BASIC))
    z = np.mean(script_select_nodedata(nodedata, script_selector_halos,GParam.Z_LASTISOLATED))

    return r"$\log_{10}" + f"(M_h / M_\odot)={np.log10(mh):.1f}, z = {z:.1f}$"

def plot_ratios(path_data, path_out):
    files = io_import_directory(path_data, inclstr=".hdf5")
    files_sorted = sort_gal_files_host(files, GParam.MASS_BASIC, reverse=True)

    rrange = PARAM_DEF_RRANGE_RVF
    rbins = 15
    mrange = PARAM_DEF_MRANGE

    fig, axs = plt.subplots(ncols=2,figsize=(18,6))
    ax1, ax2 = axs 

    for nodedata in files_sorted:
        kwargs_format = KWARGS_DEF_PLOT | dict(label=get_mhz_label(nodedata))

        plot_spatial_ratio(ax1, nodedata, rrange, rbins, mrange, GParam.MASS_BASIC, plot_log=False,kwargs_format=kwargs_format)
        plot_spatial_ratio(ax2, nodedata, rrange, rbins, mrange, GParam.MASS_BOUND, plot_log=False,kwargs_format=kwargs_format)

    for ax in axs:
        ax.hlines((1,), *rrange, linestyle="dashed", label="Host Density", color="black", **KWARGS_DEF_PLOT)
        ax.set_xlabel("$r / r_v$")
        ax.loglog()

    ax1.set_ylabel(r"$\frac{d^2N}{dm dV}(r) \times (\rho_h(r))^{-1}$")
    ax2.set_ylabel(r"$\frac{d^2N}{dm_b dV}(r) \times (\rho_h(r))^{-1}$")

    ax1.set_ylim(8E-1,3E0)

    ax2.legend(fontsize=15)

    savefig(fig, "spatial_ratio.png")
    savefig(fig, "spatial_ratio.pdf")


def main():    
    path_nd = "/home/charles/research/lensing_perspective_accompaniment/data/galacticus/xiaolong_update/multihalo"
    path_out = "/home/charles/research/lensing_perspective_accompaniment/plots/paper"

    set_plot_defaults()

    plot_ratios(path_nd, path_out)


if __name__ == "__main__":
    main()