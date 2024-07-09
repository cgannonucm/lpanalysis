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
from spatial_2d import plot_r2d
from spatial_ratio import sort_gal_files_host, get_mhz_label


def set_mass_range(ax:Axes, mrange, alpha, nrange, scale = 1):
    m0, m1 = mrange
    a1, a2 = alpha + 1, alpha + 2 
    to_m = (a1) / (a2) * (m1**(a2) - m0**(a2)) / (m1**(a1) - m0**(a1))
    ax.set_ylim(np.asarray(nrange) * to_m * scale)

def main():    
    path_nd = "/home/charles/research/lensing_perspective_accompaniment/data/galacticus/xiaolong_update/multihalo"
    name_out = "spatial_2d_multihalo"

    files = io_import_directory(path_nd)
    files_sorted = sort_gal_files_host(files, GParam.MASS_BASIC, reverse=True)

    rrange_mpc = PARAM_DEF_RRANGE_RVF
    rbins = 5
    mrange = PARAM_DEF_MRANGE

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6)) 

    offset_current = 0
    offset_delta = 0.3 * CPhys.KPC_TO_MPC
    offset_delta_current = 0

    markers = ["o", "s", "v", "D"]

    for n, (file, m) in enumerate(zip(files_sorted, markers)):

        plot_r2d(fig, ax, file, rrange_mpc, rbins, mrange, 
                 kwargs_script=dict(key_mass=GParam.MASS_BOUND, logscaling=False), 
                 kwargs_plot=dict(label=get_mhz_label(file), marker=m), 
                 units_rvf=False, offset=offset_current,
                 scale=1E3)

        sgn = 1 if n % 2 == 0 else -1
        offset_delta_current = offset_delta_current + offset_delta if n % 2 == 0 else offset_delta_current
        offset_current = offset_delta_current * sgn

    ax_twin_y = ax.twinx()
    set_mass_range(ax_twin_y, mrange, PARAM_DEF_ALPHA, ax.get_ylim())

    ax.loglog()
    ax.legend(loc="center left", bbox_to_anchor=(1.1,0.5), fancybox=True, shadow=True) 

    ax.set_xlabel(r"$r_{2d}$ [kpc]")
    ax.set_ylabel(r"Projected Number Density [kpc$^{-2}$]")
    ax_twin_y.set_ylabel(r"Projected Mass Density [$M_\odot$ kpc$^{-2}$]")

    savefig(fig, name_out + ".png")
    savefig(fig, name_out + ".pdf")


if __name__ == "__main__":
    main()