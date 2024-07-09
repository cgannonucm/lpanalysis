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
from astropy.cosmology import FlatLambdaCDM

from plotting_util import *

def convert_mpc_to_arcsecond(z, cosmo):

    radtoas = 206265

    dist = cosmo.angularDiameterDistance(z)

    #H0 =  67.36000
    #omegam = 0.31530

    #cosmo = FlatLambdaCDM(H0,omegam)
    #dist = cosmo.angular_diameter_distance(z)

    # dimensional analysis: [rad/Mpc] * [arc second/rad] 
    return  np.asarray((1/dist) * radtoas)

def plot_r2d(fig, ax:Axes, file,rrange_rvf, rbins, mrange, with_error = True, 
             units_rvf = True, kwargs_script = None, kwargs_plot = None, offset = 0, scale = 1):

    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rv = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.RVIR]))

    rrange_mpc = np.asarray(rrange_rvf) * rv if units_rvf else rrange_rvf

    dnda,dnda_error,dnda_r = script_dnda_with_error(file,rrange_mpc,rbins,script_selector_subhalos_valid,mrange=mrange, **kwargs_script)

    kwargs_def_plot = {}
    kwargs_plot = KWARGS_DEF_ERR | kwargs_def_plot | kwargs_plot


    plot_r = dnda_r / rv if units_rvf else dnda_r


    if with_error:
        ax.errorbar((plot_r + offset) * scale, dnda / CPhys.MPC_TO_KPC**2, dnda_error / CPhys.MPC_TO_KPC**2, **kwargs_plot)
        return
    
    ax.plot((plot_r + offset) * scale, dnda / CPhys.MPC_TO_KPC**2, **kwargs_plot)



def set_arcsecond_range(ax, file, rrange, cosmo, axname = None, power = 1, norm = 1):
    axname = "x" if axname is None else axname
    
    z = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.Z_LASTISOLATED])) 

    mpc_to_as = convert_mpc_to_arcsecond(z, cosmo)  

    range = np.asarray(rrange) * (mpc_to_as * norm)**power

    if axname == "x":
        ax.set_xlim(range)   
    elif axname == "y":
        ax.set_ylim(range)
    else:
        raise Exception("")

    return range

def set_arcsecond_range_x(ax, file, rrange_rvf, cosmo): 
    rv = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.RVIR]))
    set_arcsecond_range(ax,file,rrange_rvf,cosmo,"x",1,rv)

def main():    
    path_file_1E13_z05 = "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"

    file = io_importgalout(path_file_1E13_z05)[path_file_1E13_z05] 

    cosmo = cosmology.setCosmology("planck18")

    rrange_rvf = PARAM_DEF_RRANGE_RVF
    rbins = 10
    mrange = PARAM_DEF_MRANGE

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6)) 

    plot_r2d(fig, ax, file, rrange_rvf, rbins, mrange, 
                kwargs_script=dict(key_mass=GParam.MASS_BASIC), 
                kwargs_plot=dict(label="Galacticus (Unevolved)", marker="o"))

    plot_r2d(fig, ax, file, rrange_rvf, rbins, mrange,
                kwargs_script=dict(key_mass=GParam.MASS_BOUND),
                kwargs_plot=dict(label="Galacticus (Evolved)", marker="s"))

    ax.loglog()
    ax.legend()

    ax_twin_x, ax_twin_y = ax.twiny(), ax.twinx()
    ax_twin_x.set_xscale("log")
    ax_twin_y.set_yscale("log")

    set_arcsecond_range_x(ax_twin_x, file, ax.get_xlim(), cosmo)
    set_arcsecond_range(ax_twin_y, file, ax.get_ylim(), cosmo, "y", -2, CPhys.KPC_TO_MPC)

    ax.set_xlabel(r"$r_{2d}/r_v$")
    ax.set_ylabel(r"Projected Number Density [kpc$^{-2}$]")
    ax_twin_x.set_xlabel(r"$r_{2d}$ [arseconds]")
    ax_twin_y.set_ylabel(r"Projected Number Density [arseconds$^{-2}$]")

    savefig(fig, "spatial_2d.png")
    savefig(fig, "spatial_2d.pdf")


if __name__ == "__main__":
    main()