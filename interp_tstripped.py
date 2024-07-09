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
from typing import Iterable
import pandas as pd
from numpy.dtypes import StrDType

from spatial_ratio import script_nfw_ratio_with_error, plot_spatial_ratio
from plotting_util import *

def galacticus_interp_tstripped(file, rrange_rvf, rbincount, mrange, kwargs_script = None):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    rv = np.mean(script_select_nodedata(file,script_selector_halos,GParam.RVIR))
    rrange = rv * np.asarray(rrange_rvf)

    ratio, err, _, _, rspace = script_nfw_ratio_with_error(file, rrange, rbincount, script_selector_subhalos_valid,
                                                            mrange=mrange, key_mass=GParam.MASS_BOUND, log_space=False)    

    return interp.PchipInterpolator(rspace / rv, ratio, extrapolate=True)

def main():

    path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"

    filend = io_importgalout(path_file)[path_file] 

    rrange_rvf = PARAM_DEF_RRANGE_RVF
    rbins = 10
    mrange = PARAM_DEF_MRANGE

    rvfspace = np.linspace(*rrange_rvf, 100)

    tinterp = galacticus_interp_tstripped(filend, rrange_rvf, rbins, mrange)

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))
    ax:Axes = ax

    plot_spatial_ratio(ax,filend,rrange_rvf,rbins,mrange, GParam.MASS_BOUND, kwargs_format=dict(label="Galacticus"))

    ax.plot(rvfspace, tinterp(rvfspace), label="Galacticus Interpolated", **(KWARGS_DEF_PLOT | dict(linestyle="dashed")))

    ax.legend()
    
    ax.loglog()

    ax.set_xlabel("$r / r_v$")
    ax.set_ylabel(r"$\frac{d^2N}{dm_b dV}(r) \times (\rho_h(r))^{-1}$")

    savefig(fig,"tidal_stripping_interp.png")
    savefig(fig,"tidal_stripping_interp.pdf")
 

if __name__ == "__main__":
    main()