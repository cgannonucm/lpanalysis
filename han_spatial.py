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
from spatial_ratio import script_nfw_ratio_with_error, sort_gal_files_host, get_mhz_label
from scaling_fit import fits_to_df

from han_model import fit_profile_han

def main():
    fname = "han_fits.csv"
    path_nd = "/home/charles/research/lensing_perspective_accompaniment/data/galacticus/xiaolong_update/multihalo" 
    path_m13z05 =  "/home/charles/research/lensing_perspective_accompaniment/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    files = io_import_directory(path_nd)
    files |= io_importgalout(path_m13z05)
    files_sorted = sort_gal_files_host(files, GParam.MASS_BASIC, reverse=True) 
    
    rrange_rvf = PARAM_DEF_RRANGE_RVF
    #rrange_rvf = (0.03, 0.3)
    rbins = 10
    mrange = PARAM_DEF_MRANGE

    fits = {}
    
    for file in files_sorted:
        label = get_mhz_label(file)
        fits[label] = fit_profile_han(file,rrange_rvf,rbins,mrange)

    
    df = fits_to_df(fits)
    savedf(df, fname)


if __name__ == "__main__":
    main()
