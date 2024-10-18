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

from spatial_ratio import script_nfw_ratio_with_error

def profile_nfw(r, rs, p0):
    x = r / rs
    return p0 / x / (1 + x)**2

def get_profile_nfw_normalized(file, rrange_rvf, rbins, mrange, kwargs_script=None):
    kwargs_script = {} if kwargs_script is None else kwargs_script

    rv = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.RVIR]))

    dndv, dndv_rspace = script_dndv(file,np.asarray(rrange_rvf) * rv, rbins, script_selector_subhalos_valid, mrange=mrange, **kwargs_script)

    rs = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.SCALE_RADIUS])) 

    nfw = profile_nfw(dndv_rspace, rs, 1) / profile_nfw(dndv_rspace[-1], rs, 1) * dndv[-1] * 1E-9
    return dndv_rspace / rv, nfw



def get_profile_han(file,rrange_rvf, rbins, mrange, gamma, norm=1, kwargs_script=None):
    kwargs_script = {} if kwargs_script is None else kwargs_script

    rvf, nfw = get_profile_nfw_normalized(file,rrange_rvf,rbins,mrange,kwargs_script=(dict(key_mass=GParam.MASS_BOUND) | kwargs_script))

    return rvf, nfw * rvf**gamma


def fit_profile_han(file, rrange_rvf, rbins, mrange, kwargs_script_spatial = None): 

    kwargs_script_spatial = {} if kwargs_script_spatial is None else kwargs_script_spatial

    rv = np.mean(script_select_nodedata(file,script_selector_halos,GParam.RVIR))
    rrange = rv * np.asarray(rrange_rvf)
    ratio, err, _, _, rspace = script_nfw_ratio_with_error(file, rrange, rbins, script_selector_subhalos_valid,
                                                            mrange=mrange, key_mass=GParam.MASS_BOUND, **kwargs_script_spatial) 
    rvfspace = rspace / rv
 
    fitX = np.log10(rvfspace).reshape(-1,1)
    fitY = np.log10(ratio)

    return LinearRegression().fit(fitX, fitY,sample_weight=1/err**2)