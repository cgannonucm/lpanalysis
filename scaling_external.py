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

from plotting_util import savedf
from scaling_fit import  scaling_fit_mhz_def, import_csv, PARAM_KEY_N_PROJ_BOUND


def scaling_F(mh, z, k1, k2, mscale=1E13, zshift=0.5):
    return np.power(mh / mscale, k1) * np.power(z + zshift, k2)

def main(): 
    path_csv = "data/output/analysis_scaling_nd_annulus_new.csv"
    scaling_data = import_csv(path_csv)

    fit = scaling_fit_mhz_def(scaling_data, key_n_proj=PARAM_KEY_N_PROJ_BOUND)

    cosmo = cosmology.setCosmology("planck18")


    # The nadler prior using the old scaling relations, we need to rescale down to milky way mass redshift z = 0
    # Then we scale back up again
    nadler_l_0, nadler_u_0 = 1.5E-2, 3.0E-2  
    nadler_rescale = scaling_F(1.7E12, 0, 0.88, 1.7)
    #print(nadler_rescale)
    nadler_l, nadler_u = nadler_l_0 * nadler_rescale, nadler_u_0 * nadler_rescale
    

    # Ponos data https://iopscience.iop.org/article/10.3847/0004-637X/824/2/144
    # Their N_0 is the same as our sigma_sub, but with redshift / halo mass dependence
    # Need to get halo masses for ponos v, ponosq at z=0.7
    
    extrapolate_table = {
                    "label"                                 : np.asarray(("Ponos V" ,"Ponos Q"  ,"Xu"               ,"Nadler Lower Bound"   ,"Nadler Upper Bound"   )),
                    "mh [M_\odot]"                          : np.asarray((1.2E13    ,6.5E12     ,1E13 * cosmo.h     ,1.7E12                 ,1.7E12                 )),
                    "z"                                     : np.asarray((0.7       ,0.7        ,0.6                ,0.0                    ,0.0                    )),
                    "N_0 [kpc^-2] (Unscaled Sigma Sub)"     : np.asarray((6E-3      ,6E-3       ,3E-3 * cosmo.h     ,nadler_l               , nadler_u              ))
    }
    
    k1, k2 = fit.coef_

    extrapolate_table["k1"] = np.ones(extrapolate_table["label"].shape) * k1
    extrapolate_table["k2"] = np.ones(extrapolate_table["label"].shape) * k2

    fscale = scaling_F(extrapolate_table["mh [M_\odot]"],extrapolate_table["z"],k1,k2)
    extrapolate_table[r"\Sigma_{sub}[kpc^{-2}]"] = extrapolate_table["N_0 [kpc^-2] (Unscaled Sigma Sub)"] / fscale

    df = pd.DataFrame(extrapolate_table) 
    savedf(df,"scaling_extrapolate.csv")

if __name__ == "__main__":
    main()