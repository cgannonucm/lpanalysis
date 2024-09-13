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
from han_model import get_profile_nfw_normalized, get_profile_han, fit_profile_han
from conversion import cfactor_shmf

from plotting_util import *
from symutil import symphony_to_galacticus_dict
from spatial_3d import plot_spatial_3d_scatter

def main(): 
    figname = "spatial_3d_um"
    
    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    path_cat = "data/caterpillar/subhaloDistributionsCaterpillar.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"

    path_dmo = "data/galacticus/um_update/dmo.hdf5"
    path_um = "data/galacticus/um_update/umachine.hdf5"
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.02, 1)
    rbins = 15
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E10)
    mrange_rescale = (1E9, 1E10)

    file_dmo, file_um = h5py.File(path_dmo), h5py.File(path_um)

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))    

    plot_spatial_3d_scatter(fig, ax, file_dmo, rrange_rvf, rbins, (1E8, 1E13), 
                            error_plot=False,
                            kwargs_fill=dict(color="tab:orange"),
                            kwargs_plot=(KWARGS_DEF_PLOT | dict(label="Universe Machine", color="black", zorder=100)))

    plot_spatial_3d_scatter(fig, ax, file_um, rrange_rvf, rbins, (1E8, 1E13), 
                            error_plot=True,
                            kwargs_fill=dict(visible=False),
                            kwargs_plot=(KWARGS_DEF_PLOT | dict(label="Universe Machine", color="tab:blue", zorder=100)))  
    ax.loglog()

    ax.set_xlabel(r"$r / r_v$")
    ax.set_ylabel(r"Subhalo Number Density [kpc$^{-3}$]") 

    #ax.get_yaxis().set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fancybox=True, shadow=True) 

    savefig(fig, figname + ".png")
    savefig(fig, figname + ".pdf")
if __name__ == "__main__":
    main()
