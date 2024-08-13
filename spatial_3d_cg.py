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
    figname = "spatial_3d_cg"
    
    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    path_cat = "data/caterpillar/subhaloDistributionsCaterpillar.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"
    path_file = "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    path_cg1 = "data/galacticus/xiaolong_update/cg-2/lsubmodv3.1-cg-date-07.11.2024-time-00.55.15-date-07.11.2024-time-00.55.16-z-5.00000E-01-mh-1.00000E+13-mstar-1.00000E+12-dmstar-0.00000E+00-drstar-0.00000E+00.xml.hdf5"
    path_cg2 = "data/galacticus/xiaolong_update/cg-2/lsubmodv3.1-cg-date-07.11.2024-time-00.55.15-date-07.11.2024-time-00.55.16-z-5.00000E-01-mh-1.00000E+13-mstar-2.51189E+11-dmstar-0.00000E+00-drstar-0.00000E+00.xml.hdf5"
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)
    rbins = 15
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E10)
    mrange_rescale = (1E9, 1E10)

    filend = io_importgalout(path_file)[path_file] 
    filend_cg1 = io_importgalout(path_cg1)[path_cg1]
    filend_cg2 = io_importgalout(path_cg2)[path_cg2]

    #filecat = h5py.File(path_cat)
    # isnap 203 is snapshot at z~0.5
    #sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))
    
    han_fit = fit_profile_han(filend, rrange_rvf, rbins, mrange)
    han_norm, han_gamma = 10**(han_fit.intercept_), han_fit.coef_[0]


    #plot_spatial_3d_scatter(fig,ax,filend,rrange_rvf,rbins,mrange, 
    #                            error_plot=False,
    #                            kwargs_script=dict(key_mass=GParam.MASS_BASIC), 
    #                            kwargs_fill=(KWARGS_DEF_FILL | dict(color="tab:blue", label="Scatter")), 
    #                            kwargs_plot=(KWARGS_DEF_PLOT | dict(color="tab:blue", label="Galacticus (Unevolved)")),
    #                            )     
    
    plot_spatial_3d_scatter(fig,ax,filend,rrange_rvf,rbins,mrange,
                                error_plot=False, 
                                kwargs_script=dict(key_mass=GParam.MASS_BOUND), 
                                kwargs_fill=(KWARGS_DEF_FILL | dict(label="Scatter")), 
                                kwargs_plot=(KWARGS_DEF_PLOT | dict(color="tab:orange", label="Galacticus (Evolved)")),
                                )     
           

    #plot_average_density(fig,ax,filend,rrange_rvf,rbins,mrange, dict(zorder=5, color="grey"), 
    #                     mrange_rescale=mrange_rescale)

    
    #plot_han_3d(fig,ax,filend,rrange_rvf,rbins,mrange, gamma=han_gamma,norm=han_norm,
    #                kwargs_plot=dict(zorder=10, label="Han (2016) (Best Fit)", color="tab:purple"),
    #                mrange_rescale=mrange_rescale) 

    #plot_spatial_3d_scatter(fig, ax, sym_nodedata, rrange_rvf, rbins, mrange_sym, 
    #                        error_plot=True, kwargs_plot=dict(zorder=30, label="Symphony (Group)", color="tab:green"))
    plot_spatial_3d_scatter(fig, ax, filend_cg1, rrange_rvf, rbins, mrange_sym, 
                                error_plot=False, kwargs_fill=dict(visible=False),
                                kwargs_plot=(KWARGS_DEF_PLOT | dict(label="Central Galaxy [$10^{12} M_\odot$]", color="tab:brown", zorder=2)))
 
 

    plot_spatial_3d_scatter(fig, ax, filend_cg2, rrange_rvf, rbins, mrange_sym, 
                                error_plot=True, 
                                kwargs_fill=dict(visible=False),
                            kwargs_plot=(KWARGS_DEF_PLOT | dict(label="Central Galaxy [$10^{11.4} M_\odot$]", color="tab:red", zorder=100)))

   
    ax.loglog()

    ax.set_xlabel(r"$r / r_v$")
    ax.set_ylabel(r"Subhalo Number Density [kpc$^{-3}$]") 

    #ax.get_yaxis().set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fancybox=True, shadow=True) 

    #ax.set_xlim(0.014, 1.1)    
    ax.set_xlim(*rrange_rvf)
    #ax.set_ylim(1.1E-6,3E-2)

    savefig(fig, figname + ".png")
    savefig(fig, figname + ".pdf")


if __name__ == "__main__":
    main()
