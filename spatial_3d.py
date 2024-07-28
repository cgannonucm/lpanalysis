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

def plot_spatial(ax:Axes, file, rrange_rvf, rbins, mrange, kwargs_script = None,
                  kwargs_plot = None, mrange_rescale = None, alpha = None, cum=False): 
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rv = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.RVIR]))
    dndv, rspace = script_dndv(file,np.asarray(rrange_rvf) * rv, rbins, script_selector_subhalos_valid, mrange=mrange, **kwargs_script)

    mrange_rescale = mrange if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    rescale = 1 #cfactor_shmf(mrange,mrange_rescale, alpha)
 
    ax.plot(rspace/rv,dndv * 1E-9 * rescale, **kwargs_plot)

def plot_spatial_3d_scatter(fig, ax:Axes, file, rrange_rvf, rbins, mrange,
                                kwargs_script = None, kwargs_plot = None,
                                mrange_rescale=None, alpha=None, error_plot=False, factor = 1): 

    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rv = np.mean(script_select_nodedata(file,script_selector_halos,[GParam.RVIR]))
    dndv, err, rspace = script_dndv_with_error(file,np.asarray(rrange_rvf) * rv, rbins, script_selector_subhalos_valid, mrange=mrange, **kwargs_script)

    mrange_rescale = mrange if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    #rescale = cfactor_shmf(mrange,mrange_rescale, alpha) * CPhys.MPC_TO_KPC**(-3) * factor
    rescale = CPhys.MPC_TO_KPC**(-3) * factor
    
    plot_x, plot_y, plot_y_std = rspace / rv, dndv * rescale, err  * rescale

    if error_plot:
        ax.errorbar(plot_x, plot_y, plot_y_std, **(KWARGS_DEF_ERR | kwargs_plot))        
        return 

    lower,upper = plot_y - plot_y_std, plot_y + plot_y_std

    ax.fill_between(plot_x, lower, upper, **(kwargs_plot | dict(label="Scatter")))

def get_spatial_cat(f):
    cat_dn = read_dataset(f["radialDistributin"]["radialDistribution"])
    cat_dn_error = read_dataset(f["radialDistribution"]["radialDistributionError"])
    cat_rfraction = read_dataset(f["radialDistribution"]["radiusFractional"])

    # https://arxiv.org/pdf/1509.01255.pdf
    # from table 3
    # need to check if data file is agregate or for just one halo
    cat_rvir_avg = 291.791 
    cat_mass = 1.368E12
    cat_z = 0
    
    rfrac_min = 0
    
    #From scaling n-d large
    density_scaling_coef_z = 1.62067827
    density_scaling_coef_mass = 0.00335998
    
    cat_rbin_vol = (4/3)*np.pi*(cat_rfraction[1:]**3 - cat_rfraction[:-1]**3) * cat_rvir_avg**3
    
    
    cat_dndv = (cat_dn[1:] + cat_dn[:-1]) / cat_rbin_vol / 2
    # Should I weigh by number of subhalos in each bin?
    cat_dndv_error = np.sqrt((cat_dn_error[:-1])**2 + (cat_dn_error[1:])**2) / cat_rbin_vol
    cat_dndv_rfraction = cat_rfraction[1:]#util_binedgeavg(cat_rfraction)
    
    cat_coef_rescale = 1/((0.5+cat_z)**density_scaling_coef_z * (cat_mass/1E13)**density_scaling_coef_mass)
    #cat_coef_rescale = 1E13/cat_mass

    cat_select = cat_dndv_rfraction > 2E-2
    
    return cat_dndv_rfraction[cat_select], cat_dndv[cat_select] * cat_coef_rescale

def get_spatial_symphony():
    # Spatial information - symphony from Xiaolong Du
    sy_rvf = np.asarray((
                            1.584893192461114082e-03,
                            2.511886431509579437e-03,
                            3.981071705534973415e-03,
                            6.309573444801930275e-03,
                            1.000000000000000021e-02,
                            1.584893192461114125e-02,
                            2.511886431509580825e-02,
                            3.981071705534973415e-02,
                            6.309573444801933051e-02,
                            1.000000000000000056e-01,
                            1.584893192461114264e-01,
                            2.511886431509582351e-01,
                            3.981071705534973137e-01,
                            6.309573444801935826e-01,
                            1.000000000000000000e+00
                        ))
    sy_n = np.asarray((
                            0.000000000000000000e+00,
                            0.000000000000000000e+00,
                            0.000000000000000000e+00,
                            0.000000000000000000e+00,
                            0.000000000000000000e+00,
                            2.040816326530612082e-02,
                            6.122448979591836593e-02,
                            1.020408163265306145e-01,
                            3.877551020408163129e-01,
                            1.061224489795918435e+00,
                            3.775510204081632626e+00,
                            1.167346938775510168e+01,
                            2.953061224489795933e+01,
                            6.944897959183673208e+01,
                            1.360612244897959044e+02
                        )) 
    rv = 0.4292

    binvol = (4/3) * np.pi * (sy_rvf[1:]**3 - sy_rvf[:-1]**3) * rv**3 
    dndv = (sy_n[1:] + sy_n[:-1]) / binvol / 2
    return sy_rvf[1:], dndv * 1E-9
 
def plot_spatial_unevolved(fig, ax, file, rrange, rbins, mrange, kwargs_plot = None, mrange_rescale = None):
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    plot_spatial(ax,file, rrange, rbins, mrange,
                    dict(key_mass=GParam.MASS_BASIC), 
                    KWARGS_DEF_PLOT | dict(label="Galacticus (Unevolved)") | kwargs_plot, 
                    mrange_rescale=mrange_rescale)

    plot_spatial_3d_scatter(fig,ax,file,rrange,30,mrange, 
                                kwargs_script=dict(key_mass=GParam.MASS_BASIC), 
                                kwargs_plot=(KWARGS_DEF_FILL | dict(color="tab:blue")), 
                                mrange_rescale=mrange_rescale)    
  

def plot_spatial_evolved(fig, ax, file, rrange, rbins, mrange, kwargs_plot = None, mrange_rescale = None):
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot


    plot_spatial(ax,file, rrange, rbins,mrange,
                    dict(key_mass=GParam.MASS_BOUND), 
                    KWARGS_DEF_PLOT | dict(label="Galacticus (Evolved)", color="black") | kwargs_plot, 
                    mrange_rescale=mrange_rescale)

    plot_spatial_3d_scatter(fig,ax,file,rrange,30,mrange, 
                                kwargs_script=dict(key_mass=GParam.MASS_BOUND), 
                                kwargs_plot=KWARGS_DEF_FILL, 
                                mrange_rescale=mrange_rescale)





def plot_average_density(fig, ax, file, rrange_rvf, rbins, mrange, kwargs_script=None, kwargs_plot=None,
                            mrange_rescale=None, alpha=None): 
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    mrange_rescale = mrange if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    rescale = cfactor_shmf(mrange,mrange_rescale, alpha)

    rvfspace, nfw = get_profile_nfw_normalized(file,rrange_rvf,rbins,mrange,(dict(key_mass = GParam.MASS_BASIC) | kwargs_script))
    ax.plot(rvfspace, nfw * rescale, **(KWARGS_DEF_PLOT | dict(linestyle="dashed", label="Host Density \n(Rescaled)", color="grey")| kwargs_plot))

def plot_spatial_3d_cat(fig,ax,file, kwargs_plot = None, mrange_rescale = None, alpha = None):
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    rvf, dndv = get_spatial_cat(file)

    mrange = (1E8, 1E13)
    mrange_rescale = mrange if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    rescale = cfactor_shmf(mrange,mrange_rescale, alpha)
    
    ax.plot(rvf, dndv * rescale, **(KWARGS_DEF_PLOT | dict(label="Caterpillar") | kwargs_plot))

def plot_spatial_3d_symphony(fig, ax,alpha=PARAM_DEF_ALPHA,kwargs_plot = None,
                                mrange_rescale = None):
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rvf, dndv = get_spatial_symphony()
    select = (rvf > 2E-2)
    
    mrange = (1E9, 1E13)
    mrange_rescale = (1E8, 1E13) if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    rescale = cfactor_shmf(mrange,mrange_rescale, alpha)
    
    ax.plot(rvf[select], dndv[select] * rescale, **(KWARGS_DEF_PLOT | dict(label="Symphony")| kwargs_plot))

def plot_han_3d(fig, ax,file,rrange_rvf, rbins, mrange, gamma, norm=1,
                 kwargs_script=None, kwargs_plot=None, mrange_rescale = None, alpha = None):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rvf, dndv = get_profile_han(file,rrange_rvf, rbins, mrange, gamma, kwargs_script)

    mrange_rescale = mrange if mrange_rescale is None else mrange_rescale
    alpha = PARAM_DEF_ALPHA if alpha is None else alpha

    rescale = cfactor_shmf(mrange,mrange_rescale, alpha)

    ax.plot(rvf,dndv * norm * rescale, **(KWARGS_DEF_PLOT | dict(label=rf"Han (2016) ($\gamma = {gamma:.2f}$)") | kwargs_plot))

def main(): 
    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    path_cat = "data/caterpillar/subhaloDistributionsCaterpillar.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"
    path_file = "/home/charles/research/lpanalysis/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)
    rbins = 15
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E10)
    mrange_rescale = (1E9, 1E10)

    filend = io_importgalout(path_file)[path_file] 
    filecat = h5py.File(path_cat)
    # isnap 203 is snapshot at z~0.5
    sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))
    
    han_fit = fit_profile_han(filend, rrange_rvf, rbins, mrange)
    han_norm, han_gamma = 10**(han_fit.intercept_), han_fit.coef_[0]

    #print(np.mean(script_select_nodedata(filend, script_selector_halos, [GParam.RVIR])))
    #print(np.mean(script_select_nodedata(sym_nodedata, script_selector_halos, [GParam.RVIR])))



    #plot_spatial_evolved(fig,ax,filend,rrange_rvf,rbins,(1E9, 1E10), 
    #                     dict(zorder=10))

    plot_spatial_unevolved(fig,ax,filend,rrange_rvf,rbins,mrange,
                            dict(zorder=10), mrange_rescale=mrange_rescale)
    
    plot_spatial_evolved(fig,ax,filend,rrange_rvf,rbins,mrange, 
                         dict(zorder=10))

    plot_average_density(fig,ax,filend,rrange_rvf,rbins,mrange, dict(zorder=5, color="grey"), 
                         mrange_rescale=mrange_rescale)

    
    #plot_spatial_3d_cat(fig,ax,filecat, dict(zorder=5))
    #plot_spatial_3d_symphony(fig, ax, kwargs_plot=dict(zorder=5), mrange_rescale=mrange_rescale)

    #plot_han_3d(fig,ax,filend,rrange,rbins,mrange, 0.7, dict(zorder=10))
    #plot_han_3d(fig,ax,filend,rrange,rbins,mrange, 1, kwargs_plot=dict(zorder=10))
    #plot_han_3d(fig,ax,filend,rrange,rbins,mrange, 1.35, kwargs_plot=dict(zorder=10))
    plot_han_3d(fig,ax,filend,rrange_rvf,rbins,mrange, gamma=han_gamma,norm=han_norm,
                    kwargs_plot=dict(zorder=10, label="Han (2016) (Best Fit)", color="tab:purple"),
                    mrange_rescale=mrange_rescale) 

    plot_spatial_3d_scatter(fig, ax, sym_nodedata, rrange_rvf, rbins, mrange_sym, 
                            error_plot=True, kwargs_plot=dict(zorder=30, label="Symphony (Group)", color="tab:green"))

    ax.loglog()

    ax.set_xlabel(r"$r / r_v$")
    ax.set_ylabel(r"Subhalo Number Density [kpc$^{-3}$]") 

    #ax.get_yaxis().set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1,0.5), fancybox=True, shadow=True) 

    #ax.set_xlim(0.014, 1.1)    
    ax.set_xlim(*rrange_rvf)
    #ax.set_ylim(1.1E-6,3E-2)

    savefig(fig, "spatial_3d.png")
    savefig(fig, "spatial_3d.pdf")


if __name__ == "__main__":
    main()
