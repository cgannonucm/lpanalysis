#!/usr/bin/env python

import numpy as np
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from typing import Any
import matplotlib.patheffects as pe

from gstk.common.constants import CPhys
from gstk.io import  io_importgalout
from plotting_util import *
from scaling_fit import scaling_fit_mhz_def, scaling_fit_mhz, import_csv, KEY_DEF_HALOMASS, KEY_DEF_Z, galacticus_interp_tstripped, scaling_han_model, scaling_nfw
from han_model import fit_profile_han
from spatial_2d_multihalo import set_mass_range


def plot_scaling(fig, ax:Axes, data, key_toplot, key_toplot_scatter, key_x, 
                    select = None, scale = 1, xshift = 0, kwargs_fill = None,
                    kwargs_plot = None):

    kwargs_fill = {} if kwargs_fill is None else kwargs_fill  
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    x = data[key_x]
    select = np.ones(x.shape) if select is None else select

    z, z_scatter = data[key_toplot] * scale, data[key_toplot_scatter] * scale 
    z_l, z_u = z - z_scatter, z + z_scatter

    ax.fill_between(x[select] + xshift, z_l[select], z_u[select], **kwargs_fill) 
    ax.plot(x[select] + xshift, z[select],  **(KWARGS_DEF_PLOT | kwargs_plot))

def plot_scaling_def(fig, ax:Axes, data, key_toplot, key_toplot_scatter, norm = None, key_x=None,
                    key_y = None, x_shift = 0.5, select = None, kwargs_fill = None, kwargs_plot = None,
                    kwargs_fill_list = None, kwargs_plot_list = None, labeler = None, 
                    mscale=1E13, zshift=0.5):
    # Hard coded to match current csv
    key_x = KEY_DEF_Z if key_x is None else key_x
    key_y = KEY_DEF_HALOMASS if key_y is None else key_y
    labeler = (lambda y: None) if labeler is None else labeler
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill

    x,y = data[key_x], data[key_y]
    y_min, y_max = np.min(y), np.max(y) 

    select = (y == y_min) | (y == y_max) if select is None else select  

    if norm is None:
        fit = scaling_fit_mhz(data, KEY_DEF_HALOMASS, KEY_DEF_Z, key_toplot,mscale=mscale,zshift=zshift)
        norm = 10**(-fit.intercept_)

    y_unique = np.unique(y) 
    

    for n, y_select in enumerate(y_unique): 
        _select = (y == y_select) & select 
        _kwargs_fill = dict(label=labeler(y_select)) | kwargs_fill 
        _kwargs_plot = kwargs_plot
        if np.sum(_select) > 0:
            _kwargs_fill_combined = _kwargs_fill  if kwargs_fill_list is None else _kwargs_fill | kwargs_fill_list[n]
            _kwargs_plot_combined = _kwargs_plot if kwargs_plot_list is None else _kwargs_plot | kwargs_plot_list[n]

            plot_scaling(fig,ax,data,key_toplot,key_toplot_scatter,key_x,_select,norm, 
                            x_shift,kwargs_fill=_kwargs_fill_combined, kwargs_plot=_kwargs_plot_combined)


def plot_scaling_fit(fig, ax, data, key_tofit, mhspace, zspace, key_mass = None, key_z = None, mscale = 1E13, zshift = 0.5,
                        normalize = True, plot_x_label="z", kwargs_plot = None, kwargs_plot_list = None):
    key_z = KEY_DEF_Z if key_z is None else key_z
    key_mass = KEY_DEF_HALOMASS if key_mass is None else key_mass
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    fit = scaling_fit_mhz(data,key_mass,key_z,key_tofit,mscale=mscale,zshift=zshift)
    
    x1,x2 = np.meshgrid(np.log10(np.asarray(mhspace) / mscale), np.log10(np.asarray(zspace) + zshift))
    x = np.asarray((x1.flatten(),x2.flatten())).T

    plt_x1,plt_x2 = np.meshgrid(np.asarray(mhspace) , np.asarray(zspace) + zshift)
    plt_x = np.asarray((plt_x1.flatten(), plt_x2.flatten())).T

    norm = 1 / np.power(10, fit.intercept_) if normalize else 1

    predict = np.power(10,fit.predict(x)) * norm

    if plot_x_label == "z":
        y = x1
        x_index = 1
        y_index = 0

    elif plot_x_label == "m":
        y = x2
        x_index = 0
        y_index = 1
    else:
        raise Exception() 

    for n,y in enumerate(np.unique(y.flatten())):
        _kwargs_plot_n = {} if kwargs_plot_list is None else kwargs_plot_list[n]
        _kwargs_plot = KWARGS_DEF_PLOT | kwargs_plot | (_kwargs_plot_n)
        select = x[::,y_index] == y 
        ax.plot(plt_x[select][::,x_index], predict[select], **_kwargs_plot)

def set_ticks_z(ax:Axes, data, key_z = None, key_axis = "x", nticks = 4, z_shift = 0.5, fstring = None):
    key_z = KEY_DEF_Z if key_z is None else key_z
    fstring = "{:.1f}" if fstring is None else fstring

    z = data[key_z]
    z_min, z_max = np.min(z), np.max(z) 
    z_space = np.linspace(z_min,z_max, nticks)
    z_ticks = z_space + 0.5
    
    if key_axis == "x":
        ax.set_xticks(z_ticks)
        ax.set_xticklabels([fstring.format(_z) for _z in z_space])

    if key_axis == "y":
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([fstring.format(_z) for _z in z_space])



def set_ticks_scaling(ax, range, nticks=4, key_axis = "y", fstring=None):
    fstring = "{:.1f}" if fstring is None else fstring
    nspace = np.linspace(*range,nticks)

    if key_axis == "x":
        ax.set_xticks(nspace)
        ax.set_xticklabels([fstring.format(n) for n in nspace])

    if key_axis == "y":
        ax.set_yticks(nspace) 
        ax.set_yticklabels([fstring.format(n) for n in nspace])


key_n_proj_bound = PARAM_KEY_N_PROJ_BOUND
key_n_proj_bound_scatter = PARAM_KEY_N_PROJ_BOUND_SCATTER
key_n_proj_infall = PARAM_KEY_N_PROJ_INFALL
key_n_proj_infall_scatter = PARAM_KEY_N_PROJ_INFALL_SCATTER

rannulus = PARAM_DEF_RANNULUS
 
kwargs_fill = dict(
                        path_effects=[
                                        pe.Stroke(linewidth=4,foreground="black"),
                                        pe.Normal()
                                    ],
                        alpha=0.6
                    )

                
kwargs_outline = dict(
                        path_effects=[
                                        pe.Stroke(linewidth=7,foreground="black", alpha=1),
                                        pe.Normal()
                                    ]
                )   
kwargs_mean = dict(
                        color="tab:purple",
                        linestyle="dashed",
                        alpha=1,
                        visible=False
                    ) \
                    | kwargs_outline
                        
kwargs_fit = dict(
                    color="black",
                    alpha=1
            )

def convert_to_dict(mh_mg, z_mg, f_mg):
    return {
                KEY_DEF_HALOMASS:                  mh_mg.flatten(),
                KEY_DEF_Z:                         z_mg.flatten(),
                PARAM_KEY_N_PROJ_BOUND:            f_mg.flatten(),
                PARAM_KEY_N_PROJ_BOUND_SCATTER:    np.zeros(mh_mg.flatten().shape)
            }

def plot_z_scaling(fig, axs, filend, scaling_data):
    mh, z = scaling_data[KEY_DEF_HALOMASS], scaling_data[KEY_DEF_Z]
    mh_range = (np.min(mh), np.max(mh))
    z_range = (np.min(z), np.max(z))

    interp_rrange_rvf = PARAM_DEF_RRANGE_RVF
    interp_rbins = 10
    interp_mrange = PARAM_DEF_MRANGE

    h_rannulus = PARAM_DEF_RANNULUS
    h_mspace = mh_range
    h_zspace = np.linspace(*z_range,40)

    cosmo = cosmology.setCosmology("planck18")

    gamma_bf = fit_profile_han(filend,interp_rrange_rvf,interp_rbins,interp_mrange).coef_[0]
    interp_t = galacticus_interp_tstripped(filend, interp_rrange_rvf, interp_rbins, interp_mrange)

    mh_mg, z_mg, scaling_nfw_arr =  scaling_nfw(h_rannulus,h_mspace,h_zspace,cosmo)
    mh_mg, z_mg, scaling_best_fit = scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, gamma_bf)
    mh_mg, z_mg, scaling_g80 =      scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, 0.8)
    mh_mg, z_mg, scaling_g200 =     scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, 2.00)
    mh_mg, z_mg, scaling_interp =   scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, fstripped=interp_t) 

    
    dict_nfw = convert_to_dict(mh_mg, z_mg, scaling_nfw_arr) 
    dict_bestfit = convert_to_dict(mh_mg, z_mg, scaling_best_fit) 
    dict_g080 = convert_to_dict(mh_mg, z_mg, scaling_g80) 
    dict_g200 = convert_to_dict(mh_mg, z_mg, scaling_g200) 
    dict_interp = convert_to_dict(mh_mg, z_mg, scaling_interp) 

    label_hm = lambda m: "$\log_{10}(M_h / M_\odot) = " + f"{np.log10(m):.1f}" + r"$" 

    ax1, ax2 = axs

    plot_scaling_def(fig, ax1, scaling_data,key_n_proj_infall, key_n_proj_infall_scatter,
                        kwargs_fill=kwargs_fill, kwargs_plot=kwargs_mean)

    plot_scaling_def(fig, ax2, scaling_data,key_n_proj_bound, key_n_proj_bound_scatter, 
                        kwargs_fill=kwargs_fill, kwargs_plot=kwargs_mean, labeler=label_hm) 

    plot_scaling_fit(fig, ax1, scaling_data,key_n_proj_infall,np.asarray(mh_range), np.linspace(*z_range, 10), 
                     kwargs_plot=kwargs_fit)

    plot_scaling_fit(fig,ax2,scaling_data,key_n_proj_bound,np.asarray(mh_range), np.linspace(*z_range, 10), 
                     kwargs_plot=kwargs_fit, kwargs_plot_list=[dict(label="Fit"), {}])

    plot_scaling_def(fig, ax1, dict_nfw,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:cyan", linestyle="dashed"))               
    # Dummy plots for labels
    ax2.plot((0,0), (0,0), **(KWARGS_DEF_PLOT | dict(label="Host Density", color="tab:cyan", linestyle="dashed"))) 


    plot_scaling_def(fig, ax2, dict_bestfit,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:green", linestyle="dashed"),
                        kwargs_plot_list=[dict(label="Han (2016) (Best Fit)"), {}]) 

    plot_scaling_def(fig, ax2, dict_interp,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:red", linestyle=(0, (1,1))),               
                        kwargs_plot_list=[dict(label="Han (2016) (Galacticus Interp)"), {}]) 
    
    plot_scaling_def(fig, ax2, dict_g080,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:purple", linestyle="dashdot"),
                        kwargs_plot_list=[dict(label=r"Han (2016) ($\gamma = 0.8$)"), {}]) 

    plot_scaling_def(fig, ax2, dict_g200,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:brown", linestyle=(0, (3, 1, 1, 1, 1, 1))),
                        kwargs_plot_list=[dict(label=r"Han (2016) ($\gamma = 2.0$)"), {}]) 



    fit_unevo = scaling_fit_mhz(scaling_data, KEY_DEF_HALOMASS, KEY_DEF_Z,PARAM_KEY_N_PROJ_INFALL ,mscale=1E13,zshift=0.5)
    fit_evo = scaling_fit_mhz(scaling_data, KEY_DEF_HALOMASS, KEY_DEF_Z,PARAM_KEY_N_PROJ_BOUND, mscale=1E13,zshift=0.5)
    ax_norm = [np.power(10,fit_unevo.intercept_), np.power(10,fit_evo.intercept_)]

    for ax, norm in zip(axs, ax_norm):
        ax.loglog()
        ax.set_ylim((0.10340376167294436, 3.048169659541554))
        ax.set_xlim(np.asarray((0.2,0.8)) + 0.5)
        set_ticks_z(ax, scaling_data, nticks=4) 
        set_ticks_scaling(ax, (0.2, 2.5), nticks=6)
        ax.minorticks_off()
        ax.set_xlabel("z")
        ax_twin_y = ax.twinx()
        set_mass_range(ax_twin_y,(1E8,1E9), PARAM_DEF_ALPHA, np.asarray(ax.get_ylim()) *  norm)
        ax_twin_y.set_ylabel("Projected Mass in Substructure")
        ax_twin_y.set_yscale("log")

    ax1.set_ylabel("$F$")
    ax2.set_ylabel("$F_b$")

    

def plot_mh_scaling(fig, axs, filend, scaling_data):
    mh, z = scaling_data[KEY_DEF_HALOMASS], scaling_data[KEY_DEF_Z]
    mh_range = (np.min(mh), np.max(mh))
    z_range = (np.min(z), np.max(z))

    interp_rrange_rvf = PARAM_DEF_RRANGE_RVF
    interp_rbins = 10
    interp_mrange = PARAM_DEF_MRANGE

    h_rannulus = PARAM_DEF_RANNULUS
    h_mspace = np.geomspace(*mh_range)
    h_zspace = z_range

    cosmo = cosmology.setCosmology("planck18")

    gamma_bf = fit_profile_han(filend,interp_rrange_rvf,interp_rbins,interp_mrange).coef_[0]
    interp_t = galacticus_interp_tstripped(filend, interp_rrange_rvf, interp_rbins, interp_mrange)

    mh_mg, z_mg, scaling_nfw_arr =  scaling_nfw(h_rannulus,h_mspace,h_zspace,cosmo)
    mh_mg, z_mg, scaling_best_fit = scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, gamma_bf)
    mh_mg, z_mg, scaling_g80 =      scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, 0.8)
    mh_mg, z_mg, scaling_g200 =     scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, 2.00)
    mh_mg, z_mg, scaling_interp =   scaling_han_model(h_rannulus,h_mspace,h_zspace, cosmo, fstripped=interp_t) 
    
    dict_nfw = convert_to_dict(mh_mg, z_mg, scaling_nfw_arr) 
    dict_bestfit = convert_to_dict(mh_mg, z_mg, scaling_best_fit) 
    dict_g080 = convert_to_dict(mh_mg, z_mg, scaling_g80) 
    dict_g200 = convert_to_dict(mh_mg, z_mg, scaling_g200) 
    dict_interp = convert_to_dict(mh_mg, z_mg, scaling_interp) 

    label_hm = lambda m: "$\log_{10}(M_h / M_\odot) = " + f"{np.log10(m):.1f}" + r"$" 

    ax1, ax2 = axs

    plot_scaling_def(fig, ax1, scaling_data, key_n_proj_infall, key_n_proj_infall_scatter,
                        kwargs_fill=kwargs_fill, kwargs_plot=kwargs_mean, key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z)

    plot_scaling_def(fig, ax2, scaling_data,key_n_proj_bound, key_n_proj_bound_scatter, 
                        kwargs_fill=kwargs_fill, kwargs_plot=kwargs_mean, labeler=label_hm,  key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z) 

    plot_scaling_fit(fig, ax1, scaling_data,key_n_proj_infall,np.geomspace(*mh_range,10), np.asarray(z_range), 
                     kwargs_plot=kwargs_fit, plot_x_label="m")

    plot_scaling_fit(fig,ax2,scaling_data,key_n_proj_bound, np.geomspace(*mh_range,10), np.asarray(z_range), 
                     kwargs_plot=kwargs_fit, kwargs_plot_list=[dict(label="Fit"), {}], plot_x_label="m")

    plot_scaling_def(fig, ax1, dict_nfw,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:cyan", linestyle="dashed"),
                        key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z)               
    # Dummy plots for labels
    ax2.plot((0,0), (0,0), **(KWARGS_DEF_PLOT | dict(label="Host Density", color="tab:cyan", linestyle="dashed"))) 


    plot_scaling_def(fig, ax2, dict_bestfit,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:green", linestyle="dashed"),
                        kwargs_plot_list=[dict(label="Han (2016) (Best Fit)"), {}],
                        key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z) 

    plot_scaling_def(fig, ax2, dict_interp,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:red", linestyle=(0, (1,1))),               
                        kwargs_plot_list=[dict(label="Han (2016) (Galacticus Interp)"), {}],
                        key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z) 
    
    plot_scaling_def(fig, ax2, dict_g080,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:purple", linestyle="dashdot"),
                        kwargs_plot_list=[dict(label=r"Han (2016) ($\gamma = 0.8$)"), {}],
                        key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z) 

    plot_scaling_def(fig, ax2, dict_g200,key_n_proj_bound, key_n_proj_bound_scatter,norm=1,
                        kwargs_fill=dict(visible=False), kwargs_plot=dict(color="tab:brown", linestyle=(0, (3, 1, 1, 1, 1, 1))),
                        kwargs_plot_list=[dict(label=r"Han (2016) ($\gamma = 2.0$)"), {}],
                        key_x=KEY_DEF_HALOMASS, key_y=KEY_DEF_Z) 



    fit_unevo = scaling_fit_mhz(scaling_data, KEY_DEF_HALOMASS, KEY_DEF_Z,PARAM_KEY_N_PROJ_INFALL ,mscale=1E13,zshift=0.5)
    fit_evo = scaling_fit_mhz(scaling_data, KEY_DEF_HALOMASS, KEY_DEF_Z,PARAM_KEY_N_PROJ_BOUND, mscale=1E13,zshift=0.5)
    ax_norm = [np.power(10,fit_unevo.intercept_), np.power(10,fit_evo.intercept_)]

    for ax, norm in zip(axs, ax_norm):
        ax.loglog()
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim((0.10340376167294436, 3.048169659541554))
        ax.set_xlim(*mh_range)
        #set_ticks_z(ax, scaling_data, nticks=4) 
        set_ticks_scaling(ax, (0.2, 2.5), nticks=6)
        ax.set_xlabel(r"Halo Mass [$M_\odot$]")
        ax_twin_y = ax.twinx()
        ax_twin_y.set_ylabel("Projected Mass in Substructure")
        ax_twin_y.set_yscale("log")

    ax1.set_ylabel("$F$")
    ax2.set_ylabel("$F_b$")




def main():      
    path_csv = "../data/output/analysis_scaling_nd_annulus_new.csv"
    path_file =  "/home/charles/research/lensing_perspective_accompaniment/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"

    scaling_data = import_csv(path_csv)
    filend = io_importgalout(path_file)[path_file] 

    set_plot_defaults()

    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(18,12))

    plot_mh_scaling(fig,axs[0],filend,scaling_data)
    plot_z_scaling(fig,axs[1],filend,scaling_data)

    #axs[1,1].legend(loc="upper right", bbox_to_anchor=(-0.5,-1), fancybox=True, shadow=True) 
    
    fig.tight_layout()


    savefig(fig,"scaling.png")
    savefig(fig,"scaling.pdf")

if __name__ == "__main__":
    main()