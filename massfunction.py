#!/usr/bin/env python

import h5py
import numpy as np
import symlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any, Callable
import os

from subscript.scripts.nodes import nodedata
from subscript.wrappers import freeze, gscript, multiproj
from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.histograms import massfunction, bin_avg
from subscript.scripts.nfilters import nfilter_most_massive_progenitor, nfilter_subhalos_valid, nfilter_project2d, nfand
from subscript.defaults import ParamKeys

from plotting_util import KWARGS_DEF_ERR, KWARGS_DEF_FILL, KWARGS_DEF_PLOT, plot_histogram, savefig_pngpdf, set_plot_defaults
from symutil import symphony_to_galacticus_dict, symphony_to_galacticus_hdf5

def plot_massfunction(
                        fig, 
                        ax:Axes, 
                        gout, 
                        bins=None, 
                        range=None,
                        nfilter=None,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=False,  
                        plot_ratio = False,
                        error_plot=False,
                        projection=False,
                        scale_x = 1.0,
                        scale_y = 1.0,
                        nsigma = 1,
                        kwargs_plot=None,
                        kwargs_fill=None,
                        kwargs_script=None,
                       ):
    
    _, mbins = np.histogram((1, ), bins=bins, range=range)
    nfitler_def = freeze(nfilter_subhalos_valid, mass_min=mbins[0], mass_max=mbins[-1], key_mass=key_mass)
    _nfilter = nfitler_def if nfilter is None else nfilter
    
    @gscript
    def _mf(gout, **kwargs):
        mf, mf_bins = massfunction(gout, key_mass=key_mass, bins=bins, range=range, **kwargs)
        _hist, _bins = mf, mf_bins
        if plot_dndlnm:
            _hist = mf * bin_avg(mf_bins)
        if plot_ratio:
            mh = nodedata(gout, key=ParamKeys.mass_basic, nfilter=nfilter_most_massive_progenitor, summarize=True) 
            _bins = mf_bins / mh
        return _hist, _bins
    
    return plot_histogram(
                          fig, 
                          ax, 
                          gout,
                          get_histogram=_mf,
                          nfilter=_nfilter,
                          nsigma=nsigma,                       
                          error_plot=error_plot,
                          scale_x=scale_x,
                          scale_y=scale_y,
                          projection=projection,
                          kwargs_script=kwargs_script, 
                          kwargs_plot=kwargs_plot,
                          kwargs_fill=kwargs_fill
                         )

def plot_massfunction_ratio(
                            fig, 
                            ax:Axes, 
                            gout_numerator, 
                            gout_denominator,
                            bins=None, 
                            range=None,
                            nfilter=None,
                            key_mass=ParamKeys.mass_bound,
                            plot_ratio=False,
                            projection=False,
                            kwargs_plot=None,
                            kwargs_script=None,
                       ):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    
    _, mbins = np.histogram((1, ), bins=bins, range=range)
    nfitler_def = freeze(nfilter_subhalos_valid, mass_min=mbins[0], mass_max=mbins[-1], key_mass=key_mass)
    _nfilter = nfitler_def if nfilter is None else nfilter
    
    @gscript
    def _mf(gout, **kwargs):
        mf, mf_bins = massfunction(gout, key_mass=key_mass, bins=bins, range=range, **kwargs)
        _hist, _bins = mf, mf_bins
        if plot_ratio:
            mh = nodedata(gout, key=ParamKeys.mass_basic, nfilter=nfilter_most_massive_progenitor, summarize=True) 
            _bins = mf_bins / mh
        return _hist, _bins

    _get_hist = multiproj(_mf, nfilter) if projection else freeze(_mf, nfilter=nfilter)
    mf, _     = _get_hist(gout_denominator, summarize=True, **kwargs_script)    
 
    return plot_massfunction(
                             fig, 
                             ax, 
                             gout_numerator,
                             bins=bins,
                             range=range,
                             nfilter=_nfilter,
                             error_plot=False,
                             plot_dndlnm=False,
                             plot_ratio=plot_ratio,
                             projection=projection,
                             kwargs_script=kwargs_script, 
                             kwargs_plot=kwargs_plot,
                             kwargs_fill=dict(
                                               visible=False
                                             ),
                             scale_y=1 / mf,
                            )   
        
def main():
    fname = "massfunction"
    path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5" 
    path_symphony = "data/symphony/SymphonyGroup/"

    gout_nd = tabulate_trees(h5py.File(path_file))

    #script_test(filend) 
    #sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203) 
    sym_hdf5 = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    rap_in, rap_out = 0, 5E-2
    plot_dndlnm = True
    plot_ratio  = True

    bins_symphony   = np.logspace(9, 13, 20)
    bins_galacticus = np.logspace(8, 13, 30)  

    xlim = 1E-5, 1E0
    ylim_ratio = 0, 2

    nfilter_subh = freeze(nfilter_subhalos_valid, mass_min=1E8, mass_max=1E13, key_mass=ParamKeys.mass_bound)
    nfilter_proj = freeze(nfilter_project2d, rmin=rap_in, rmax=rap_out)
    nfilter_proj_subh = nfand(nfilter_subh, nfilter_proj)
    
    proj_norm_vectors = np.identity(3)
    kwargs_script_proj = dict(normvector=proj_norm_vectors)
    area_ap = np.pi * (rap_out**2 - rap_in**2)

    # style 
    color_gal_fill            = "tab:orange"
    color_gal_plot            = "tab:orange"
    color_gal_plot_foreground = "black"
    color_sym_plot            = "tab:blue"

    kwargs_gal_plot = dict(
                            color=color_gal_plot,
                            path_effects=[pe.Stroke(linewidth=8, foreground=color_gal_plot_foreground), pe.Normal()]                    
                          )
    
    set_plot_defaults()

    fig, axs = plt.subplots(figsize=(18,12), ncols = 2, nrows=2)
    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    for ax in axs.flatten():
        ax.set_xlim(*xlim)

    # ax0
    plot_massfunction(
                        fig, 
                        ax0, 
                        gout_nd,
                        bins=bins_galacticus,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nfilter=nfilter_subh,
                        kwargs_plot=(kwargs_gal_plot |  dict(label="Galacticus")),
                        kwargs_fill=dict(
                                         color=color_gal_fill
                                        )
                       )


    plot_massfunction(
                        fig, 
                        ax0, 
                        gout_nd,
                        bins=bins_galacticus,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nsigma=2,
                        nfilter=nfilter_subh,
                        kwargs_plot=dict(                                            
                                            visible=False
                                        ),
                        kwargs_fill=dict(
                                         color=color_gal_fill
                                        )                       
                       )

    plot_massfunction(
                        fig, 
                        ax0, 
                        sym_hdf5,
                        bins=bins_symphony, 
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,                        
                        plot_ratio=plot_ratio,
                        error_plot=True,
                        nfilter=nfilter_subh,
                        kwargs_plot=dict(
                                            color=color_sym_plot,
                                            label="Symphony"
                                        )                             
                       )

    ax0.loglog()
    
    ax0.set_xlabel(r"$m / M_h$")
    ax0.set_ylabel(r"$\frac{dN}{d \ln m}$")
    ax0.legend()


    # ax 1
    plot_massfunction(
                        fig, 
                        ax1, 
                        gout_nd,
                        bins=bins_galacticus,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nfilter=nfilter_proj_subh,
                        projection=True,
                        kwargs_script=kwargs_script_proj,
                        scale_y=1/area_ap,
                        kwargs_plot=kwargs_gal_plot,
                        kwargs_fill=dict(
                                            color=color_gal_fill
                                       ) 
                       )  

    plot_massfunction(
                        fig, 
                        ax1, 
                        gout_nd,
                        bins=bins_galacticus,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nsigma=2,
                        nfilter=nfilter_proj_subh,
                        projection=True,
                        kwargs_script=kwargs_script_proj,
                        scale_y=1/area_ap,
                        kwargs_plot=dict(
                                            visible=False                                            
                                        ),
                        kwargs_fill=dict(
                                            color=color_gal_fill
                                        )  
                       )

    plot_massfunction(
                        fig, 
                        ax1, 
                        sym_hdf5,
                        bins=bins_symphony, 
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,                        
                        plot_ratio=plot_ratio,
                        error_plot=True,
                        nfilter=nfilter_proj_subh,
                        projection=True,
                        kwargs_script=kwargs_script_proj,
                        scale_y=1/area_ap,
                        kwargs_plot=dict(
                                            color=color_sym_plot
                                        )
                       )

    ax1.set_xlabel(r"$m / M_h$")
    ax1.set_ylabel(r"$\frac{d^2 N}{d \ln m dA}$ [kpc$^{-2}$]")

    ax1.loglog()

    # ax2
    ax2.hlines(1.0, *xlim, **(KWARGS_DEF_PLOT | kwargs_gal_plot))

    plot_massfunction_ratio(
                            fig, 
                            ax2, 
                            sym_hdf5,
                            gout_nd,
                            bins=bins_symphony,
                            key_mass=ParamKeys.mass_bound,
                            plot_ratio=plot_ratio,
                            nfilter=nfilter_subh,
                            kwargs_plot=dict(
                                             color=color_sym_plot
                                            )
                           ) 
    ax2:Axes = ax2
    
    ax2.set_xscale("log") 
    ax2.set_xlim(1E-5, 1E0)
    ax2.set_ylim(*ylim_ratio)
    ax2.set_xlabel("$M / M_h$")
    ax2.set_ylabel("ratio")
 

    #ax3
    ax3.hlines(1.0, *xlim, **(KWARGS_DEF_PLOT | kwargs_gal_plot))

    plot_massfunction_ratio(
                            fig, 
                            ax3, 
                            sym_hdf5,
                            gout_nd,
                            bins=bins_symphony,
                            key_mass=ParamKeys.mass_bound,
                            plot_ratio=plot_ratio,
                            nfilter=nfilter_proj_subh,
                            projection=True,
                            kwargs_script=kwargs_script_proj,                       
                            kwargs_plot=dict(
                                             color=color_sym_plot
                                            )
                           )     
    ax3.set_xscale("log") 
    ax3.set_xlim(1E-5, 1E0)
    ax3.set_ylim(*ylim_ratio)


    ax3.set_xlabel("$M / M_h$")
    ax3.set_ylabel("ratio (inner 50kpc)")


    savefig_pngpdf(fig, fname)

if __name__ == "__main__":
    main()


