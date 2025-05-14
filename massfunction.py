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

from plotting_util import KWARGS_DEF_ERR, KWARGS_DEF_FILL, KWARGS_DEF_PLOT, plot_histogram, savefig_pngpdf, set_plot_defaults, PlotStyling
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
                            kwargs_fill=None
                       ):
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    
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
                             kwargs_fill=kwargs_fill,
                             scale_y=1 / mf
                            )   
        
def main():
    fname = "massfunction"

    path_nd     = "data/galacticus/mh1E13z05/dmo.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"
    path_um       = "data/galacticus/mh1E13z05_um/umachine.hdf5" 

    gout_nd = tabulate_trees(h5py.File(path_nd))
    gout_um = tabulate_trees(h5py.File(path_um))

    #script_test(filend) 
    #sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203) 
    sym_hdf5 = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    rap_in, rap_out = 0, 5E-2
    plot_dndlnm = True
    plot_ratio  = True

    bins_symphony   = np.logspace(9, 13, 10)
    bins_galacticus = np.logspace(8, 13, 15)  

    xlim = 1E-5, 1E0
    ylim_ratio = 0, 2

    nfilter_subh = freeze(nfilter_subhalos_valid, mass_min=1E8, mass_max=1E13, key_mass=ParamKeys.mass_bound)
    nfilter_proj = freeze(nfilter_project2d, rmin=rap_in, rmax=rap_out)
    nfilter_proj_subh = nfand(nfilter_subh, nfilter_proj)
    
    proj_norm_vectors = np.identity(3)
    kwargs_script_proj = dict(normvector=proj_norm_vectors)
    area_ap = np.pi * (rap_out**2 - rap_in**2)


    set_plot_defaults()

    fig, axs = plt.subplots(figsize=(18,12), ncols = 2, nrows=2)
    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    ax0.text(0.8, 0.7, r'$r < r_{v}$', horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
    ax1.text(0.8, 0.7, r'$r_{2d} < 50 \mathrm{~ kpc}$', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)


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
                        kwargs_plot=(PlotStyling.kwargs_gal_plot |  dict(label="Galacticus")),
                        kwargs_fill=dict(
                                         color=PlotStyling.color_gal_fill
                                        )
                       )

    plot_massfunction(
                        fig, 
                        ax0, 
                        gout_um,
                        bins=bins_galacticus,
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nfilter=nfilter_subh,
                        kwargs_plot=(
                                     PlotStyling.kwargs_gal_plot | 
                                     dict(label="Galacticus (central galaxy)", color=PlotStyling.color_gal_um)
                                    ),
                        kwargs_fill=dict(
                                         visible=False
                                        )
                       )
    # Plots 2 sigma
    #plot_massfunction(
    #                    fig,
    #                    ax0,
    #                    gout_nd,
    #                    bins=bins_galacticus,
    #                    key_mass=ParamKeys.mass_bound,
    #                    plot_dndlnm=plot_dndlnm,
    #                    plot_ratio=plot_ratio,
    #                    error_plot=False,
    #                    nsigma=2,
    #                    nfilter=nfilter_subh,
    #                    kwargs_plot=dict(
    #                                        visible=False
    #                                    ),
    #                    kwargs_fill=dict(
    #                                     color=PlotStyling.color_gal_fill
    #                                    )
    #                   )

    plot_massfunction(
                        fig, 
                        ax0, 
                        sym_hdf5,
                        bins=bins_symphony, 
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,                        
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nfilter=nfilter_subh,
                        kwargs_plot=dict(
                                            color=PlotStyling.color_sym_plot,
                                            path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()],
                                            label="Symphony"
                                        ),                
                        kwargs_fill=dict(visible=False)
                       )

    ax0.loglog()
    
    ax0.set_xlabel(r"$m / M_{\mathrm{h}}$")
    ax0.set_ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d} \ln m}$")
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
                        kwargs_plot=PlotStyling.kwargs_gal_plot,
                        kwargs_fill=dict(
                                            color=PlotStyling.color_gal_fill
                                       )  
                       )  
    # Plots 2 sigma
    #plot_massfunction(
    #                    fig,
    #                    ax1,
    #                    gout_nd,
    #                    bins=bins_galacticus,
    #                    key_mass=ParamKeys.mass_bound,
    #                    plot_dndlnm=plot_dndlnm,
    #                    plot_ratio=plot_ratio,
    #                    error_plot=False,
    #                    nsigma=2,
    #                    nfilter=nfilter_proj_subh,
    #                    projection=True,
    #                    kwargs_script=kwargs_script_proj,
    #                    scale_y=1/area_ap,
    #                    kwargs_plot=dict(
    #                                        visible=False
    #                                    ),
    #                    kwargs_fill=dict(
    #                                        color=PlotStyling.color_gal_fill
    #                                    )
    #                   )

    plot_massfunction(
                        fig, 
                        ax1, 
                        gout_um,
                        bins=bins_galacticus, 
                        key_mass=ParamKeys.mass_bound,
                        plot_dndlnm=plot_dndlnm,                        
                        plot_ratio=plot_ratio,
                        error_plot=False,
                        nfilter=nfilter_proj_subh,
                        projection=True,
                        kwargs_script=kwargs_script_proj,
                        scale_y=1/area_ap,
                        kwargs_plot=dict(
                                            color=PlotStyling.color_gal_um,
                                            path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                        ),
                        kwargs_fill=dict(
                                         visible=False
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
                        error_plot=False,
                        nfilter=nfilter_proj_subh,
                        projection=True,
                        kwargs_script=kwargs_script_proj,
                        scale_y=1/area_ap,
                        kwargs_plot=dict(
                                            color=PlotStyling.color_sym_plot,
                                            path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                        ),
                        kwargs_fill=dict(visible=False)
                       )

    ax1.set_xlabel(r"$m / M_{\mathrm{h}}$")
    ax1.set_ylabel(r"$\frac{\mathrm{d}^2 N}{\mathrm{d} \ln m \mathrm{d}A}$ [kpc$^{-2}$]")

    ax1.loglog()

    # ax2
    ax2.hlines(1.0, *xlim, **(KWARGS_DEF_PLOT | PlotStyling.kwargs_gal_plot))

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
                                             color=PlotStyling.color_sym_plot,
                                             path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                            ),
                            kwargs_fill=dict(
                                             color=PlotStyling.color_sym_plot
                                            )
                           ) 

    plot_massfunction_ratio(
                            fig, 
                            ax2, 
                            gout_um,
                            gout_nd,
                            bins=bins_symphony,
                            key_mass=ParamKeys.mass_bound,
                            plot_ratio=plot_ratio,
                            nfilter=nfilter_subh,
                            kwargs_plot=dict(
                                             color=PlotStyling.color_gal_um,
                                             path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                            ),
                            kwargs_fill=dict(
                                             color=PlotStyling.color_gal_um
                                            )
                           ) 
    ax2:Axes = ax2
    
    ax2.set_xscale("log") 
    ax2.set_xlim(1E-5, 1E0)
    ax2.set_ylim(*ylim_ratio)

    ax2.set_xlabel(r"$m / M_{\mathrm{h}}$")
    ax2.set_ylabel("ratio")
 

    #ax3
    ax3.hlines(1.0, *xlim, **(KWARGS_DEF_PLOT | PlotStyling.kwargs_gal_plot))

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
                                             color=PlotStyling.color_sym_plot,
                                             path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                            ),
                            kwargs_fill=dict(
                                             color=PlotStyling.color_sym_plot
                                            )
                           )     

    plot_massfunction_ratio(
                            fig, 
                            ax3, 
                            gout_um,
                            gout_nd,
                            bins=bins_symphony,
                            key_mass=ParamKeys.mass_bound,
                            plot_ratio=plot_ratio,
                            nfilter=nfilter_proj_subh,
                            projection=True,
                            kwargs_script=kwargs_script_proj,                       
                            kwargs_plot=dict(
                                             color=PlotStyling.color_gal_um,
                                             path_effects=[pe.Stroke(linewidth=8, foreground="black"), pe.Normal()]
                                            ),
                            kwargs_fill=dict(
                                             color=PlotStyling.color_gal_um
                                            )
                           )     
    ax3.set_xscale("log") 
    ax3.set_xlim(1E-5, 1E0)
    ax3.set_ylim(*ylim_ratio)

    ax3.set_xlabel(r"$m / M_{\mathrm{h}}$")
    ax3.set_ylabel("ratio (inner 50kpc)")


    savefig_pngpdf(fig, fname)

if __name__ == "__main__":
    main()


