#!/usr/bin/env python

import h5py
import numpy as np
import symlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any, Callable
import os

from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import nfilter_most_massive_progenitor
from subscript.wrappers import freeze, gscript
from subscript.tabulatehdf5 import tabulate_trees, get_galacticus_outputs
from subscript.defaults import ParamKeys

from plotting_util import *
from symutil import symphony_to_galacticus_dict

def plot_most_massive_progenitor_property(fig,
                                            ax:Axes,
                                            gout: h5py.File, 
                                            toplot_x: (str | Callable),
                                            toplot_y: (str | Callable),
                                            x_scale = 1, 
                                            y_scale = 1,
                                            nsigma = 1,
                                            error_plot = True,
                                            kwargs_script = None, 
                                            kwargs_plot = None,
                                            kwargs_fill = None,
                                            out_indexes=None
                                        ):
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot    
    out_indexes = get_galacticus_outputs(gout) if out_indexes is None else out_indexes

    x, y_avg, y_std = np.zeros(out_indexes.shape), np.zeros(out_indexes.shape), np.zeros(out_indexes.shape)

    get_x = freeze(nodedata, key=toplot_x) if isinstance(toplot_x, str) else toplot_x
    get_y = freeze(nodedata, key=toplot_y) if isinstance(toplot_y, str) else toplot_y
    nfmmp = nfilter_most_massive_progenitor

    for n, oindex, in enumerate(out_indexes):
        trees = tabulate_trees(gout, out_index=oindex) 
        x[n], y_avg[n], y_std[n] = get_x(trees, summarize=True, nfilter=nfmmp), *get_y(trees, summarize=True, statfuncs=(np.mean, np.std), nfilter=nfmmp)

    x *= x_scale
    y_avg *= y_scale
    y_std *= y_scale

    y_std *= nsigma
    
    if error_plot: 
        ax.errorbar(x, y_avg, y_std, **(KWARGS_DEF_ERR | kwargs_plot))
        return

    ax.plot(x, y_avg, **(KWARGS_DEF_PLOT | kwargs_plot))
    ax.fill_between(x,y_avg - y_std, y_avg + y_std, **(KWARGS_DEF_FILL | kwargs_fill))

def main():
    fname = "um_evolution"
    path_file = "data/galacticus/um_update/um-multiz/multiz.hdf5"  
    #path_file = "/home/charles/research/lpanalysis/data/galacticus/scaling_um/lsubmodv3.2-um-scaling-date-09.18.2024-time-19.20.41-z-2.00000E-01-hm-1.00000E+13.xml.hdf5"

    with h5py.File(path_file) as gout:
        outputs = get_galacticus_outputs(gout)
    
        set_plot_defaults()


        mpl.rcParams.update({"font.size":25})
    
        fig, axs = plt.subplots(figsize=(18,18), ncols=2, nrows=3)

        ax2, ax3 = axs[0]
        ax0, ax1 = axs[1]
        ax4, ax5 = axs[2]
    
        out_indexes = get_galacticus_outputs(gout)[::5]


        # ax0
        plot_most_massive_progenitor_property(
                                                fig, 
                                                ax0,
                                                gout,
                                                ParamKeys.z_lastisolated,
                                                ParamKeys.sphere_mass_stellar,
                                                error_plot=False, 
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot
                                            )

    
        ax0.set_yscale("log")
    
        ax0.set_xlabel("z")
        ax0.set_ylabel(r"M$_\star$ [$M_\odot$]")
    

        # ax1
        plot_most_massive_progenitor_property(  
                                                fig, 
                                                ax1,
                                                gout, 
                                                ParamKeys.z_lastisolated,
                                                ParamKeys.sphere_radius, 
                                                error_plot=False, 
                                                y_scale=1E3,
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot
                                            )
    
        ax1.set_yscale("log")
    
        ax1.set_xlabel("z")
        ax1.set_ylabel(r"r$_\star$ [kpc]")

        #ax 2
        plot_most_massive_progenitor_property(  
                                                fig, 
                                                ax2,
                                                gout, 
                                                ParamKeys.z_lastisolated,
                                                ParamKeys.mass_basic,
                                                error_plot=False, 
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot
                                            )
    
        ax2.set_yscale("log")
    
        ax2.set_xlabel("z")
        ax2.set_ylabel(r"M$_h$ [$M_\odot$]")
    
        # ax3
        plot_most_massive_progenitor_property(  
                                                fig, 
                                                ax3,
                                                gout, 
                                                ParamKeys.z_lastisolated,
                                                ParamKeys.rvir, 
                                                error_plot=False, 
                                                y_scale=1E3,
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot

                                            )
    
        ax3.set_yscale("log")
    
        ax3.set_xlabel("z")
        ax3.set_ylabel(r"r$_v$ [kpc]")
    
        # ax4
        @gscript
        def mhmstarratio(o, **k):
            return o[ParamKeys.sphere_mass_stellar] / o[ParamKeys.mass_basic]
    
        plot_most_massive_progenitor_property(
                                                fig, 
                                                ax4,
                                                gout, 
                                                ParamKeys.z_lastisolated,
                                                mhmstarratio,
                                                error_plot=False, 
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot
                                            ) 
    
        ax4.set_yscale("log")
    
        ax4.set_xlabel("z")
        ax4.set_ylabel(r"M$_\star$ / M$_h$")
    
        # ax5
        @gscript
        def rstarrvirratio(o, **k):
            return o[ParamKeys.sphere_radius] / o[ParamKeys.rvir]
    
        plot_most_massive_progenitor_property(
                                                fig,
                                                ax5,
                                                gout, 
                                                ParamKeys.z_lastisolated,
                                                rstarrvirratio,
                                                error_plot=False,                                    
                                                out_indexes=out_indexes,
                                                kwargs_plot=PlotStyling.kwargs_gal_plot
                                            )
     
        ax5.set_yscale("log")
    
        ax5.set_xlabel("z")
        ax5.set_ylabel(r"r$_\star$ / r$_{v}$")
    
        for ax in (ax4, ax5):
            ax.set_ylim(1E-3, 2E-2)
    
        savefig_pngpdf(fig, fname)
    
if __name__ == "__main__":
    main()
