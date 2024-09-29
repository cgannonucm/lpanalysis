#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any
import h5py

from subscript.wrappers import gscript, freeze
from subscript.scripts.nodes import nodedata
from subscript.scripts.histograms import spatial3d_dndv
from subscript.scripts.nfilters import nfilter_most_massive_progenitor, nfilter_subhalos_valid
from subscript.defaults import ParamKeys

from plotting_util import set_plot_defaults, savefig_pngpdf, plot_histogram
from symutil import symphony_to_galacticus_hdf5

def plot_dndv(
                    fig, 
                    ax:Axes, 
                    gout, 
                    nfilter,
                    bins=None, 
                    range=None,
                    rvfraction=False,
                    nsigma=1.0,
                    scale_x=1.0,
                    scale_y=1.0,
                    error_plot=False,
                    plot_ratio=False,
                    kwargs_plot=None,
                    kwargs_fill=None,
                    kwargs_script=None
             ):
    @gscript
    def _spatial_dndv(gout, **kwargs):
        _, _bins = np.histogram((1, ), bins=bins, range=range)
        
        rv = nodedata(gout, ParamKeys.rvir, nfilter=nfilter_most_massive_progenitor, summarize=True) 
        scale_y = rv if rvfraction else 1

        radii = _bins * scale_y        
        dndv, _ = spatial3d_dndv(gout, bins=radii, nfilter=nfilter)
        
        if plot_ratio:
            return dndv, radii / rv

        return dndv, radii

    return plot_histogram(
                          fig=fig, 
                          ax=ax,
                          gout=gout,
                          get_histogram=_spatial_dndv,
                          nfilter=nfilter,
                          nsigma=nsigma,
                          error_plot=error_plot,
                          scale_x=scale_x,
                          scale_y=scale_y,
                          kwargs_plot=kwargs_plot,
                          kwargs_fill=kwargs_fill,
                          kwargs_script=kwargs_script,
                          projection=False
                        )
    

def main(): 
    fname = "spatial_3dv2"

    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    path_symphony = "data/symphony/SymphonyGroup/"
    path_gout     = "data/galacticus/um_update/dmo.hdf5"
    path_um       = "data/galacticus/um_update/umachine.hdf5" 
 
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)

    rbin_count = 20
    rbins = np.geomspace(*rrange_rvf, rbin_count)

 
    mrange = (1E9, 1E10)
    
    gout = h5py.File(path_gout)
    gout_um = h5py.File(path_um)
    symout = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)
    
    #print(nodedata(gout,[ParamKeys.mass, ParamKeys.z_lastisolated], nfilter=nfilter_most_massive_progenitor, summarize=True))
    #print(nodedata(symout,[ParamKeys.mass, ParamKeys.z_lastisolated], nfilter=nfilter_most_massive_progenitor, summarize=True))
    #print(nodedata(gout_um,[ParamKeys.mass, ParamKeys.z_lastisolated], nfilter=nfilter_most_massive_progenitor, summarize=True))
    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))
    
    nfsubh = freeze(
                    nfilter_subhalos_valid, 
                    mass_min=mrange[0],
                    mass_max=mrange[1],
                    key_mass=ParamKeys.mass_bound
                   ) 
    
    plot_dndv(
              fig=fig, 
              ax=ax, 
              gout=gout,
              nfilter=nfsubh,
              bins=rbins,
              rvfraction=True,
              plot_ratio=True,
              scale_y=1E-9,
              kwargs_plot=dict(
                               color="tab:orange",
                               label="Galacticus"
                              ),
              kwargs_fill=dict(visible=False)
             )

    plot_dndv(
              fig=fig, 
              ax=ax, 
              gout=gout_um,
              nfilter=nfsubh,
              bins=rbins,
              rvfraction=True,
              plot_ratio=True,
              scale_y=1E-9,
              kwargs_plot=dict(
                               color="tab:green",
                               label="Galacticus (Central Galaxy)"
                              ),
              kwargs_fill=dict(visible=False)
             )

    plot_dndv(
              fig=fig, 
              ax=ax, 
              gout=symout,
              nfilter=nfsubh,
              bins=rbins,
              rvfraction=True,
              plot_ratio=True,
              scale_y=1E-9,
              kwargs_plot=dict(
                               color="tab:blue",
                               label="Symphony"
                              ),
              kwargs_fill=dict(visible=False)
             )
    
    ax.loglog()       
    ax.xaxis.set_ticks(np.asarray((0.1, 0.5, 1.0)))

    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_minor_formatter(ticker.NullFormatter()) 

    ax.set_xlabel(r"$r / r_v$")
    ax.set_ylabel(r"$\frac{dN}{dV}$ [kpc$^{-3}$]")

    ax.legend()    
    ax.set_ylim(1E-7,1E-5)
    ax.set_xlim(*rrange_rvf)

    savefig_pngpdf(fig, fname)
    


if __name__ == "__main__":
    main()
