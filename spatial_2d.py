#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import h5py

from colossus.cosmology import cosmology
from subscript.wrappers import gscript_proj, freeze
from subscript.scripts.nodes import nodedata
from subscript.scripts.histograms import spatial2d_dnda, bin_avg
from subscript.scripts.nfilters import nfilter_subhalos_valid, nfilter_project2d, nfilter_most_massive_progenitor, nfand
from subscript.defaults import ParamKeys, Meta

from plotting_util import set_plot_defaults, savefig_pngpdf, plot_histogram, PlotStyling, KWARGS_DEF_PLOT, getlogger
from symutil import symphony_to_galacticus_hdf5

def plot_dnda(
              fig, 
              ax:Axes, 
              gout, 
              nfilter,
              normvector,
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
    kwargs_script = {} if kwargs_script is None else kwargs_script

    @gscript_proj
    def _spatial_dnda(gout, normvector, **kwargs):
        _, _bins = np.histogram((1, ), bins=bins, range=range)
        
        rv = nodedata(gout, ParamKeys.rvir, nfilter=nfilter_most_massive_progenitor, summarize=True) 
        scale_y = rv if rvfraction else 1

        radii = _bins * scale_y        
        dnda, _ = spatial2d_dnda(gout, bins=radii, nfilter=nfilter, normvector=normvector)
        
        if plot_ratio:
            return dnda, radii / rv

        return dnda, radii

    return plot_histogram(
                          fig=fig, 
                          ax=ax,
                          gout=gout,
                          get_histogram=_spatial_dnda,
                          nfilter=nfilter,
                          nsigma=nsigma,
                          error_plot=error_plot,
                          scale_x=scale_x,
                          scale_y=scale_y,
                          kwargs_plot=kwargs_plot,
                          kwargs_fill=kwargs_fill,
                          kwargs_script=(kwargs_script | dict(normvector=normvector)),                    
                          projection=False
                        )

def equal_area_bins(rmin, rmax, rbin_count):
    bina = np.pi * (rmax**2 - rmin**2) / rbin_count
    return np.sqrt(
                   np.arange(rbin_count)             
                   *bina
                   /np.pi
                   +rmin**2
                  )

def main():    
    #Meta.cache = False
    fname = "spatial_2d"

    logger = getlogger(fname + ".txt")
    #path_gout = "/home/charles/research/lpanalysis/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"   
    path_gout =  "data/galacticus/um_update/dmo/dmo.hdf5"
    path_um =  "data/galacticus/um_update/umachine.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"

    gout = h5py.File(path_gout)
    gout_um = h5py.File(path_um)
    symout = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    #print(np.log10(np.min(nodedata(gout, key=ParamKeys.mass_bound)[0])))
    
    rrange = np.asarray((1E-3, 22E-3))
    rbin_count = 20
    rbins = equal_area_bins(*rrange, rbin_count)

    bina_arr = np.pi * (rbins[1:]**2 - rbins[:-1]**2)
    np.testing.assert_allclose(bina_arr[0], bina_arr[-1])
    bina = np.mean(bina_arr)

    logger.info(f"Bin area: {bina * 1E6} kpc^2")
    logger.info(f"Bin radius: {np.sqrt(bina * 1E6 / np.pi)}")

    #marr = np.logspace(8,10, 3)
    #mmin_arr = marr[:-1]
    #mmax_arr = marr[1:]
    mmin_arr = [1E8, ]
    mmax_arr = [1E13, ]
    
    colors_galacticus = [PlotStyling.color_gal_fill , None]
    colors_sym        = [None                       , PlotStyling.color_sym_plot]

    set_plot_defaults()


    fig, ax = plt.subplots(figsize=(9,6))

    rbins_avg = bin_avg(rbins)
    xlim = np.asarray((rbins_avg[0]*1E3, rbins_avg[-1]*1E3))
    ax.set_xlim(xlim)

    twinx = ax.twiny()

    cosmo = cosmology.setCosmology("planck18")
    rad_to_as = 2.06E5
    to_as =  1 / cosmo.angularDiameterDistance(0.5) * cosmo.h * rad_to_as

    twinx.set_xlim(xlim * 1E-3 * to_as)
    twiny = ax.twinx()
    ylim = np.asarray((5E-4, 3E0))
    ax.set_ylim(ylim)
    
    twiny.set_ylim(ylim * (1E3 * 1 / to_as)**2)
    twiny.set_yscale("log")

    for n, (mmin, mmax) in enumerate(zip(mmin_arr, mmax_arr)):
        normvectors = np.identity(3)
        nfsubh_evo = freeze(
                            nfilter_subhalos_valid, 
                            mass_min=mmin,
                            mass_max=mmax,
                            key_mass=ParamKeys.mass_bound
                           ) 

        nfsubh_unevo = freeze(
                              nfilter_subhalos_valid, 
                              mass_min=mmin,
                              mass_max=mmax,
                              key_mass=ParamKeys.mass_basic
                             ) 
 
        colorg = colors_galacticus[n]
        colorsym = colors_sym[n]

        plot_dnda(
                  fig, 
                  ax, 
                  gout,
                  nfilter=nfsubh_unevo,
                  normvector=normvectors,
                  bins=rbins, 
                  scale_y=1E-6,
                  scale_x=1E3,
                  error_plot=False,
                  kwargs_plot=(PlotStyling.kwargs_gal_plot | (dict(color=PlotStyling.color_gal_fill_unevo))),
                  kwargs_fill=dict(
                                    color=PlotStyling.color_gal_fill_unevo,
                                    label="Galacticus (unevolved)"
                                  )
                 )                

        plot_dnda(
                  fig, 
                  ax, 
                  gout,
                  nfilter=nfsubh_evo,
                  normvector=normvectors,
                  bins=rbins, 
                  scale_y=1E-6,
                  scale_x=1E3,
                  error_plot=False,
                  kwargs_plot=(PlotStyling.kwargs_gal_plot | (dict(color=colorg))),
                  kwargs_fill=dict(
                                   color=colorg,
                                   label="Galacticus (evolved)"
                                  )
                 )


    plot_dnda(
              fig,
              ax,
              gout_um,
              nfilter=nfsubh_evo,
              normvector=normvectors,
              bins=rbins,
              scale_y=1E-6,
              scale_x=1E3,
              error_plot=False,
              kwargs_plot=(
                           PlotStyling.kwargs_gal_plot |
                           dict(
                                color="tab:green",
                                linestyle="dotted",
                                label="Galacticus (central galaxy)"
                               )
                          ),
              kwargs_fill=dict(
                               visible=False
                              )
             )

    slacs_median_einstien_radius_arcsec = 1.17
    print(slacs_median_einstien_radius_arcsec / to_as * 1E3)


    ax.vlines(
              slacs_median_einstien_radius_arcsec / to_as * 1E3,
              ymin=ylim[0],
              ymax=ylim[-1], 
              label="SLACS median Einstien radius",
              color="tab:brown", 
              linestyle="dashed",
              **KWARGS_DEF_PLOT      
             )
    
 
    ax.set_xlabel(r"$r_{\mathrm{2d}}$ [kpc]")
    ax.set_ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d}A}$ [kpc$^{-2}$] ($m > 10^8 M_\odot$)")

    twinx.set_xlabel(r"$r_{\mathrm{2d}}$ [arcsec]")
    twiny.set_ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d}A}$ [arcsec$^{-2}$] ($m > 10^8 M_\odot$)")
   
    #ax.set_ylim(5E-5, 6E-2)
    ax.legend(loc="lower right", fontsize="17")

    ax.set_yscale("log")

    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_formatter('{x:.0f}')    


    twinx.xaxis.set_major_formatter('{x:.0f}\"')
    ax.xaxis.set_minor_formatter('{x:.0f}')    
    
    savefig_pngpdf(fig, fname)    

if __name__ == "__main__":
    main()
