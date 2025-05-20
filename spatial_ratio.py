#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib import ticker
from colossus.cosmology import cosmology
from colossus.halo import concentration
from sklearn.linear_model import LinearRegression

from subscript.scripts.histograms import bin_avg, spatial3d_dndv
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys
from subscript.scripts.nfilters import nfilter_subhalos_valid, nfilter_halos
from subscript.wrappers import freeze

from plotting_util import savefig_pngpdf, set_plot_defaults, KWARGS_DEF_PLOT, PlotStyling
from spatial_3d import get_dndv
from symutil import symphony_to_galacticus_hdf5
from han_model_fit import fit_han_model_gout, fit_han_model, profile_nfw, select_index_closest_mh_z

def spatial_ratio(dndv, rbins_rvf, host_rv, host_c):
    rvf = bin_avg(rbins_rvf)
    r = rvf * host_rv
    rs = host_rv / host_c

    nfw_norm = profile_nfw(r, rs, 1) / profile_nfw(r[-1], rs, 1)

    return dndv / nfw_norm / dndv[-1]    


def plot_spatial_ratio(fig, ax, dndv, rbins_rvf, host_rv, host_c, kwargs_plot=None):
    from sklearn.linear_model import LinearRegression

    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rvf = bin_avg(rbins_rvf)

    ratio = spatial_ratio(dndv, rbins_rvf, host_rv, host_c)
    ax.plot(rvf,ratio, **(KWARGS_DEF_PLOT | kwargs_plot))

    select = rvf < 0.6
    fitX = np.log10(rvf[select]).reshape(-1,1)
    fitY = np.log10(ratio[select])

    fit = LinearRegression().fit(fitX, fitY)
    #print("coef", fit.coef_)
    
    predict = 10**fit.intercept_ * (rvf)**fit.coef_
    
    #ax.plot(rvf,predict, **(KWARGS_DEF_PLOT | kwargs_plot))

def plot_spatial_ratio_gout(fig, ax, gout, nfilter, rbins_rvf, kwargs_plot=None):
    dndv, _ = get_dndv(gout, nfilter=nfilter, bins=rbins_rvf, rvfraction=True, summarize=True)
    host_c, host_rv = nodedata(gout, key=("concentration", ParamKeys.rvir), nfilter=nfilter_halos, summarize=True)
    plot_spatial_ratio(fig, ax, dndv, rbins_rvf, host_rv, host_c, kwargs_plot=kwargs_plot)     


def plot_spatial_ratio_fit(fig, ax, dndv, fit:LinearRegression, rrange_rvf, kwargs_plot):
    ratio = 10**(fit.intercept_) * (rrange_rvf) ** fit.coef_  / dndv[-1]
    ax.plot(rrange_rvf, ratio, **(KWARGS_DEF_PLOT | kwargs_plot))

def main():
    fname = "spatial_ratio"
 
    path_symphony = "data/symphony/SymphonyGroup/"
    path_summary_scaling = "out/hdf5/summary_scaling.hdf5"
    #path_summary_scaling = "out/hdf5/scaling_test.hdf5"

    scalingsum = h5py.File(path_summary_scaling)
    hdf5_sym = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    halo_mass = scalingsum["halo mass (mean)/out0"][:]
    halo_z = scalingsum["z (mean)/out0"][:]

    set_plot_defaults()

    fig, axs = plt.subplots(figsize=(18,6), ncols=2)
    ax0, ax1 = axs

    hm_select = np.asarray((10**12, 10**13.5))
    z_select = np.asarray((0.2, 0.8)) 
    mhg, zg = np.meshgrid(hm_select, z_select)
 
    
    for ax in axs:
        ax.set_prop_cycle("color", mpl.colormaps["Dark2"](np.arange(4) + 1))

    markers = [".", "s", "*", "X"] 

    reltol_select = 1E-5
    
    for i, (hm_s, z_s) in enumerate(zip(mhg.flatten(), zg.flatten())): 
        n = select_index_closest_mh_z(halo_mass, halo_z, hm_s,z_s)
        mh, z = halo_mass[n], halo_z[n]

        dndv_group_evo = scalingsum["dNdV (evolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (mean)"]
        dndv_group_unevo = scalingsum["dNdV (unevolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (mean)"]

        dndv_evo, dndv_evo_rbins_rvf = dndv_group_evo["out0"][:][n], dndv_group_evo["out1"][:][n]
        dndv_unevo, dndv_unevo_rbins_rvf = dndv_group_unevo["out0"][:][n], dndv_group_unevo["out1"][:][n]
        
        c, rv = scalingsum["concentration (host) (mean)/out0"][:][n], scalingsum["rvir (host) (mean)/out0"][:][n], 
        m = markers[i]

        plot_spatial_ratio(fig, ax0, dndv_evo, dndv_evo_rbins_rvf, rv, c, kwargs_plot=dict(label=f"z={z:.1f}" + r", $\log_{10} \left( M_{\mathrm{h}} / M_\odot \right)$" + f" = {np.log10(mh):.1f}"))
        plot_spatial_ratio(fig, ax1, dndv_unevo, dndv_unevo_rbins_rvf, rv, c)       

        # fit 
        fit = fit_han_model(dndv_evo, dndv_evo_rbins_rvf, c, rv)

        continue
        plot_spatial_ratio_fit(
                               fig, 
                               ax0, 
                               dndv_evo,  
                               fit, 
                               dndv_evo_rbins_rvf,
                               kwargs_plot=dict(
                                                color="grey",
                                                linestyle="dotted"
                                               )
                              )

    # plot symphony
    mrange_sym = (1E9, 1E10)
    nfsubh_evo = freeze(
                        nfilter_subhalos_valid,
                        mass_min=mrange_sym[0],
                        mass_max=mrange_sym[-1],
                        key_mass=ParamKeys.mass_bound                                
                       )

    mass = np.concatenate(nodedata(hdf5_sym, key=ParamKeys.mass_basic))

    # Concentrations are not tabulated in the symphony results
    # So lets use the diemer model instead

    cosmology.setCosmology("planck18")
    z = nodedata(hdf5_sym, key=ParamKeys.z_lastisolated, nfilter=nfilter_halos, summarize=True)
    c = concentration.concentration(mass, 'vir', z, model="diemer19")

    hdf5_sym.create_dataset("Outputs/Output1/nodeData/concentration", data=c)    
    
    plot_spatial_ratio_gout(
                            fig, 
                            ax0,
                            hdf5_sym, 
                            nfilter=nfsubh_evo, 
                            rbins_rvf=dndv_evo_rbins_rvf,
                            kwargs_plot=dict(
                                             label="Symphony",
                                             color=PlotStyling.color_sym_plot
                                            )
                           )

    fit = fit_han_model_gout(
                             hdf5_sym, 
                             dndv_evo_rbins_rvf,
                             nfilter=nfsubh_evo,
                            )

    dndv_sym_evo, _ = get_dndv(
                               hdf5_sym,
                               bins=dndv_evo_rbins_rvf, 
                               rvfraction=True,
                               nfilter=nfsubh_evo,
                               summarize=True,  
                              )
    print(dndv_sym_evo)

    #plot_spatial_ratio_fit(
    #                        fig, 
    #                        ax0, 
    #                        dndv_sym_evo,  
    #                        fit, 
    #                        dndv_evo_rbins_rvf,
    #                        kwargs_plot=dict(
    #                                         color="grey",
    #                                         linestyle="dotted",
    #                                         label="fit"
    #                                        )
    #                       )

    ax0.set_ylabel(r"$n_{\mathrm{sub}} / \rho_{\mathrm{h}}$ (evolved)")
    ax1.set_ylabel(r"$n_{\mathrm{sub}} / \rho_{\mathrm{h}}$ (unevolved)")


    for ax in axs.flatten():
        ax.loglog()
        #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.hlines(1, 0.1, 1, **(KWARGS_DEF_PLOT), color="black", linestyle="dashed", label=r"$\rho_h$")
        #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(nbins=6))
        ax.xaxis.set_minor_formatter('{x:.1f}')
        ax.xaxis.set_major_formatter('{x:.1f}')

        ax.yaxis.set_minor_locator(ticker.MaxNLocator(nbins=6))
        ax.yaxis.set_minor_formatter('{x:.1f}')
        ax.yaxis.set_major_formatter('{x:.1f}')

        ax.set_xlabel(r"$r / r_{\mathrm{v}}$")

    #labels 
    ax0.legend()
    fig.tight_layout()    
    savefig_pngpdf(fig, fname) 

if __name__ == "__main__":
    main()
