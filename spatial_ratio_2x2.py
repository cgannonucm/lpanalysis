#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib import ticker

from han_modelv2 import profile_nfw
from subscript.scripts.histograms import bin_avg
from plotting_util import savefig_pngpdf, set_plot_defaults, KWARGS_DEF_PLOT

def select_index_closest_mh_z(mh, z, select_mh, select_z):
    vec = np.asarray((mh - select_mh, z - select_z))
    return np.argmin(np.linalg.norm(vec, axis=0))

def plot_spatial_ratio(fig, ax, dndv, rbins_rvf, host_rv, host_c, kwargs_plot=None):
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot

    rvf = bin_avg(rbins_rvf)
    r = rvf * host_rv
    rs = host_rv / host_c

    nfw_norm = profile_nfw(r, rs, 1) / profile_nfw(r[-1], rs, 1)

    ratio = dndv / nfw_norm / dndv[-1] 

    ax.plot(rvf,ratio, **(KWARGS_DEF_PLOT | kwargs_plot))

def main():
    fname = "spatial_ratiov2"

    path_summary_scaling = "out/hdf5/summary_scaling.hdf5"
    #path_summary_scaling = "out/hdf5/scaling_test.hdf5"

    scalingsum = h5py.File(path_summary_scaling)

    halo_mass = scalingsum["halo mass (mean)/out0"][:]
    halo_z = scalingsum["z (mean)/out0"][:]

    set_plot_defaults()

    fig, axs = plt.subplots(figsize=(18,12), ncols=2, nrows=2)
    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    hm_select = np.asarray((10**12, 10**13.5))
    z_select = np.asarray((0.2, 0.8)) 
    mhg, zg = np.meshgrid(hm_select, z_select)
 
    
    for ax in axs[0]:
        ax.set_prop_cycle("color", mpl.colormaps["Set2"](np.arange(4)))

    reltol_select = 1E-5
    
    for n, (hm_s, z_s) in enumerate(zip(mhg.flatten(), zg.flatten())): 
        n = select_index_closest_mh_z(halo_mass, halo_z, hm_s,z_s)
        mh, z = halo_mass[n], halo_z[n]

        dndv_group_evo = scalingsum["dNdV (evolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (mean)"]
        dndv_group_unevo = scalingsum["dNdV (unevolved) [MPC^${-3}$] 1.00E+08 < m <= 3.16E+08 (mean)"]

        dndv_evo, dndv_evo_rbins_rvf = dndv_group_evo["out0"][:][n], dndv_group_evo["out1"][:][n]
        dndv_unevo, dndv_unevo_rbins_rvf = dndv_group_unevo["out0"][:][n], dndv_group_unevo["out1"][:][n]
        
        c, rv = scalingsum["concentration (host) (mean)/out0"][:][n], scalingsum["rvir (host) (mean)/out0"][:][n], 

        plot_spatial_ratio(fig, ax0, dndv_evo, dndv_evo_rbins_rvf, rv, c, kwargs_plot=dict(label=f"z={z:.1f}, log $M_h$ = {np.log10(mh):.1f}"))
        plot_spatial_ratio(fig, ax1, dndv_unevo, dndv_unevo_rbins_rvf, rv, c)

    for ax in axs.flatten():
        _ax:Axes = ax

    
    massrange3d = np.logspace(8, 11, 7)
    mmin_arr = massrange3d[:-1]
    mmax_arr = massrange3d[1:]

    nselect = select_index_closest_mh_z(halo_mass, halo_z, 1E13, 0.5)

    for mmin, mmax in zip(mmin_arr, mmax_arr):
        dndv_group_evo = scalingsum[r"dNdV (evolved) [MPC^${-3}$] " + f"{mmin:.2E} < m <= {mmax:.2E} (mean)"]
        dndv_group_unevo = scalingsum[r"dNdV (unevolved) [MPC^${-3}$] " + f"{mmin:.2E} < m <= {mmax:.2E} (mean)"]

        dndv_evo, dndv_evo_rbins_rvf = dndv_group_evo["out0"][:][nselect], dndv_group_evo["out1"][:][nselect]
        dndv_unevo, dndv_unevo_rbins_rvf = dndv_group_unevo["out0"][:][nselect], dndv_group_unevo["out1"][:][nselect]
        
        c, rv = scalingsum["concentration (host) (mean)/out0"][:][nselect], scalingsum["rvir (host) (mean)/out0"][:][nselect], 

        plot_spatial_ratio(fig, ax2, dndv_evo, dndv_evo_rbins_rvf, rv, c)
        plot_spatial_ratio(fig, ax3, dndv_unevo, dndv_unevo_rbins_rvf, rv, c)
        

    for ax in axs.flatten():
        ax.loglog()
        #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.hlines(1, 0.1, 1, **(KWARGS_DEF_PLOT), color="black", linestyle="dashed", label="NFW")
        ax.yaxis.set_minor_formatter('{x:.1f}')
        ax.yaxis.set_major_formatter('{x:.1f}')
        ax.set_ylabel("Ratio to NFW")
        ax.set_xlabel("$r / r_v$")


    #labels 
    ax0.legend()
    fig.tight_layout()    
    savefig_pngpdf(fig, fname)
    



if __name__ == "__main__":
    main()