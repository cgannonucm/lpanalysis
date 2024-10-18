#!/usr/bin/env python
import numpy as np
import h5py
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from subscript.scripts.nodes import nodedata
from subscript.scripts.histograms import massfunction, bin_avg
from subscript.scripts.nfilters import nfilter_halos, nfilter_subhalos_valid, nfilter_project2d, nfand
from subscript.defaults import ParamKeys
from subscript.wrappers import freeze, multiproj

from massfunction_fits import fit_loglog_massfunction, fits_to_df
from plotting_util import savedf, savefig_pngpdf, getlogger

def main():
    fname = "massfunction_normalization_fits_singlehalo"

    logger = getlogger(fname + ".txt")

    path_dmo = "/home/charles/research/lpanalysis/data/galacticus/um_update/dmo/dmo.hdf5"
    path_um =  "data/galacticus/um_update/umachine.hdf5"

    gout_dmo = h5py.File(path_dmo)
    gout_um = h5py.File(path_um)

    #print(nodedata(gout_dmo, key=[ParamKeys.mass_basic, ParamKeys.z_lastisolated], nfilter=nfilter_halos, summarize=True))
    #print(nodedata(gout_um, key=[ParamKeys.mass_basic, ParamKeys.z_lastisolated], nfilter=nfilter_halos, summarize=True))

    mmin, mmax = 1E8, 1E10

    nfsubh_evo = freeze(
                        nfilter_subhalos_valid, 
                        key_mass=ParamKeys.mass_bound,
                        mass_min=mmin, 
                        mass_max=mmax
                       )

    mf_bins = np.geomspace(mmin, mmax, 30)
    
    mf_dmo, _ = massfunction(gout_dmo, key_mass=ParamKeys.mass_bound, bins=mf_bins, nfilter=nfsubh_evo, summarize=True)
    mf_um, _ = massfunction(gout_um, key_mass=ParamKeys.mass_bound, bins=mf_bins, nfilter=nfsubh_evo, summarize=True)

    fit_dmo = fit_loglog_massfunction(mf_dmo, mf_bins)
    fit_um = fit_loglog_massfunction(mf_um, mf_bins)

    rmin, rmax = 1E-3, 1E-2
    normvector = np.identity(3)

    nfproj = freeze(
                    nfilter_project2d,
                    rmin=rmin, 
                    rmax=rmax
                   )
    nfsubh_proj_evo =  nfand(nfproj, nfsubh_evo)

    out_dmo_proj = multiproj(massfunction, nfilter=nfsubh_proj_evo)(gout_dmo, key_mass=ParamKeys.mass_bound, bins=mf_bins, summarize=True, normvector=normvector, statfuncs=(np.mean, np.std))
    out_um_proj  = multiproj(massfunction, nfilter=nfsubh_proj_evo)(gout_um , key_mass=ParamKeys.mass_bound, bins=mf_bins, summarize=True, normvector=normvector, statfuncs=(np.mean, np.std))   
    
    mf_dmo_proj, mf_dmo_proj_std = out_dmo_proj[0][0], out_dmo_proj[1][0]
    mf_um_proj , mf_um_proj_std  = out_um_proj[0][0] , out_um_proj[1][0]
    
    fitX = np.log10((bin_avg(mf_bins))).reshape(-1, 1) 
    fitY_dmo = np.log10(mf_dmo_proj)
    fit_dmo_proj = LinearRegression().fit(fitX, fitY_dmo, sample_weight=mf_dmo_proj)

    fitY_um = np.log10(mf_um_proj)
    fit_um_proj = LinearRegression().fit(fitX, fitY_um, sample_weight=mf_dmo_proj)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(fitX, fitY_dmo, label="DMO")
    ax.plot(fitX, fitY_um, label="UM")

    predict_dmo = fit_dmo_proj.predict(fitX)
    predict_um = fit_um_proj.predict(fitX)

    ax.plot(fitX, predict_dmo, linestyle="dotted", label="dmo fit")
    ax.plot(fitX, predict_um, linestyle="dashed", label="um fit")

    ax.legend()
 
    savefig_pngpdf(fig, fname)


    #fit_dmo_proj = fit_loglog(mf_dmo_proj, mf_bins)
    #fit_um_proj = fit_loglog(mf_um_proj, mf_bins)

    out = {
             "DMO (Total Massfunction)": fit_dmo,
             "UM (Total Massfunction)": fit_um,
            f"DMO ({rmin} < r_2d <= {rmax})": fit_dmo_proj,
            f"UM ({rmin} < r_2d <= {rmax})": fit_um_proj,
    }

    df = fits_to_df(out)

    savedf(df, fname + ".csv")    

    test_x = np.asarray((9.00)).reshape(1,-1)

    ratio_total = 10**(fit_um.predict(test_x)) / 10**(fit_dmo.predict(test_x))
    ratio_proj = 10**(fit_um_proj.predict(test_x)) / 10**(fit_dmo_proj.predict(test_x))

    logger.info(f"Universe machine ratio to DMO (Total){ratio_total} %")
    logger.info(f"Universe machine ratio to DMO ({rmin} < r_2d <= {rmax} ){ratio_proj} %")

if __name__ == "__main__":
    main()