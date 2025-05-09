#!/usr/bin/env python
import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from subscript.scripts.histograms import bin_avg
from subscript.defaults import ParamKeys
from subscript.scripts.histograms import massfunction
from subscript.scripts.nfilters import nfilter_subhalos_valid, nfilter_most_massive_progenitor
from subscript.wrappers import freeze
from subscript.scripts.nodes import nodedata

from summary import HDF5Wrapper
from plotting_util import savedf


def fits_to_df(fits:dict[str,LinearRegression])->pd.DataFrame:
    l = len(fits)

    out_dict = {}

    for n, coef in enumerate(fits[fits.__iter__().__next__()].coef_):
        out_dict[f"k_{n}"] = np.zeros(l)

    out_dict["x_0"] = np.zeros(l)
    out_dict["label"] = ["" for i in range(l)]
    
    for n,(label, fit) in enumerate(fits.items()): 
        for i, coef in enumerate(fit.coef_):
            out_dict[f"k_{i}"][n] = coef 

        out_dict["x_0"][n] = fit.intercept_
        out_dict["label"][n] = label

    
    return pd.DataFrame(out_dict)

def fit_loglog_massfunction(massfunction, massfunction_bins):
    x = np.log10((bin_avg(massfunction_bins))).reshape(-1, 1)
    y = np.log10(massfunction)
    fit = LinearRegression().fit(x, y, sample_weight=massfunction)    

    return fit

def fit_multihalos(summaryd, key_tofit, key_tofit_bins, key_id, key_mhalo, key_z):
    fit_dict = {_id:fit_loglog_massfunction(summaryd[key_tofit][n], summaryd[key_tofit_bins][n]) for n,_id in enumerate(summaryd[key_id])}
    out_df = fits_to_df(fit_dict) 
    mh, z = summaryd[key_mhalo], summaryd[key_z]  
    out_df.insert(0, "Halo Mass [M_\odot]", mh)
    out_df.insert(0, "z", z)
    return out_df

def main():
    fname_scaling = "summary_massfunction_fits.csv"
    fname_scaling_um = "summary_massfunction_fits_um.csv"

    KEY_MASSFUNCTION_HDF5 = "massfunction (evolved) (mean)/out0"
    KEY_MASSFUNCTION_BINS_HDF5 = "massfunction (evolved) (mean)/out1"
    KEY_Z_HDF5 = "z (mean)/out0"
    KEY_MH_HDF5 = "halo mass (mean)/out0"
    KEY_ID_HDF5 = "id/out0"

    path_summary_scaling = "out/hdf5/summary_scaling.hdf5"
    path_summary_scaling_um = "out/hdf5/scaling_um.hdf5"

    summary_scaling = HDF5Wrapper(h5py.File(path_summary_scaling))
    summary_scaling_um = HDF5Wrapper(h5py.File(path_summary_scaling_um))

    out_df = fit_multihalos(
                            summary_scaling, 
                            key_tofit=KEY_MASSFUNCTION_HDF5,
                            key_tofit_bins=KEY_MASSFUNCTION_BINS_HDF5,
                            key_id=KEY_ID_HDF5, 
                            key_mhalo=KEY_MH_HDF5,
                            key_z=KEY_Z_HDF5
                           )

    out_df_um = fit_multihalos(
                            summary_scaling_um, 
                            key_tofit=KEY_MASSFUNCTION_HDF5,
                            key_tofit_bins=KEY_MASSFUNCTION_BINS_HDF5,
                            key_id=KEY_ID_HDF5, 
                            key_mhalo=KEY_MH_HDF5,
                            key_z=KEY_Z_HDF5
                           )
 
    
    savedf(out_df, fname_scaling)
    savedf(out_df_um, fname_scaling_um)


if __name__ == "__main__":
    main()
