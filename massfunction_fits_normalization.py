#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from massfunction_fits import fits_to_df

from plotting_util import savedf

def fit_csv_mhz(
                df:pd.DataFrame, 
                key_mh,
                key_z, 
                key_tofit,
                logx=False,
                logy=False
               ):  
    mh = df[key_mh] / 1E13
    z  = df[key_z] + 0.5

    fitX = pd.DataFrame(
                        {
                         "hm": mh, 
                         "z": z
                        }
                       )
    fitY = df[key_tofit]

    if logx:
        fitX = np.log10(fitX)
    if logy:
        fitY = np.log10(fitY)

    return LinearRegression().fit(fitX,  fitY)

def main(): 
    fname = "masfunction_normalization_fits.csv"
    path_csv = "out/csv/summary_massfunction_fits.csv" 
    path_csv_um = "out/csv/summary_massfunction_fits_um.csv" 

    summary_scaling = pd.read_csv(path_csv)
    summary_scaling_um = pd.read_csv(path_csv_um)

    key_mh =  "Halo Mass [M_\odot]"
    key_z = "z"
    key_tofit = "x_0"
    
    fit_scaling = fit_csv_mhz(
                              summary_scaling,
                              key_mh=key_mh,
                              key_z=key_z,
                              key_tofit=key_tofit,
                              logx=True,
                              logy=False 
                             ) 
                    
    fit_scaling_um = fit_csv_mhz(
                                 summary_scaling_um,
                                 key_mh=key_mh,
                                 key_z=key_z,
                                 key_tofit=key_tofit,
                                 logx=True,
                                 logy=False 
                                ) 
    df = fits_to_df(
                    {  
                     "Scaling (galacticus)": fit_scaling,
                     "Scaling (galacticus, um)": fit_scaling_um
                    }
                   )

    print(fit_scaling.intercept_)
    print(fit_scaling_um.intercept_)

    factor = 10**(fit_scaling_um.intercept_) / 10**(fit_scaling.intercept_)
    print(f"Universe machine ratio to DMO {factor:.2f} %")
    savedf(df, fname) 

if __name__ == "__main__":
    main()