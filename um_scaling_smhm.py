#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any
from colossus.cosmology import cosmology
from colossus.cosmology.cosmology import Cosmology

from gstk.common.constants import GParam, CPhys
from gstk.io import io_importgalout
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus, script_selector_all

from plotting_util import savefig_pngpdf, set_plot_defaults, KWARGS_DEF_PLOT

from gstk.util import util_binedgeavg


def um_smhm_scaling_param_def():
    # Scaling variables from universe machine table j1
    # https://arxiv.org/pdf/1806.07893     
    return dict(
                 e0     = -1.435, 
                 ea     = +1.831,
                 elna   = +1.368,        
                 ez     = -0.217,
                 m0     = +12.035,
                 ma     = +4.556,
                 mlna   = +4.417, 
                 mz     = -0.731, 
                 a0     = +1.963, 
                 aa     = -2.316,
                 alna   = -1.732,
                 az     = +0.178,
                 b0     = +0.482,
                 ba     = -0.841,
                 blna   = +0.000,
                 bz     = -0.471,
                 d0     = +0.411,                
                 da     = +0.000,
                 dlna   = +0.000,
                 dz     = +0.000,
                 g0     = -1.034,
                 ga     = -3.100,
                 glna   = +0.000,
                 gz     = -1.055, 
                )                
def um_smhm_scaling_param_fromfile(filepath:str):
    """
    Loads universe machine parameter information from a universe machine parameter .txt file
    Universe machine output can be downloaded here:
    https://halos.as.arizona.edu/UniverseMachine/DR1/umachine-dr1.tar.gz
    Once extracted, parameter files can be found in the umachine-dr1/data/smhm/params directory
    """
    # Code modified from https://bitbucket.org/pbehroozi/universemachine/src/main/
    # File scripts/gen_smhm.py
    param_file = open(filepath, "r")
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))

    if (len(param_list) != 20):
        print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
        return

    names = "e0 ea elna ez m0 ma mlna mz a0 aa alna az b0 ba bz d0 g0 ga gz chi2".split(" ")
    params = dict(zip(names, param_list))

    return params 

def um_smhm_scaling_fromdict(scaling_parameters:dict[str, float], vstr:str, prefix = "y"):
    ensurenotnone = lambda y: 0.0 if y is None else y
    return {
                prefix + "0"   : ensurenotnone(scaling_parameters.get(vstr + "0"  )),
                prefix + "a"   : ensurenotnone(scaling_parameters.get(vstr + "a"  )),
                prefix + "lna" : ensurenotnone(scaling_parameters.get(vstr + "lna")),
                prefix + "z"   : ensurenotnone(scaling_parameters.get(vstr + "z"  )),
    }

def um_smhm_scaling(z, y0, ya, ylna, yz):
    # Scaling function from universe machine apprendix equations j3 - j8
    # https://arxiv.org/pdf/1806.07893     
    a = 1 / (1 + z)
   
    return  (
             +y0 
             +ya
             *(
                 +a
                 -1
              ) 
             -ylna
             *np.log(a) 
             +yz
             *z
             ) 

def um_smhm_mstar(m, z, umparams = None):
    # Variable M_\star from universe machine, appendix j1
    # https://arxiv.org/pdf/1806.07893     
    
    if umparams is None:
        umparams = um_smhm_scaling_param_def()

    logm1    = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "m")) 
    loggamma = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "g")) 

    epsilon  = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "e")) 
    alpha    = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "a")) 
    beta     = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "b")) 
    delta    = um_smhm_scaling(z, **um_smhm_scaling_fromdict(umparams, "d")) 

    m1       = np.power(10,logm1   )
    gamma    = np.power(10,loggamma)


    x = np.log10(m / m1)

    log_msr = (
               +epsilon
               -np.log10(
                          +np.power(
                                     +10,
                                     -alpha 
                                     *x
                                    )
                          +np.power(
                                    +10,
                                    -beta
                                    *x
                                   )
                        )
               +gamma
               *np.exp  (
                          -0.5
                          *np.power(
                                     +x
                                     /delta
                                     ,2
                                    ) 
                        )
               )
              
    return (
            +np.power(
                    +10,
                    log_msr
                   )
            *m1
           ) 

def main():
    figname = "um_smhm"


    path_um_params = "data/umachine/umachine-dr1/data/smhm/params/smhm_med_params.txt"
    #path_gal = "data/galacticus/experimental/cg/universe-machine-smhm-test-finalMass.hdf5"
    path_gal = "data/galacticus/experimental/cg/universe-machine-smhm-test.hdf5"

    params = um_smhm_scaling_param_fromfile(path_um_params)

    gal_cg = io_importgalout(path_gal)[path_gal]
    
    hm = np.geomspace(1E10, 1E15, 100)
    zarr = np.linspace(0,10,11)
    zarr = np.asarray(((0.1, )))
    zarr[0] = 0.1

    set_plot_defaults()
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(27,6)) 

    # Universe Machine
    smz0 = um_smhm_mstar(hm, 0.1, umparams=params)
    pltsmz0 = util_binedgeavg(smz0)

    for z in zarr:
        sm = um_smhm_mstar(hm, z, umparams=params)
        # Make a cut where the slope goes negative
        valid = (sm[1:] - sm[:-1]) >= 0 
        plthm, pltsm = util_binedgeavg(hm)[valid], util_binedgeavg(sm)[valid]

        ax1.plot(np.log10(plthm), np.log10(pltsm), **KWARGS_DEF_PLOT, label=f"z={z:.1f}", alpha=0.8)

        ax2.plot(np.log10(plthm), np.log10(pltsm / plthm), **KWARGS_DEF_PLOT, label=f"z={z:.1f}", alpha=0.8)

        ax3.plot(np.log10(plthm), pltsm / pltsmz0[valid], **KWARGS_DEF_PLOT, label=f"z={z:.1f}", alpha=0.8)

    # Galacticus 
    gal_hm, gal_sm = script_select_nodedata(gal_cg, script_selector_halos, [GParam.MASS_BASIC,GParam.SPHERE_MASS_STELLAR])
    print(gal_hm, gal_sm)
    
    ax1.scatter(np.log10(gal_hm), np.log10(gal_sm / 20), alpha=1, color="black", zorder=20)

    #ax1.set_xlim(10, 15)
    #ax1.set_ylim(7, 12)

    ax1.set_xlabel(r"$\log_{10} (M_h)$")
    ax1.set_ylabel(r"$\log_{10} (M_\star)$")

    ax2.set_xlabel(r"$\log_{10} (M_h)$")
    ax2.set_ylabel(r"$\log_{10} (M_\star / M_h)$")

    ax3.set_xlabel(r"$\log_{10} (M_h)$")
    ax3.set_ylabel(r"$M_\star / M_{\star, z = 0.1}$")
    
    ax3.legend(ncol=2)

    savefig_pngpdf(fig, figname)

if __name__ == "__main__":
    main()

