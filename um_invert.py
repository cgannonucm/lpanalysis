#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any
from scipy.interpolate import interp1d

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
    mhspace = np.logspace(10, 15, 1000)
    ms = um_smhm_mstar(mhspace, 0.5)
    interp = interp1d(ms, mhspace)
    
    # Approximate mean stellar mass from SLACS
    # https://iopscience.iop.org/article/10.3847/1538-4357/aa9794/pdf
    log_m_s_slacs = [
    11.53,
    11.03,
    11.31,
    11.71,
    11.23,
    11.13,
    11.34,
    11.38,
    11.19,
    11.05,
    12.08,
    11.77,
    11.46,
    11.68,
    11.22,
    10.99,
    11.07,
    11.22,
    11.16,
    11.12,
    11.23,
    11.29,
    11.44,
    11.31,
    11.21,
    10.74,
    11.09,
    10.92,
    11.74,
    10.78,
    11.32,
    11.45,
    11.25,
    11.74,
    11.30,
    11.26,
    11.55,
    11.39,
    11.68,
    11.32
    ]
    ms_mean = np.mean(log_m_s_slacs)
    print(ms_mean)
    
    mh = interp(10**(ms_mean))
    print(np.log10(mh))
    
    pass

if __name__ == "__main__":
    main()

