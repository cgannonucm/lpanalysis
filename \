#!/usr/bin/env python3
# sigmasub.py : Calculates variables related to sigma_sub (sigma_sub, f_s) for both galacticus and symphony

import numpy as np
import h5py
import pandas as pd

from subscript.defaults import ParamKeys
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import nfilter_halos, nfilter_project2d, nfilter_range, nfilter_subhalos, nfilter_subhalos_valid
from subscript.wrappers import gscript, gscript_proj, multiproj, freeze

from symutil import symphony_to_galacticus_hdf5

def sigsub_var_N0(mass_bounds:tuple[float,float],alpha:float,m0:float,numerical = False,divs:int = None) -> float:
    """
    mass_bounds - the bounds at which counting starts and stops\n
    alpha - Eponent of the subhalo mass function \n
    divs - number of divisions to use when integrating \n
    m0 - Pivot mass \n
    numerical - If true uses numerical integration\n
    """
    if numerical is True and divs == None: raise Exception("If using numerical integration bin divisions must be specified")
    (m_start,m_end) = mass_bounds
    if numerical is True:
        #n0_int = integrate_with_log_spacing(lambda m: m**alpha,m_start,m_end,divs)
        raise NotImplementedError()
    if numerical is False:
        # Integral m^alpha = 1/(alpha+1) * m^(alpha + 1)
        n0_int = (1/(alpha + 1)) * (m_end**(alpha + 1) - m_start**(alpha + 1))
    N0 =  n0_int / (m0**(alpha + 1))
    return N0


def sigsub_var_M0(mass_bounds_extrap:tuple[float,float],alpha:float,m0:float,numerical = False,divs:int = None) -> float:
    """
    Gets the M0 constant for the mass extrapolation \n
    mass_bounds_extrap - The mass bounds for where the extrapolation starts and stops\n
    alpha - Exponent of the subhalo mass function \n
    divs - number of divisions to use when integrating \n
    m0 - Pivot mass \n
    numerical - If true uses numerical integration\n
    """
    #M0 integral
    if numerical is True and divs == None: raise Exception("If using numerical integration bin divisions must be specified")
    extrapm_start, extrapm_end = mass_bounds_extrap
    if numerical is True:
        #M0_int = integrate_with_log_spacing(lambda m: m**(alpha + 1),extrapm_start,extrapm_end,divs)
        raise NotImplementedError
    if numerical is False:
        #Integral m**(alpha + 1) = 1/(alpha + 2) * m ^ (alpha + 2)
        M0_int = (1/(alpha + 2)) * (extrapm_end**(alpha + 2) - extrapm_start**(alpha + 2))
    M0 =  M0_int / (m0 ** (alpha + 1))
    return M0

def sigsub_var_sigma_sub(n:float,N0:float,rmin:float, rmax:float) -> float:
    """
    Returns sigma_sub parameter from equations above \n
    n - number of subhalos within apeture radius \n
    N0 - see equations above \n
    r_ap - The radius of the apeture \n
    """
    dnda = n / (np.pi * (rmax**2 - rmin**2))
    sigma_sub = dnda / N0
    return sigma_sub

def sigsub_var_M_extrap(sigma_sub:float,M0:float) -> float:
    """
    Returns M_extrap from the equations above, see equations for definitions of all variables
    """
    return sigma_sub * M0

@gscript_proj
def get_sigmsub(gout, normvector, mmin, mmax, rmin, rmax, alpha, mpivot=1E8, **kwargs):
    """
    Returns an array of $\Sigma_{sub}$, $f_s \cdot \Sigma_{sub}$ and $f_s$
    alpha - exponent of the subhalo mass function
    mpivot - pivot mass (same as Gilman (2020))
    """
    nfilter_proj      = nfilter_project2d(gout, normvector=normvector, rmin=rmin, rmax=rmax)
    nfilter_mass_evo  = nfilter_range(gout, mmin, mmax, key=ParamKeys.mass_bound)
    nfilter_mass_uevo = nfilter_range(gout, mmin, mmax, key=ParamKeys.mass_basic)

    nuevo  = nodecount(gout, nfilter=(kwargs["nfilter"] & nfilter_proj & nfilter_mass_uevo))
    nevo   = nodecount(gout, nfilter=(kwargs["nfilter"] & nfilter_proj & nfilter_mass_evo ))

    N0 = sigsub_var_N0((mmin, mmax), alpha, mpivot, numerical=False)

    sigsub = sigsub_var_sigma_sub(nuevo, N0, rmin, rmax)
    fs = nevo / nuevo

    return sigsub, fs * sigsub, fs, nevo

def macro_sigmasub():

    pass

def main():
    fname = "sigmasub.hdf5"
    path_gout = "data/galacticus/um_update/dmo/dmo.hdf5"
    path_symphony = "data/symphony/SymphonyGroup/"

    gout = h5py.File(path_gout)
    symout = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    normvector = np.identity(3)
    alpha = -1.94

    nfilter = nfilter_subhalos_valid(None, 1E8, 1E13, ParamKeys.mass_basic)
    #nfilter = nfilter_subhalos(None)
    nfilter_sym = nfilter_subhalos_valid(None, 1E9, 1E13, ParamKeys.mass_basic)

    print(get_sigmsub(gout,normvector=normvector, mmin=1E8, mmax=1E9, rmin=1E-2, rmax=2E-2, alpha=-1.93, nfilter=nfilter, summarize=True))
    print(get_sigmsub(symout,normvector=normvector, mmin=1E9, mmax=1E10, rmin=5E-2, rmax=10E-2, alpha=-1.93, nfilter=nfilter_sym, summarize=True))

    mh, z = nodedata(symout, [ParamKeys.mass_basic, ParamKeys.z_lastisolated], nfilter=nfilter_halos(None), summarize=True)
    print(np.log10(mh))
    print(z)

    pass

if __name__ == "__main__":
    main()
