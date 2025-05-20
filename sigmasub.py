#!/usr/bin/env python3
# sigmasub.py : Calculates variables related to sigma_sub (sigma_sub, f_s) for both galacticus and symphony

import numpy as np
import h5py
import pandas as pd

from sklearn.linear_model import LinearRegression
from subscript.defaults import ParamKeys
from subscript.macros import tabulate_trees
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import nfilter_halos, nfilter_project2d, nfilter_range, nfilter_subhalos, nfilter_subhalos_valid, nfilter_virialized, nfilter_all, nfand
from subscript.wrappers import gscript, gscript_proj, multiproj, freeze
from subscript.scripts.histograms import massfunction, bin_avg

from symutil import symphony_to_galacticus_hdf5
from plotting_util import savedf
from massfunction_fits import fit_loglog_massfunction


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
    r"""
    Returns an array of $\Sigma_{sub}$, $f_s \cdot \Sigma_{sub}$ and $f_s$
    alpha - exponent of the subhalo mass function
    mpivot - pivot mass (same as Gilman (2020))
    """
    nfilter_proj      = nfilter_project2d(gout, normvector=normvector, rmin=rmin, rmax=rmax)
    nfilter_mass_evo  = nfilter_range(gout, mmin, mmax, key=ParamKeys.mass_bound)
    nfilter_mass_uevo = nfilter_range(gout, mmin, mmax, key=ParamKeys.mass_basic)
    nfilter_v         = nfilter_virialized(gout)

    nuevo  = nodecount(gout, nfilter=(kwargs["nfilter"] & nfilter_proj & nfilter_mass_uevo & nfilter_v))
    nevo   = nodecount(gout, nfilter=(kwargs["nfilter"] & nfilter_proj & nfilter_mass_evo & nfilter_v))

    N0 = sigsub_var_N0((mmin, mmax), alpha, mpivot, numerical=False)

    sigsub = sigsub_var_sigma_sub(nuevo, N0, rmin, rmax)
    fs = nevo / nuevo
    fm = fs**(-1/(alpha+1))

    return sigsub, fs * sigsub, fs, fm, nevo


def fit_mf_proj(gout, normvector, mmin, mmax, mbins, rmin, rmax, **kwargs):
    nfilter_proj      = nfilter_project2d(None, normvector=normvector, rmin=rmin, rmax=rmax)
    nfilter_evo       = nfand(nfilter_subhalos_valid(None, mmin, mmax, key_mass=ParamKeys.mass_basic), nfilter_proj)
    nfilter_uevo      = nfand(nfilter_subhalos_valid(None, mmin, mmax, key_mass=ParamKeys.mass_bound), nfilter_proj)

    _mbins = np.geomspace(mmin, mmax, mbins)

    mf_uevo = massfunction(gout, key_mass=ParamKeys.mass_basic, bins=_mbins, nfilter=nfilter_uevo,  summarize=True)
    mf_evo  = massfunction(gout, key_mass=ParamKeys.mass_bound, bins=_mbins, nfilter=nfilter_evo ,  summarize=True)


    fit_uevo, fit_evo = fit_mf(*mf_uevo), fit_mf(*mf_evo)

    return fit_uevo, fit_evo

def fit_mf(massfunction, massfunction_bins):
    x = np.log10((bin_avg(massfunction_bins))).reshape(-1, 1)
    y = np.log10(massfunction)
    return LinearRegression().fit(x, y)

def main():
    fname = "sigmasub.csv"
    #path_gout = "data/galacticus/um_update/dmo/dmo.hdf5"
    #path_gout = "data/galacticus/summary/dmo.hdf5"
    path_gout = "data/galacticus/mh1E13z05/dmo.hdf5"

    path_symphony = "data/symphony/SymphonyGroup/"

    from subscript.tabulatehdf5 import tabulate_trees


    gouth5 = h5py.File(path_gout)
    tab = tabulate_trees(gouth5)

    symout = symphony_to_galacticus_hdf5(path_symphony, iSnap=203)

    normvector = np.identity(3)
    alpha = -1.93

    gal_mmin, gal_mmax = 10**(8.0) , 10**(9)
    gal_rmin, gal_rmax = 1E-2, 2E-2

    nfilter = nfilter_all(None)

    labels_mean = ["sigsub", "fs * sigsub", "fs", "fm", "nevo"]
    labels_std  = [l + " [std]" for l in labels_mean]

    print("Galacticus")
    gout = get_sigmsub(gouth5,normvector=normvector, mmin=gal_mmin, mmax=gal_mmax, rmin=gal_rmin, rmax=gal_rmax, alpha=alpha, nfilter=nfilter, summarize=True, statfuncs=(np.mean, np.std))
    print(labels_mean)
    print(gout[0])
    print(labels_std)
    print(gout[1])

    print("----")

    print("Symphony")
    sout = get_sigmsub(symout,normvector=normvector, mmin=1E9, mmax=1E10, rmin=5E-2, rmax=10E-2, alpha=alpha, nfilter=nfilter, summarize=True, statfuncs=(np.mean, np.std))
    print(labels_mean)
    print(sout[0])
    print(labels_std)
    print(sout[1])

    mh, z = nodedata(symout, [ParamKeys.mass_basic, ParamKeys.z_lastisolated], nfilter=nfilter_halos(None), summarize=True)
    print("log10(M_h)=",np.log10(mh))
    print("z=",z)

    fits_gout = fit_mf_proj(gouth5, normvector=np.identity(3)[2], mmin=1E8, mmax=1E9, mbins=3, rmin=1E-2, rmax=5E-2)
    fits_sout = fit_mf_proj(symout, normvector=np.identity(3)[2], mmin=1E9, mmax=1E10, mbins=3, rmin=0, rmax=1E-1)

    #fits_sout = fit_mf_proj(symout, normvector=np.identity(3)[2], mmin=1E8, mmax=1E10, mbins=10, rmin=1E-2, rmax=2E-2)

    d = {"filepath": np.asarray([path_gout, path_symphony])}

    for n, _ in enumerate(sout[0]):
        d[labels_mean[n]] = np.asarray([gout[0][n], sout[0][n]])
        d[labels_std[n]] = np.asarray([gout[1][n], sout[1][n]])

    d["alpha"]   = np.asarray((fits_gout[0].coef_, fits_sout[0].coef_)).flatten()
    d["alpha_b"] = np.asarray((fits_gout[1].coef_, fits_sout[1].coef_)).flatten()

    print(d)

    savedf(pd.DataFrame.from_dict(d), fname)

if __name__ == "__main__":
    main()
