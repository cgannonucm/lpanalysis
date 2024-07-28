#!/usr/bin/env python

import numpy as np
import symlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any
import os

from gstk.common.constants import GParam, CPhys
from gstk.util import util_binedgeavg
from gstk.io import io_importgalout
from gstk.scripts.common import ScriptProperties, script, NodeSelector
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus
from gstk.scripts.massfunction import script_massfunction
from gstk.scripts.meta import script_eachtree
from gstk.util import TabulatedGalacticusData

from plotting_util import *
from symutil import symphony_to_galacticus_dict


@script(**ScriptProperties.KEY_KWARG_DEF)
def script_massfunction2(data:TabulatedGalacticusData, selector_function:NodeSelector,
                         mrange:tuple[int,int], bincount:int, normalize_treecount=False, normalize_bin_width = True,
                         bin_logspace=True, useratio=False, key_mass=None, **kwargs): 
    """Improved massfunction script. Use instead of script mass function, by default does not normalize histogram to number of trees"""
    mass = script_select_nodedata(data, selector_function, [key_mass])[0]

    if useratio:
        pass
        #mh = script_select_nodedata(data, selector_function & , [key_mass])[0] 
        #mass /= 

    bins = np.geomspace(*mrange, bincount) if bin_logspace else np.linspace(*mrange, bincount)

    node_n, node_m = np.histogram(mass, bins=bins) 



def Compute_SHMF_N_body_Symphony(folder, halo, iSnap, binMin, binMax, binCount=20,    \
                                 withinRvirOnly=True, useRatio=True, useLogBins=True, \
                                 usePeakMass=False):
    # iSanp         : select the i-th snapshot of the simulations
    # withinRvirOnly: include only the subhalos with the host's virial radius
    # useRatio      : compute the subhalo mass function in terms of the mass ratio M_sub/M_host
    # useLogBins    : use logarithmic bins
    # usePeakMass   : use peak mass (infall mass) instead of the bound mass

    sim_dir     = folder+halo
    
    h, hist     = symlib.read_subhalos(folder+halo)
    params      = symlib.simulation_parameters(sim_dir)
    hubble      = params['H0']/100.0
    Mp          = params['mp']/hubble
    
    Mhost       = h['mvir' ][0 ,iSnap]/hubble
    RvirHost    = h['rvir' ][0 ,iSnap]/hubble/1.0e3
    posHost     = h['x'    ][0 ,iSnap]/hubble/1.0e3
    posSub      = h['x'    ][1:,iSnap][h['ok'][1:,iSnap]]/hubble/1.0e3
    MboundSub   = h['mvir' ][1:,iSnap][h['ok'][1:,iSnap]]/hubble
    # Maximum mass of subahlo in the history (treated as infall mass in Galacticus).
    MpeakSub    = hist['mpeak'][1:][h['ok'][1:,iSnap]]/hubble
    
    #print(RvirHost)
    
    # Distance to the host center
    Distance    = np.sqrt(np.sum((posSub-posHost[None,:])**2, axis=1))
    
    if (withinRvirOnly):
        indexSelect = np.logical_and(Distance > 0.0, Distance <= RvirHost)
    else:
        indexSelect =                Distance > 0.0
    
    if (usePeakMass):
        MsubSelect  = MpeakSub[indexSelect]
    else:
        MsubSelect  = MboundSub[indexSelect]
    
    if (useRatio):
        MsubSelect = MsubSelect/Mhost
    
    # Check whether to use logarithmic bin.
    if (useLogBins):
        massBins   = np.geomspace(binMin, binMax, binCount+1)
        binCenter  = np.sqrt(massBins[:-1]*massBins[1:])
    else:
        massBins   = np.linspace (binMin, binMax, binCount+1)
        binCenter  = 0.5*(massBins[:-1]+massBins[1:])

    hist, binEdge = np.histogram(MsubSelect,bins=massBins,range=(binMin,binMax))
    binWidth      = massBins[1:]-massBins[:-1]
    hist          = hist/binWidth

    return binCenter, hist

def massfunction_symphony_dir(haloFolder, bincount, mrange, withinRvirOnly=True,useRatio=False, useLogBins=True,usePeakMass=False):
    haloNames  = os.listdir(haloFolder)
    NHostNbody = len(haloNames)

    M_sub_Nbody         = np.zeros((NHostNbody, bincount))
    dNdM_sub_Nbody      = np.zeros((NHostNbody, bincount))

    for iH in range(NHostNbody):
        M_sub_Nbody[iH], dNdM_sub_Nbody[iH] = Compute_SHMF_N_body_Symphony(                                    
                                                                            haloFolder, haloNames[iH], -1,      
                                                                            binMin=mrange[0], binMax=mrange[1],     
                                                                            binCount=bincount,                     
                                                                            withinRvirOnly=withinRvirOnly,                
                                                                            useRatio=useRatio,                      
                                                                            useLogBins=useLogBins,                    
                                                                            usePeakMass=usePeakMass,                    
                                                                    )
    return M_sub_Nbody, dNdM_sub_Nbody

def plot_massfunction(fig, ax, data, mrange, bincount, selector_function=None, plot_kwargs=None,
                        script_kwargs=None, plot_dndlnm=False, useratio = True):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    script_kwargs = {} if script_kwargs is None else script_kwargs
    selector_function = script_selector_subhalos_valid if selector_function is None else selector_function


    mhost = np.mean(script_select_nodedata(data,script_selector_halos, [GParam.MASS_BASIC])[0])

    mrange_scale = 1
    if useratio:
        mrange_scale = mhost

    _mrange = np.asarray(mrange) * mrange_scale

    hist, mbins = script_massfunction(data, selector_function=selector_function,mrange=_mrange,bincount=bincount, **script_kwargs)

    plotx = util_binedgeavg(mbins) / mrange_scale

    ploty = hist

    if plot_dndlnm:
        ploty = hist * util_binedgeavg(mbins)


    ax.plot(plotx, ploty, **(KWARGS_DEF_PLOT | plot_kwargs))

def plot_massfunction_scatter(fig, ax:Axes, data, mrange, bincount, selector_function=None, plot_kwargs=None,
                                script_kwargs=None, plot_dndlnm=False, useratio = True, error_plot=False, 
                                factor=1, nsigma = 1):

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    script_kwargs = {} if script_kwargs is None else script_kwargs
    selector_function = script_selector_subhalos_valid if selector_function is None else selector_function

    mhost = np.mean(script_select_nodedata(data,script_selector_halos, [GParam.MASS_BASIC])[0])

    mrange_scale = 1

    if useratio:
        mrange_scale = mhost

    _mrange = np.asarray(mrange) * mrange_scale

    sub_dndm, sub_m = script_eachtree(data, script=script_massfunction, 
                          selector_function=selector_function, mrange=_mrange, 
                          bincount=bincount,normalize_treecount=False, **script_kwargs)
    
    sub_dndm, sub_m = np.asarray(sub_dndm), np.asarray(sub_m) 

    sub_m, sub_dndm_avg, sub_dndm_std = np.mean(sub_m, axis=0), np.mean(sub_dndm, axis=0), np.std(sub_dndm, axis=0)

    plotx = util_binedgeavg(sub_m) / mrange_scale

    ploty, ploty_std = sub_dndm_avg * factor, sub_dndm_std * factor

    if plot_dndlnm:
        ploty       = sub_dndm_avg * util_binedgeavg(sub_m) * factor
        ploty_std   = sub_dndm_std * util_binedgeavg(sub_m) * factor * nsigma

    ploty_min, ploty_max = ploty - ploty_std, ploty + ploty_std

    if error_plot:
        ax.errorbar(plotx, ploty, ploty_std, **(KWARGS_DEF_ERR | plot_kwargs))
        return 

    ax.fill_between(plotx, ploty_min, ploty_max, **(KWARGS_DEF_FILL | plot_kwargs))


def plot_massfunction_symphony(fig, ax:Axes, haloFolder, bincount, mrange,
                                withinRvirOnly=True,useRatio=False, useLogBins=True,usePeakMass=False,
                                plot_kwargs=None, plot_dnlnm = False, mhost=1E13):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    mrange_scale = 1
    if useRatio:
        mrange_scale = mhost

    _mrange = np.asarray(mrange) * mrange_scale   

    M_sub_Nbody, dNdM_sub_Nbody = massfunction_symphony_dir(haloFolder, bincount, _mrange, withinRvirOnly,False, useLogBins,usePeakMass)

    plot_x, plot_y, plot_y_std = np.mean(M_sub_Nbody,axis=0), np.mean(dNdM_sub_Nbody,axis=0), np.std(dNdM_sub_Nbody,axis=0) 
    
    if plot_dnlnm:
        plot_x, plot_y, plot_y_std = np.mean(M_sub_Nbody,axis=0), np.mean(M_sub_Nbody * dNdM_sub_Nbody,axis=0), np.std(M_sub_Nbody * dNdM_sub_Nbody,axis=0) 

    if useRatio:
        plot_x = plot_x / mrange_scale

    ax.errorbar(plot_x,plot_y,plot_y_std, **(KWARGS_DEF_ERR | plot_kwargs))


def main():
    fname = "massfunction"
    path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5" 
    path_symphony = "data/symphony/SymphonyGroup/"

    filend = io_importgalout(path_file)[path_file] 
    mres = 1E9

    #script_test(filend)
    
    sym_nodedata = symphony_to_galacticus_dict(path_symphony, iSnap=203)

    plot_dndlnm = True
    useratio = True

    scale = 1E13

    if useratio:
        scale = 1
    
    mrange = np.asarray((1E8 / 1E13,1)) * scale
    mrange_sym = np.asarray((mres / 10**(13.0), 1)) * scale
    bincount = 20    

    set_plot_defaults()

    fig, ax = plt.subplots(figsize=(9,6))

    #plot_massfunction_symphony(fig, ax, path_symphony, bincount, mrange_sym,
    #                            plot_dnlnm=plot_dndlnm, useRatio=useratio, 
    #                            plot_kwargs=dict(label="Symphony (Group)", zorder=2))

    plot_massfunction_scatter(fig, ax, sym_nodedata, mrange_sym, bincount, plot_dndlnm=plot_dndlnm, 
                                useratio=useratio,
                                plot_kwargs=dict(label="Symphony", zorder=2), 
                                error_plot=True)   

    #plot_massfunction_scatter(fig, ax, sym_nodedata, mrange_sym, bincount, plot_dndlnm=plot_dndlnm, 
    #                            useratio=useratio,
    #                            plot_kwargs=dict(label="Symphony", zorder=2), 
    #                            error_plot=True, nsigma=2)   

    plot_massfunction(fig, ax, filend, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                useratio=useratio, 
                                plot_kwargs=dict(label="Galacticus", color="black", zorder=3))

    plot_massfunction_scatter(fig, ax, filend, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                useratio=useratio,
                                plot_kwargs=dict(label="Galacticus (scatter)", zorder=1))

    plot_massfunction_scatter(fig, ax, filend, mrange, bincount, plot_dndlnm=plot_dndlnm, 
                                useratio=useratio, nsigma=2,
                                plot_kwargs=dict(label="Galacticus (scatter)", zorder=1))






    plt.loglog()

    ax.set_xlabel(r"$m / M_{h}$")
    ax.set_ylabel(r"$\frac{dN}{d\ln m}$")

    ax.legend()

    savefig(fig, fname + ".png")
    savefig(fig, fname + ".pdf")




if __name__ == "__main__":
    main()


