#!/usr/bin/env python

import h5py
import numpy as np
import symlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any, Callable
import os

from gstk.common.constants import GParam, CPhys
from gstk.util import util_binedgeavg
from gstk.io import io_importgalout
from gstk.scripts.common import ScriptProperties, script, NodeSelector
from gstk.scripts.selection import script_selector_subhalos_valid, script_selector_halos, script_select_nodedata, script_selector_tree, script_selector_annulus, script_selector_all
from gstk.scripts.massfunction import script_massfunction
from gstk.scripts.meta import script_eachtree
from gstk.scripts.selection import script_selector_all, script_selector_most_massive_projenitor
from gstk.scripts.misc import script_treecount
from gstk.util import TabulatedGalacticusData
from gstk.scripts.meta import script_eachtree
from gstk.tabulatehdf import tabulate_nodes, get_galacticus_outputs

from plotting_util import *
from symutil import symphony_to_galacticus_dict

def plot_most_massive_progenitor_property(fig, ax:Axes, gout: h5py.File, 
                                            toplot_x: (str | Callable), toplot_y: (str | Callable),
                                            error_plot = True, kwargs_script = None, 
                                            kwargs_plot = None, kwargs_fill = None):
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    kwargs_script = {} if kwargs_script is None else kwargs_script
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot    

    outputs = get_galacticus_outputs(gout)

    get_x = (lambda o, **k: o[toplot_x][k["selector_function"](o)]) if isinstance(toplot_x, str) else toplot_x
    get_y = (lambda o, **k: o[toplot_y][k["selector_function"](o)]) if isinstance(toplot_y, str) else toplot_y

    x, y_avg, y_std = np.zeros(outputs.shape), np.zeros(outputs.shape), np.zeros(outputs.shape)

    for n,i in enumerate(outputs): 
        gouti = tabulate_nodes(gout, i)          
        mmp_select = lambda *a, **k: script_selector_most_massive_projenitor(gouti)
        xarr, yarr = get_x(gouti, selector_function=mmp_select), get_y(gouti, selector_function=mmp_select)
        x[n], y_avg[n], y_std[n] = np.mean(xarr), np.mean(yarr), np.std(yarr)
    
    if error_plot: 
        ax.errorbar(x, y_avg, y_std, **(KWARGS_DEF_ERR | kwargs_plot))
        return

    ax.plot(x, y_avg, **(KWARGS_DEF_PLOT | kwargs_plot))
    ax.fill_between(x,y_avg - y_std, y_avg + y_std, **(KWARGS_DEF_FILL | kwargs_fill))


def main():
    fname = "um_evolution_mass"
    path_file = "data/galacticus/um_update/um-multiz/multiz.hdf5" 

    gout = h5py.File(path_file)
    outputs = get_galacticus_outputs(gout)

    set_plot_defaults()

    fig, axs = plt.subplots(figsize=(18,12), ncols=2, nrows=2)
    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    # ax0
    plot_most_massive_progenitor_property(fig,ax0,gout,GParam.Z_LASTISOLATED,
                                            GParam.MASS_BASIC, error_plot=True,
                                            kwargs_plot=dict(label="Halo"))
    plot_most_massive_progenitor_property(fig,ax0,gout,GParam.Z_LASTISOLATED,
                                            GParam.SPHERE_MASS_STELLAR, error_plot=True,
                                            kwargs_plot=dict(label="Central Galaxy"))

    ax0.set_yscale("log")

    ax0.set_xlabel("z")
    ax0.set_ylabel(r"M [M$_\odot$]")

    ax0.set_xlim(*ax0.get_xlim())
    ax0.set_ylim(*ax0.get_ylim())

    #ax.errorbar((0,), (0,), (0,), color="green")

    ax0.legend()

    #ax1 
    _mh0, _ms0 = script_select_nodedata(gout, script_selector_halos, (GParam.MASS_BASIC, GParam.SPHERE_MASS_STELLAR))
    mh0, ms0 = np.mean(_mh0), np.mean(_ms0)
    mhratio0 = lambda o, **k: o[GParam.MASS_BASIC][k["selector_function"](o, **k)] / _mh0
    msratio0 = lambda o, **k: o[GParam.SPHERE_MASS_STELLAR][k["selector_function"](o, **k)] / _ms0

    plot_most_massive_progenitor_property(fig,ax1,gout,GParam.Z_LASTISOLATED,mhratio0, error_plot=True)
    plot_most_massive_progenitor_property(fig,ax1,gout,GParam.Z_LASTISOLATED,msratio0, error_plot=True)

    ax1.set_yscale("log")

    ax1.set_xlabel("z")
    ax1.set_ylabel(r"M / M$_0$")

    #ax2 

    plot_most_massive_progenitor_property(fig,ax2,gout,GParam.MASS_BASIC,
                                            GParam.SPHERE_MASS_STELLAR, error_plot=True,
                                            kwargs_plot=dict(color="tab:green"))

    ax2.loglog()

    ax2.set_xlabel(r"M$_h$ [M$_\odot$]")
    ax2.set_ylabel(r"M$_\star$ [M$_\odot$]")

    #ax3
    def mhmstarratio(o, **k):
        mh, ms = script_select_nodedata(o, k["selector_function"], (GParam.MASS_BASIC, GParam.SPHERE_MASS_STELLAR))
        # print(mh, ms)
        return ms / mh

    plot_most_massive_progenitor_property(fig,ax3,gout,GParam.Z_LASTISOLATED,
                                            mhmstarratio, error_plot=True, 
                                            kwargs_plot=dict(color="tab:green"))

    ax3.set_yscale("log")

    ax3.set_xlabel("z")
    ax3.set_ylabel(r"M$_\star$ / M$_h$")



    savefig(fig, fname + ".png")
    savefig(fig, fname + ".png")

if __name__ == "__main__":
    main()

