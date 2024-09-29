import h5py
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
import os.path as path
import matplotlib.patheffects as pe
from typing import Callable

from subscript.macros import macro_write_out_hdf5
from subscript.wrappers import multiproj, freeze
from subscript.scripts.histograms import bin_avg


KWARGS_DEF_PLOT = dict(linewidth=5)

KWARGS_DEF_ERR = KWARGS_DEF_PLOT |  dict(capsize=5, markersize=8,
                                        path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

KWARGS_DEF_FILL = dict(zorder=1, alpha=0.5, color="tab:orange")

PATH_OUT = "out"
PATH_PLOTS = "plots"
PATH_CSV = "csv"
PATH_HDF5 = "hdf5"

PARAM_DEF_RRANGE_RVF = (0.015, 1)
PARAM_DEF_MRANGE = (1E8, 1E9)
PARAM_DEF_RANNULUS = (1E-2, 2E-2)
PARAM_DEF_ALPHA = -1.93

PARAM_KEY_N_PROJ_BOUND = "n_projected 1.00E-02[MPC] < r_2d < 2.00E-02[MPC] 1.0E+08 < M < 1.0E+09 mass_bound"
PARAM_KEY_N_PROJ_BOUND_SCATTER = "n_(projected,sd) 1.00E-02[MPC] < r_2d < 2.00E-02[MPC] 1.0E+08 < M < 1.0E+09 mass_bound"
PARAM_KEY_N_PROJ_INFALL = "n_projected 1.00E-02[MPC] < r_2d < 2.00E-02[MPC] 1.0E+08 < M < 1.0E+09 mass_infall"
PARAM_KEY_N_PROJ_INFALL_SCATTER = "n_(projected,sd) 1.00E-02[MPC] < r_2d < 2.00E-02[MPC] 1.0E+08 < M < 1.0E+09 mass_infall"

KEY_DEF_HALOMASS = "TreeMass [M_sol]"
KEY_DEF_Z = "z"

def set_plot_defaults():
    path_fonts = "/home/charles/research/lensing_perspective_accompaniment/fonts"
    fonts = font_manager.findSystemFonts(fontpaths=path_fonts)

    for font in fonts:
        font_manager.fontManager.addfont(font)

    mpl.rcParams.update({"font.family": "Source Sans 3"})

    mpl.rcParams.update({"font.size":20})

    mpl.rcParams.update({"xtick.major.width":3})
    mpl.rcParams.update({"xtick.major.size":6})

    mpl.rcParams.update({"xtick.minor.width":2})
    mpl.rcParams.update({"xtick.minor.size":3})

    mpl.rcParams.update({"ytick.major.width":3})
    mpl.rcParams.update({"ytick.major.size":6})

    mpl.rcParams.update({"ytick.minor.width":2})
    mpl.rcParams.update({"ytick.minor.size":3})

    mpl.rcParams.update({'axes.linewidth':2})

def savefig(fig, name):
    fig.savefig(path.join(PATH_OUT,PATH_PLOTS,name), bbox_inches="tight")

def savefig_pngpdf(fig, name):
    savefig(fig,name + ".png")
    savefig(fig,name + ".pdf")

def savedf(df, name):
    df.to_csv(path.join(PATH_OUT,PATH_CSV,name), index=False)

def savemacro(macro_out, name, notes=None, stamp_date=True):
    with h5py.File(path.join(PATH_OUT, PATH_HDF5, name), "w") as f:
        macro_write_out_hdf5(f, macro_out, notes=notes, stamp_date=stamp_date)

def plot_scatter(fig                                          , ax:Axes                              , gout: h5py.File                                 , 
                 toplot_x   : (str | Callable)                , toplot_y   : (str | Callable)        , nfilter_x       : (np.ndarray | Callable)       , 
                 nfilter_y  : (np.ndarray | Callable)         , nsigma     :float             = 1.0  , error_plot      : bool                    = True,
                 x_scale    : float                     = 1.0 , y_scale    :float             = 1.0  , kwargs_plot                               = None, 
                 kwargs_fill                            = None                                                                                         ):
    kwargs_fill = {} if kwargs_fill is None else kwargs_fill
    kwargs_plot = {} if kwargs_plot is None else kwargs_plot
    kwargs_scri
    
    get_x = freeze(nodedata, key=toplot_x) if isinstance(toplot_x, str) else toplot_x
    get_y = freeze(nodedata, key=toplot_y) if isinstance(toplot_y, str) else toplot_y 
    
    x       = get_x(gout, nfilter=nfilter_x, summarize=True, statfuncs=(np.mean        ))
    y, ystd = get_y(gout, nfilter=nfilter_y, summarize=True, statfuncs=(np.mean, np.std))
    
    x    *= x_scale
    y    *= y_scale
    ystd *= y_scale * nsigma
    
    if error_plot: 
        ax.errorbar(x, y, ystd, **(KWARGS_DEF_ERR | kwargs_plot))
        return

    ax.plot(x, y_avg, **(KWARGS_DEF_PLOT | kwargs_plot))
    ax.fill_between(x,y - ystd, y + ystd, **(KWARGS_DEF_FILL | kwargs_fill))

def plot_histogram(fig                                             , ax:Axes                              , gout: h5py.File                                 , 
                   get_histogram   : (Callable)                    , nfilter    : (np.ndarray | Callable) ,  nsigma     :float             = 1.0  , 
                   error_plot      : bool                    = True, scale_x         : float                   = 1.0 , 
                   scale_y         :float             = 1.0  , kwargs_plot                               = None, kwargs_fill                               = None,
                   kwargs_script = None, projection=False): 
    kwargs_fill   = {} if kwargs_fill   is None else kwargs_fill
    kwargs_plot   = {} if kwargs_plot   is None else kwargs_plot
    kwargs_script = {} if kwargs_script is None else kwargs_script

    _get_hist = multiproj(get_histogram, nfilter) if projection else freeze(get_histogram, nfilter=nfilter)
    _hist     = _get_hist(gout, summarize=True, statfuncs=(np.mean, np.std), **kwargs_script)

    y   , x = _hist[0][0], bin_avg(_hist[0][1])
    ystd, _ = _hist[1]
    
    x    *= scale_x
    y    *= scale_y
    ystd *= scale_y * nsigma

    if error_plot: 
        ax.errorbar(x, y, ystd, **(KWARGS_DEF_ERR | kwargs_plot))
        return

    ax.plot(x, y, **(KWARGS_DEF_PLOT | kwargs_plot))
    ax.fill_between(x,y - ystd, y + ystd, **(KWARGS_DEF_FILL | kwargs_fill))

