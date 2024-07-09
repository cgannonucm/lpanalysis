import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os.path as path
import matplotlib.patheffects as pe


KWARGS_DEF_PLOT = dict(linewidth=5)

KWARGS_DEF_ERR = KWARGS_DEF_PLOT |  dict(capsize=5, markersize=8,
                                        path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

PATH_OUT = "out"
PATH_PLOTS = "plots"
PATH_CSV = "csv"

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

def savedf(df, name):
    df.to_csv(path.join(PATH_OUT,PATH_CSV,name), index=False)
