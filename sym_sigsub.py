#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.axes import Axes
from typing import Any
import h5py
import os.path
import pandas as pd

from plotting_util import *
from symutil import symphony_to_galacticus_dict
from summary_M1E13_z05 import macro_sigma_sub


def main(): 
    #path_file =  "data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    path_cat = "data/caterpillar/subhaloDistributionsCaterpillar.hdf5"
    path_symdir = "data/symphony/SymphonyGroup/"

    #path_file = "data/galacticus/xiaolong_update/m1e13_z0_2/lsubmodv3.1-scaling-nd-date-05.12.2024-time-18.29.10-basic-date-05.12.2024-time-18.29.10-z-2.00000E-01-hm-8.85867E+12.xml.hdf5"
    #path_file = "/home/charles/research/lpanalysis/data/galacticus/xiaolong_update/m1e13_z0_5/lsubmodv3.1-M1E13-z0.5-nd-date-06.12.2024-time-14.12.04-basic-date-06.12.2024-time-14.12.04-z-5.00000E-01-hm-1.00000E+13.xml.hdf5"
    
    #rrange_rvf = PARAM_DEF_RRANGE_RVF
    rrange_rvf = (0.1, 1)
    rbins = 15
    #mrange = PARAM_DEF_MRANGE
    mrange = (1E9, 1E10)
    mrange_sym = (1E9, 1E13)
    mrange_rescale = (1E9, 1E10)
    
    sym_nodedata = symphony_to_galacticus_dict(path_symdir, iSnap=203)

    #mh = np.mean(script_select_nodedata(sym_nodedata, script_selector_halos, [GParam.MASS_BASIC]))
    out = macro_sigma_sub(sym_nodedata, 1E8, -1.97, 0, 5E-2, mrange_sym)
    #out = macro_sigma_sub(filend, 1E8, -1.97, 1E-2, 2E-2, (1E8, 1E9))

    df = pd.DataFrame(out, index=[0])
    
    savedf(df, "sym_sigsub.csv")


if __name__ == "__main__":
    main()
