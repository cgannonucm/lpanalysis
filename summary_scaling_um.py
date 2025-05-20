#!/usr/bin/env python
import numpy as np
import h5py
import os
import re
from multiprocessing import Pool
from typing import Callable, Iterable
import logging

from subscript.defaults import ParamKeys
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import nfand, nfilter_halos, nfilter_subhalos, nfilter_subhalos_valid, nfilter_project2d
from subscript.scripts.histograms import massfunction, spatial3d_dndv
from subscript.wrappers import gscript_proj, multiproj
from subscript.wrappers import freeze
from subscript.macros import macro_run, macro_write_out_hdf5, macro_gen_runner, macro_run_file 
from subscript.tabulatehdf5 import tabulate_trees

from plotting_util import savemacro

from summary import summary_macro, get_hdf5_dir_path, macro_gen_runner_parallel, get_hdf5_dir

def main():
    logging.root.setLevel(logging.INFO)

    fname = "scaling_um.hdf5"
    path_gout = "data/galacticus/scaling_um"

    gout_paths = get_hdf5_dir_path(path_gout)
    gouts = get_hdf5_dir(path_gout)

    massrange_3d = np.logspace(8, 11, 7)
    rbins_rvf = np.geomspace(0.1, 1, 20)
    
    macros = summary_macro(1E8, 1E10, 6, 1E-2, 2E-2, massrange_3d, rbins_rvf)    

    statfuncs = [np.mean, np.std]

    def mrp(path):
        return macro_gen_runner_parallel(macros, statfuncs=statfuncs)(path)

    #with Pool(32) as pool: 
        #macro_results = pool.map(mrp, gout_paths)

    macro_results = [mrp(path) for path in gout_paths]

    runner = macro_gen_runner(lambda *a,**k: macro_results)

    macro_out = macro_run(macros, gouts, statfuncs=statfuncs, runner=runner)

    logging.info("Finished")

    savemacro(macro_out, fname)

if __name__ == "__main__":
    main()
