#!/usr/bin/env python
import numpy as np
import h5py
import os
import re
from multiprocessing import Pool
from typing import Callable, Iterable
import logging
import time

from subscript.defaults import ParamKeys
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import nfand, nfilter_halos, nfilter_subhalos, nfilter_subhalos_valid, nfilter_project2d, nfilter_range
from subscript.scripts.histograms import massfunction, spatial3d_dndv
from subscript.wrappers import gscript_proj, multiproj
from subscript.wrappers import freeze
from subscript.macros import macro_run, macro_write_out_hdf5, macro_gen_runner, macro_run_file 
from subscript.tabulatehdf5 import tabulate_trees

from plotting_util import savemacro

def summary_macro(mass_min, mass_max, mbincount, rap_min, rap_max):
    mbins = np.geomspace(mass_min, mass_max, mbincount)
    nfiltersub = freeze(nfilter_subhalos_valid, mass_min=mass_min, mass_max=mass_max, key_mass=ParamKeys.mass_basic)
    nfilterproj = freeze(nfilter_project2d, rmin=rap_min, rmax=rap_max)
    filtermbound = freeze(nfilter_range, min=mass_min, max=mass_max, key=ParamKeys.mass_bound)
    nfilterprojsub_u = nfand(nfiltersub, nfilterproj)
    nfilterprojsub_e = nfand(nfilterprojsub_u, filtermbound)    

    normvectors = np.identity(3)

    label_mfpj_u =  f"massfunction (unevolved) ({rap_min} < " + "r_{2d}" + f" / MPC <= {rap_max})" 
    label_mfpj_e =  f"massfunction (evolved) ({rap_min} < " + "r_{2d}" + f" / MPC <= {rap_max})"    
    label_nrap_u  =  f"n evolved {rap_min} < " + "r_{2d}" + f" <= {rap_max}, {mass_min:.2e} < m < {mass_max:.2e}"
    label_nrap_e  =  f"n evolved {rap_min} < " + "r_{2d}" + f" <= {rap_max}, {mass_min:.2e} < m_e < {mass_max:.2e}"

    macros = {
                "halo mass"                 : freeze(nodedata, key=ParamKeys.mass_basic    , nfilter=nfilter_halos),
                "z"                         : freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=nfilter_halos),
                "massfunction (evolved)"    : freeze(massfunction, key_mass=ParamKeys.mass_bound, bins=mbins, nfilter=nfiltersub),
                "massfunction (unevolved)"  : freeze(massfunction, key_mass=ParamKeys.mass_basic, bins=mbins, nfilter=nfiltersub),
                label_mfpj_u                : freeze(multiproj(massfunction, nfilter=nfilterprojsub_u), key_mass=ParamKeys.mass_basic, bins=mbins, normvector=normvectors),
                label_mfpj_e                : freeze(multiproj(massfunction, nfilter=nfilterprojsub_e), key_mass=ParamKeys.mass_basic, bins=mbins, normvector=normvectors),
                label_nrap_u                : freeze(multiproj(nodecount, nfilter=nfilterprojsub_u),summarize=True, normvector=normvectors),
                label_nrap_e                : freeze(multiproj(nodecount, nfilter=nfilterprojsub_e),summarize=True, normvector=normvectors)
    }

    return macros

def get_hdf5_dir(path):
    return [h5py.File(os.path.join(path, fpath)) for fpath in os.listdir(path) if re.findall(".hdf5$", fpath)]

def get_hdf5_dir_path(path):
    return [os.path.join(path, fpath) for fpath in os.listdir(path) if re.findall(".hdf5$", fpath)]

def macro_gen_runner_parallel(macros, statfuncs):
    def macro_runner_parallel(path_gout):
        f = h5py.File(path_gout)
        logging.info(f"Reading file: {f.filename}")
        t0 = time.perf_counter()
        out = (f.filename, macro_run_file(f, macros,  statfuncs=statfuncs))
        t1 = time.perf_counter()
        logging.info(f"Finished ({t1 - t0} s)")
        return out
    return macro_runner_parallel

class WrapperHDF5():
    def __init__(self, file):
        self.wrap = file
    
    def __get__(key):
        return self.wrap[key][:] 

if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)

    fname = "summary.hdf5"
    path_gout = "data/galacticus/summary"
    #gouts     = get_hdf5_dir(path_gout)
    gouts     = [get_hdf5_dir(path_gout)[0], ]
    gout_paths = get_hdf5_dir_path(path_gout)

    macros = summary_macro(1E8, 1E10, 6, 1E-2, 2E-2)    

    statfuncs = [np.mean, np.std]

    def mrp(path):
        return macro_gen_runner_parallel(macros, statfuncs)(path)

    with Pool(16) as pool: 
        macro_results = pool.map(mrp, gout_paths)

    runner = macro_gen_runner(lambda *a,**k: macro_results)

    macro_out = macro_run(macros, gouts, statfuncs=statfuncs, runner=runner)

    savemacro(macro_out, fname)