#!/usr/bin/env python

import os
import getopt
import sys
from multiprocessing import Pool
from typing import Iterable
import numpy as np

from gstk.scripts.common import GParam, CPhys
from gstk.io import csv_cacheable, io_list_directory
from gstk.scripts.selection import script_selector_annulus, script_selector_all
from gstk.macros.common import macro_combine, macro_run, macro_run_filepath
from gstk.macros.scaling import macro_scaling_filemassz, macro_scaling_ntotal, macro_scaling_projected_annulus
from gstk.macros.halo import macro_halo_scaleradius, macro_rvir
from gstk.macros.massfunction import macro_massfunction_fit
from gstk.scripts.sigmasub import script_sigmasub_alltrees, script_n_annulus_alltrees, sigsub_var_M0, sigsub_var_sigma_sub, sigsub_var_N0
from gstk.macros.common import MacroScalingKeys, macro_mass, macro
from gstk.scripts.common import ScriptProperties, script
from gstk.common.constants import GParam
from gstk.scripts.selection import script_selector_halos, script_selector_subhalos_valid
from gstk.scripts.spatial import script_rvir
from gstk.gfile import GFile
from gstk.scripts.misc import script_treecount
from gstk.common.util import ifisnone

from gstk.common.r2d import r2d_project
import pandas as pd
from plotting_util import savedf, PARAM_DEF_ALPHA

from summary_M1E13_z05 import macro_scaling_projected_annulus_rv,macro_sigma_sub, script_tree_avg, macro_mass_function_fit_manytree


if __name__ == "__main__":

    data_dir = "data/galacticus/xiaolong_update/cg-2"
    csv_out = "summary_cg.csv"

    nodedsetstoread = [GParam.MASS_BOUND,GParam.MASS_BASIC,GParam.IS_ISOLATED,GParam.RVIR,GParam.X,GParam.Y,GParam.Z,
                       GParam.DENSITY_PROFILE,GParam.DENSITY_PROFILE_RADIUS,GParam.SCALE_RADIUS,GParam.Z_LASTISOLATED]
     
    dirlist = os.listdir(data_dir)
    inclstr = ".hdf5"
    
    mrange_primary = (1E8,1E9)
    
    mrange_1 = (10**(8.0),10**(8.5))
    mrange_2 = (10**(8.5),10**(9))
    mrange_3 = (10**(9.0),10**(9.5))
    mrange_4 = (10**(9.5),10**(10))
    
    mrange_mass_function = (1E8,1E10)
    
    r0_mpc, r1_mpc = 10 * CPhys.KPC_TO_MPC, 20 * CPhys.KPC_TO_MPC
    alpha = PARAM_DEF_ALPHA
    mpivot = 10**8
    
    mmin = 1E8
    mmax = 1E9
    mbins = 20    
    
    
    add_macro = macro_combine()
    #Get redshift / halo mass
    add_macro(macro_scaling_filemassz)
    
    #Get sigmasub / nprojected
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_primary,key_mass=GParam.MASS_BOUND) 
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_primary,key_mass=GParam.MASS_BASIC)
    
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_1,key_mass=GParam.MASS_BOUND) 
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_1,key_mass=GParam.MASS_BASIC)
    
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_2,key_mass=GParam.MASS_BOUND) 
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_2,key_mass=GParam.MASS_BASIC)
    
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_3,key_mass=GParam.MASS_BOUND) 
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_3,key_mass=GParam.MASS_BASIC)
    
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_4,key_mass=GParam.MASS_BOUND) 
    add_macro(macro_scaling_projected_annulus,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_4,key_mass=GParam.MASS_BASIC)
    

    #add_macro(macro_mass_function_fit_manytree,mrange=mrange_primary,bincount=10,key_mass=GParam.MASS_BASIC)
    #add_macro(macro_mass_function_fit_manytree,mrange=mrange_primary,bincount=10,key_mass=GParam.MASS_BOUND)


    add_macro(macro_sigma_sub,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_primary)    
    add_macro(macro_sigma_sub,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_1)
    add_macro(macro_sigma_sub,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_2) 
    add_macro(macro_sigma_sub,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_3) 
    add_macro(macro_sigma_sub,mpivot=mpivot,alpha=alpha,r0=r0_mpc,r1=r1_mpc,mrange=mrange_4)




#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.05,mrange=mrange_primary,key_mass=GParam.MASS_BOUND)
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.1,mrange=mrange_primary,key_mass=GParam.MASS_BOUND)
#
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.05,mrange=mrange_1,key_mass=GParam.MASS_BOUND)
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.1,mrange=mrange_1,key_mass=GParam.MASS_BOUND) 
#
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.05,mrange=mrange_2,key_mass=GParam.MASS_BOUND)
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.1,mrange=mrange_2,key_mass=GParam.MASS_BOUND) 
#
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.05,mrange=mrange_3,key_mass=GParam.MASS_BOUND)
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.1,mrange=mrange_3,key_mass=GParam.MASS_BOUND)
#       
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.05,mrange=mrange_4,key_mass=GParam.MASS_BOUND)
#    add_macro(macro_scaling_projected_annulus_rv,mpivot=mpivot,alpha=alpha,rvf0=0.02,rvf1=0.1,mrange=mrange_4,key_mass=GParam.MASS_BOUND) 
    
    #Get total number of subhalos
    add_macro(macro_scaling_ntotal,mrange=mrange_primary,key_mass=GParam.MASS_BOUND)
    add_macro(macro_scaling_ntotal,mrange=mrange_primary,key_mass=GParam.MASS_BASIC)
    
    # Mass function
    #add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
    #          bincount=mbins,key_mass=GParam.MASS_BASIC,selector=script_selector_all)
    
    #add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
    #          bincount=mbins,key_mass=GParam.MASS_BOUND,selector=script_selector_all)
    
    #add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
    #            bincount=mbins,key_mass=GParam.MASS_BASIC,selector=script_selector_annulus,
    #            r0=r0_mpc,r1=r1_mpc,macro_stamp=f" {r0_mpc:.2E}[MPC] < r_2d < {r1_mpc:.2E}[MPC]")
            
    #add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
    #            bincount=mbins,key_mass=GParam.MASS_BOUND,selector=script_selector_annulus,
    #            r0=r0_mpc,r1=r1_mpc,macro_stamp=f" {r0_mpc:.2E}[MPC] < r_2d < {r1_mpc:.2E}[MPC]")
    
    #Host halo scale radius
    add_macro(macro_halo_scaleradius)
    
    #Virial radius
    macros = add_macro(macro_rvir)
    
    run_parallel = True

    def run_macros_file_(fpath):
        #print(fpath)
        out = {
                    "filepath":fpath
        }

        out |= macro_run_filepath(fpath,macros,nodedata_allowed=nodedsetstoread)

        return out
    
    def run_macros_loop_(file_paths:Iterable[str],*args,**kwargs):
    
        if run_parallel:
            with Pool(16) as p:
                results = p.map(run_macros_file_,file_paths) 
            return results

        return [run_macros_file_(fpath) for fpath in file_paths]
     
    file_paths = io_list_directory(data_dir,inclstr=inclstr,include_path=True)

    table = macro_run(file_paths,macros,run_loop=run_macros_loop_)
    
    df = pd.DataFrame(table)
   
    savedf(df, csv_out)

