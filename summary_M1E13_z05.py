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
from gstk.scripts.selection import script_selector_halos, script_selector_subhalos_valid, script_select_nodedata
from gstk.scripts.spatial import script_rvir
from gstk.gfile import GFile
from gstk.scripts.misc import script_treecount
from gstk.common.util import ifisnone

from gstk.common.r2d import r2d_project
import pandas as pd
from plotting_util import savedf, PARAM_DEF_ALPHA

@macro_mass(**ScriptProperties.KEY_KWARG_DEF)
def macro_scaling_projected_annulus_rv(file:GFile, mpivot:float, alpha:float, rvf0:float,rvf1:float,mrange:tuple[float,float],**kwargs)->dict[str,float]:
    """
    Runs macro. Returns a dictionary with the following elements:
    See the class MacroScalqing Keys for a discription of keys in the dictionary 
    """
    outputs:dict[str,float] = {}       
    
    rv = np.mean(script_rvir(file, select_function=script_selector_halos))
    r0, r1 = rvf0 * rv, rvf1 * rv
 
    sigsub_at,sigsub_var  = script_n_annulus_alltrees(file,mpivot,alpha,r0,r1,script_selector_subhalos_valid,mrange,**kwargs)
 
    annulus_str = f" {rvf0} < r_2d/r_v < {rvf1}"
 
    outputs[MacroScalingKeys.PROJECTED_COUNT + annulus_str] = np.mean(sigsub_var["n"])
    outputs[MacroScalingKeys.PROJECTED_COUNT_SD + annulus_str] = np.std(sigsub_var["n"])
 
    return outputs

@macro_mass(**ScriptProperties.KEY_KWARG_DEF)
def macro_sigma_sub(file:GFile, mpivot:float, alpha:float, r0:float,r1:float,mrange:tuple[float,float],
                    scale_hmz=False, scale_hmz_coef=None, scale_hmz_zshift=None, scale_hmz_mscale=None,**kwargs)->dict[str,float]:
    scale_hmz_coef = (0.37, 1.05) if scale_hmz_coef is None else scale_hmz_coef
    scale_hmz_zshift = 0.5 if scale_hmz_zshift is None else scale_hmz_zshift
    scale_hmz_mscale = 1E13 if scale_hmz_mscale is None else scale_hmz_mscale

    outputs = {}    

    sigsub_at_i,sigsub_var_i  = script_n_annulus_alltrees(file,mpivot,alpha,r0,r1,script_selector_subhalos_valid,mrange,key_mass=GParam.MASS_BASIC)
    sigsub_at_b,sigsub_var_b  = script_n_annulus_alltrees(file,mpivot,alpha,r0,r1,script_selector_subhalos_valid,mrange,key_mass=GParam.MASS_BOUND)

    n_arr_i, n_arr_b = sigsub_var_i["n"], sigsub_var_b["n"]
    n_arr_fs = n_arr_b / n_arr_i
    n_arr_fs = n_arr_b / n_arr_i

    n_avg_i, n_std_i = np.mean(n_arr_i), np.std(n_arr_i)
    n_avg_b, n_std_b = np.mean(n_arr_b), np.std(n_arr_b)
    n_avg_fs, n_std_fs = np.mean(n_arr_fs), np.std(n_arr_fs) 
    
    n0 = sigsub_var_N0(mrange,alpha,mpivot)
    area = np.pi * (r1**2 - r0**2) * CPhys.MPC_TO_KPC**2

    scale = 1
    if scale_hmz:
        raise NotImplementedError()
        hm = np.mean(script_select_nodedata(file, script_selector_halos, [GParam.MASS_BASIC]))
        z = np.mean(script_select_nodedata(file, script_selector_halos, [GParam.Z_LASTISOLATED]))
        k1, k2 = scale_hmz_coef
        scale = 1 / (hm / scale_hmz_mscale)**k1 / (z + scale_hmz_zshift)**k2 
        

    outputs["\Sigma_{sub} [kpc^{-2}]"]              = n_avg_i / area / n0 * scale
    outputs["\Sigma_{sub} [std] [kpc^{-2}]"]        = n_std_i / area / n0 * scale
    outputs["f_s \Sigma_{sub} [kpc^{-2}]"]          = n_avg_b / area / n0 * scale
    outputs["f_s \Sigma_{sub} [std] [kpc^{-2}]"]    = n_std_b / area / n0 * scale
    outputs["f_s"]                                  = n_avg_fs
    outputs["f_s [std]"]                            = n_std_fs 
       
    return outputs

@script(**ScriptProperties.KEY_KWARG_DEF)
def script_tree_avg(data, script, selector_function, script_args = None, script_kwargs = None, **kwargs):
    script_args = [] if script_args is None else script_args
    script_kwargs = {} if script_kwargs is None else script_kwargs
    
    select = selector_function(data,**kwargs)


    node_trees_select = data[kwargs[ScriptProperties.KEY_KWARG_TREENUMBER]]

    #Number of axes to project over
    treecount = script_treecount(data)

    out = []

    for treenum in range(treecount): 
        tree_selected = select & (node_trees_select == treenum)
        selector = lambda d, **k: tree_selected
        out.append(script(data, *script_args, selector=selector, **script_kwargs)) 

    return out



@macro(**ScriptProperties.KEY_KWARG_DEF)
def macro_mass_function_fit_manytree(data, mrange, bincount, **kwargs):
    fits = script_tree_avg(data, macro_massfunction_fit, script_selector_subhalos_valid, 
                    script_kwargs=(dict(mrange=mrange, bincount=bincount) | kwargs), mrange=mrange)
    
    alpha = np.zeros(len(fits))
    coef = np.zeros(len(fits)) 
    key_alpha, key_coef = None, None

    for n, entry in enumerate(fits):
        for key, value in entry.items():
            if "alpha" in key:
                key_alpha = key
                alpha[n] = value
            elif "coef" in key:
                key_coef = key
                coef[n] = value
            else:
                raise Exception()

    key_alpha = key_alpha + " [Averaged]"
    key_coef = key_coef + " [Averaged]"

    
    return {
                key_alpha: np.mean(alpha), 
                key_alpha + "[std]": np.std(alpha),
                key_coef: np.mean(coef),
                key_coef + " [std]": np.std(coef)
    }



if __name__ == "__main__":
    data_dir = "data/galacticus/xiaolong_update/m1e13_z0_5"
    path_csv = "out/csv/summary_m1E13_z05.csv"

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
    add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
              bincount=mbins,key_mass=GParam.MASS_BASIC,selector=script_selector_all)
    
    add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
              bincount=mbins,key_mass=GParam.MASS_BOUND,selector=script_selector_all)
    
    add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
                bincount=mbins,key_mass=GParam.MASS_BASIC,selector=script_selector_annulus,
                r0=r0_mpc,r1=r1_mpc,macro_stamp=f" {r0_mpc:.2E}[MPC] < r_2d < {r1_mpc:.2E}[MPC]")
            
    add_macro(macro_massfunction_fit,mrange=mrange_mass_function,
                bincount=mbins,key_mass=GParam.MASS_BOUND,selector=script_selector_annulus,
                r0=r0_mpc,r1=r1_mpc,macro_stamp=f" {r0_mpc:.2E}[MPC] < r_2d < {r1_mpc:.2E}[MPC]")
    
    #Host halo scale radius
    add_macro(macro_halo_scaleradius)
    
    #Virial radius
    macros = add_macro(macro_rvir)

    run_parallel = True

    def run_macros_file_(fpath):
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

    savedf(df, "summary_m1E13_z05.csv")

