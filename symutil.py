#!/usr/bin/env python
import numpy as np
import symlib
import os.path
import os
import pickle
from zlib import adler32
import tempfile


from subscript.defaults import ParamKeys
from plotting_util import *


def symphony_to_galacticus_dict(haloFolder:str, iSnap=-1, cache=True,  path_cache="cache"):
    # Reading symphony data from file takes very long, cache results to greatly increase speed
    name = f"{haloFolder}-{iSnap}"
    fhash = adler32(name.encode("utf-8"))
    path_fhash = path.join(path_cache, str(fhash))

    if cache and path.exists(path_fhash):
        with open(path_fhash, "rb") as pf:
            return pickle.load(pf) 

    if cache and not path.exists(path_cache):
        os.makedirs(path_cache)
        
    halonames  = os.listdir(haloFolder)

    def get_halo_info(haloFolder, halo, iSnap=-1):
        sim_dir     = haloFolder+halo

        snapz = 1 / symlib.scale_factors(sim_dir)[iSnap] - 1
    
        out = {}
       
        h0, hist0     = symlib.read_subhalos(haloFolder+halo)
        h, hist     = symlib.read_symfind(haloFolder + halo)
        
        params      = symlib.simulation_parameters(sim_dir)
        hubble      = params['H0']/100.0

        posHost     = h0['x'    ][0 ,iSnap]/1.0e3
        Mhost       = h0['mvir' ][0 ,iSnap]
        RvirHost    = h0['rvir' ][0 ,iSnap]/1.0e3
    
        posSub      = h['x'    ][1:,iSnap][h['ok'][1:,iSnap]]/1.0e3
        MboundSub   = h['m'    ][1:,iSnap][h['ok'][1:,iSnap]]
        # Maximum mass of subahlo in the history (treated as infall mass in Galacticus).
        MpeakSub    = hist['mpeak'][1:][h['ok'][1:,iSnap]]
        RVirSub     = -1 * np.ones(MpeakSub.shape)#h['rvir' ][1:,iSnap][h['ok'][1:,iSnap]]

        nsub = MpeakSub.shape[0]
        
        # Coordinates
        XRel = (posSub[:,0]-posHost[None,0])
        YRel = (posSub[:,1]-posHost[None,1])
        ZRel = (posSub[:,2]-posHost[None,2])

        XRel = np.insert(XRel, 0, 0)
        YRel = np.insert(YRel, 0, 0)
        ZRel = np.insert(ZRel, 0, 0)

        out[ParamKeys.x] = XRel
        out[ParamKeys.y] = YRel
        out[ParamKeys.z] = ZRel

        out["coordinates"] = np.asarray((XRel, YRel, ZRel)).T

        # Bound (peak) mass
        MpeakSub = np.insert(MpeakSub, 0, Mhost)
        out[ParamKeys.mass_basic] = MpeakSub

        # Bound mass
        MboundSub = np.insert(MboundSub, 0, Mhost)
        out[ParamKeys.mass_bound] = MboundSub

        # virial radius
        RVirSub = np.insert(RVirSub, 0, RvirHost)
        out[ParamKeys.rvir] = RVirSub

        # Is isolated
        isisolated = np.zeros(nsub)
        isisolated = np.insert(isisolated, 0, 1)
        out[ParamKeys.is_isolated] = isisolated

        zlastiso = -1 * np.ones(nsub + 1)
        zlastiso[0] = snapz
        out[ParamKeys.z_lastisolated] = zlastiso

        return out
    
    out = {}

    for n, hname in enumerate(halonames):
        halo_dict = get_halo_info(haloFolder, hname, iSnap=iSnap)
        
        for key, val in halo_dict.items():
            nsub = val.shape[0]

            if n == 0:
                out[key] = val
                continue
            out[key] = np.concatenate((out[key], val), axis=0)
        
        KEY_ORDER = "custom_node_tree_outputorder"
        KEY_TREE = "custom_node_tree"

        if n==0:
            out[KEY_ORDER]  = np.ones(nsub, dtype=int) * n
            out[KEY_TREE]   = np.ones(nsub, dtype=int) * n + 1
            continue

        out[KEY_ORDER] = np.concatenate((out[KEY_ORDER], np.ones(nsub,dtype=int) * n), axis=0)
        out[KEY_TREE]  = np.concatenate((out[KEY_TREE], np.ones(nsub,dtype=int) * n + 1), axis=0)


    if cache:
        with open(path_fhash,"wb") as pf:
            pickle.dump(out,pf)

    return out


def symphony_to_galacticus_hdf5(haloFolder:str, iSnap=-1, 
                                    cache=True, path_cache="cache", 
                                    hdf5_file=None):
    """Create a hdf5 file in Galacticus format"""                                
    sym_out = symphony_to_galacticus_dict(haloFolder=haloFolder, iSnap=iSnap, cache=cache, path_cache=path_cache)

    if hdf5_file is None:
        tf = tempfile.TemporaryFile()
        hdf5_file =  h5py.File(tf, "r+")

    gout   = hdf5_file.create_group(f"Outputs/Output1")
    gndata = hdf5_file.create_group(f"Outputs/Output1/nodeData")

    for key, val in sym_out.items():
        gndata.create_dataset(key, data=val)

    KEY_TREE_START     = "tree_start"
    KEY_TREE           = "custom_node_tree"       

    KEY_GAL_TREE_COUNT = "mergerTreeCount"
    KEY_GAL_TREE_INDEX = "mergerTreeIndex"
    KEY_GAL_TREE_START = "mergerTreeStartIndex"

    tree_index = sym_out[KEY_TREE] 
    tree_index_unique = np.unique(tree_index)

    treecount = np.asarray([np.sum(tree_index == key_tree) for key_tree in tree_index_unique])
    tree_start = np.insert(np.cumsum(treecount)[:-1], 0, 0)

    gout.create_dataset(KEY_GAL_TREE_COUNT, data=treecount)
    gout.create_dataset(KEY_GAL_TREE_INDEX, data=tree_index_unique)
    gout.create_dataset(KEY_GAL_TREE_START, data=tree_start)

    return hdf5_file
