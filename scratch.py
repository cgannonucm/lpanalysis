#!/usr/bin/env python
from __future__ import annotations
from typing import Any, Callable, Iterable, ParamSpec, Concatenate, TypeVar, Generic
from functools import wraps
import numpy as np
import h5py

from gstk.tabulatehdf import lazyNodeProperties, tabulate_nodes
from gstk.common.constants import GParam

def nodedata(gout, tree_index=-1):
    if isinstance(gout, lazyNodeProperties):
        _gout = gout
    elif isinstance(gout, h5py.File):
        # TODO: Refactor key_index to tree_inxex
        _gout = tabulate_nodes(gout, key_index=tree_index)
    elif isinstance(gout, dict):
        _gout = lazyNodeProperties(gout)
    else:
        raise RuntimeError("Unrecognized data type for gout")
    return _gout

def nfiltercallwrapper(func):
    def wrap(self, gout:(h5py.File | lazyNodeProperties | dict), *args, tree_index=-1, **kwargs):
        return func(self, nodedata(gout, tree_index=tree_index), *args, tree_index=-1, **kwargs)
    return wrap

class NodeFilterWraper(): 
    def __init__(self, func = None):
        self.wrap = func
    
    @nfiltercallwrapper
    def __call__(gout, *args, **kwargs)->np.ndarray[bool]:
        return self.wrap(*args, **kwargs)

    def __and__(self, other:NodeFilterWraper | Callable | np.ndarray):
        if isinstance(other, Callable):
            return NodeFilterWraper(lambda *a, **k: self(*a,**k) & other(*a, **k))
        if isinstance(other, np.ndarray):
            return NodeFilterWraper(lambda *a, **k: self(*a,**k) & other)
        raise RuntimeError("Invalid __and__ operation") 
    
    def __or__(self, other:(NodeFilterWraper | np.ndarray)):
        if isinstance(other, Callable):
            return NodeFilterWraper(lambda *a, **k: self(*a,**k) | other(*a, **k))
        if isinstance(other, np.ndarray):
            return NodeFilterWraper(lambda *a, **k: self(*a,**k) | other)
        raise RuntimeError("Invalid __or__ operation")
    
    def logical_not(self):
        return NodeFilterWraper(lambda *a, **k: np.logical_not(self(*a,**k)))

    __invert__ = logical_not

    def freeze(self, *args, **kwargs):
        return lambda gout: self(gout, *args, **kwargs)

class _nfilter_all(NodeFilterWraper):
    @nfiltercallwrapper
    def __call__(self, gout, **kwargs):
        return np.ones(gout[next(iter(gout))].shape, dtype=bool)

nfilter_all = _nfilter_all()

def gscript(func):
    def wrap(gout:(h5py.File | lazyNodeProperties | dict), *args, 
                nodefilter=None, treestats=False, treestatfuncs:Iterable[Callable] = None,
                tree_index=-1, **kwargs): 
        _gout = nodedata(gout, tree_index)

        trees = np.unique(_gout["custom_node_tree"])

        if nodefilter is None:
            nodefilter = np.ones(_gout[next(iter(_gout))].shape, dtype=bool)

        if isinstance(nodefilter, np.ndarray):
            _nodefilter = nodefilter
        elif isinstance(nodefilter, Callable):
            _nodefilter = nodefilter(_gout, *args, **kwargs)
        else:
            raise RuntimeError("Unrecognized data type for nodefilter")
          
        summary = None

        for treen, tree in enumerate(trees):
            filtertree = _gout["custom_node_tree"] == tree
            _gout_ftree    = _gout.filter(filtertree)
            _gout_ftree_nf = _gout.filter(filtertree & _nodefilter)
    
            out = func(_gout_ftree_nf, *args, 
                        nodefilter=_nodefilter, **kwargs, 
                        gout_free=_gout_ftree)

            single_out = isinstance(out, np.ndarray)

            _out = out
            if single_out:
                _out = [out, ]
            
            if summary is None:
                summary = []
                for o in _out:
                    summary.append([[] for tree in trees])

            for n,o in enumerate(_out):
                summary[n][treen] = o 
        
        if not treestats:
            if single_out:
                return summary[0]
            return summary 

        _treestatfuncs = (np.mean, ) if treestatfuncs is None else treestatfuncs        

        stat_summary = []
        for _func in _treestatfuncs:
            stat_summary.append([])
            for arr in summary:
                stat_summary[-1].append(_func(arr, axis=0))

        if single_out:
            return stat_summary[0]
        return stat_summary 
    return wrap

@gscript
def nodevalue(gout, label:(str | Iterable[str]), **kwargs):
    return gout[label]

def main():
    path_dmo = "data/galacticus/um_update/dmo.hdf5"
    gout = h5py.File(path_dmo)
    #print(nodevalue(gout, label=GParam.MASS_BASIC))    
    print(nfilter_all(gout))


if __name__ == "__main__": 
    main()
    
