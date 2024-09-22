#!/usr/bin/env python
import h5py
from typing import Callable, Iterable
import numpy as np
import time
from collections import UserDict

class lazyNodeProperties(UserDict):
  nodeFilter = None
  nodecount = None

  class lazyStatus():
    read = 0
    unread = 1
    func = 2 

  def __getitem__(self, key):
      val = self.data[key]
      if val[0] == 0:
        return val[1][self.nodeFilter]
      if val[0] == 1:
        _val = val[1][:]
        self.data[key] = (0, _val)
        return _val[self.nodeFilter]
      if val[0] == 2:
        _val = val[1]()
        self.data[key] = (0, _val)
        return _val[self.nodeFilter]
        
def get_galacticus_outputs(galout:h5py.File)->np.ndarray[int]:
  output_groups:h5py.Group = galout["Outputs"] 

  outputs = np.zeros(len(output_groups), dtype=int)
  for n,key in enumerate(output_groups.keys()):
    outputs[n] = int(key[6:])
  return np.sort(outputs)

def get_custom_dsets(goutn:h5py.Group):
  """Generates standar custom datasets to make data analysis easier"""  

  # Total number of nodes
  nodecount = np.sum(goutn["mergerTreeCount"][:]) 
  # Node counts
  counts    = goutn["mergerTreeCount"]
  # Tree indexes
  treenums  = goutn["mergerTreeIndex"]
  
  return {
    "custom_node_tree"     : lambda _: np.concatenate([np.full(count,index) for (count,index) in zip(counts,treenums)]),
    "node_tree_outputorder": lambda _: np.concatenate([np.full(count,i) for (i,count) in enumerate(counts)]),
    "custom_id"            : lambda _: np.arange(nodecount)
  }


def get_node_properties(gout:h5py.File, key_index:int=-1, custom_dsets:Callable = None, **kwargs)->lazyNodeProperties:
  """Reads node propreties from a galacticus HDF5 file"""
  outs = gout["Outputs"] 

  _key_index = key_index
  if key_index == -1:
    _key_index = np.max(get_galacticus_outputs(gout))
  
  outn:h5py.Group = outs[f"Output{_key_index}"]
  nd:h5py.Group   = outn["nodeData"]

  # Total number of nodes in output can be obtained by summing this dataset
  # Which contains the number of nodes per tree
  nodecount = np.sum(outn["mergerTreeCount"][:]) 

  nodedata = {}
  for key, val in nd.items():
    # Add to return dictionary if we have a dateset that matches the total number of nodes
    if not isinstance(val, h5py.Dataset):
      continue
    if val.shape[0] != nodecount:
      continue
    # First entry in tuple marks this as a h5py dataset 
    nodedata[key] = (1, val)

  get_cdsets = get_custom_dsets if custom_dsets is None else custom_dsets
  cdsets = {key:(2, val) for key,val in get_cdsets(outn).items()}
  
  props = lazyNodeProperties(cdsets | nodedata)
  props.nodecount = nodecount 
  return props

      


  
  