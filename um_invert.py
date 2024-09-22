#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any
from um_scaling import um_smhm_mstar
from scipy.interpolate import interp1d

def main():
    mhspace = np.logspace(10, 15, 1000)
    ms = um_smhm_mstar(mhspace, 0.5)
    interp = interp1d(ms, mhspace)
    
    mh = interp(10**(11.3))
    print(np.log10(mh))
    
    pass

if __name__ == "__main__":
    main()

