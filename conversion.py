import numpy as np

def cfactor_dndlm(mrange):
    m0, m1 = mrange[0],  mrange[1]
    return (m0 + m1) / 2 / (m0 - m1)

def cfactor_dndm(mrange):
    m0, m1 = mrange[0],  mrange[1]
    return 1 / (m0 - m1)

def cfactor_shmf(mrange_old, mrange_new, alpha):
    g = alpha + 1
    m0, m1 = mrange_old
    m0p, m1p = mrange_new
    return (m1p**g - m0p**g) / (m1**g - m0**g)