#!/usr/bin/env python3
import numpy as np
from colossus.cosmology import cosmology

def main():

    cosmo = cosmology.setCosmology("planck18")

    z = 0.5

    rad_to_as = 2.06E5

    print(cosmo.angularDiameterDistance(z) / cosmo.h)

    # Convert kpc to arcseconds at z = 0.5
    to_as =  1E-3 * 1 / cosmo.angularDiameterDistance(z) * cosmo.h * rad_to_as

    slacs_median_einstien_radius_arcsec = 1.17

    print(slacs_median_einstien_radius_arcsec / to_as)



if __name__ == "__main__":
    main()
