#!/usr/bin/env python
"""
This auto runs scripts to generate figures / tables presented in Gannon et al. 2025 (https://arxiv.org/abs/2501.17362)
See README.md for setup instructions
"""
import spatial_3d
import spatial_ratio
import spatial_2d
import scaling_plot
import massfunction
import um_evolution
import summary_scaling
import summary_scaling_um
import han_model_fit
import sigmasub
import scaling_external
import scaling_fit

def askyn(question):
    ans = False
    while(True):
        _in = input(f"{question} n or y?").lower()
        if _in == "n" or _in == "":
            break        
        if _in == "y":
            ans = True
            break
        print("Please answer y or n") 
    return ans

def main():
    printfiglabel = lambda i: print(f"----------\n Generating figure {i}")
    printtablabel = lambda i: print(f"----------\n Generating table {i}")
    
    if askyn("Analyze DMO scaling summary (this may take a while)?"):
        summary_scaling.main()

    if askyn("Analyze scaling with central galaxy (this may take a while)?"):
        summary_scaling_um.main()

    #Figures 1-6
    print("Now Generating figures")
    printfiglabel(1)
    spatial_3d.main()


    printfiglabel(2)
    spatial_ratio.main()


    printfiglabel(3)
    spatial_2d.main()


    printfiglabel(4)
    scaling_plot.main()


    printfiglabel(5)
    massfunction.main()


    printfiglabel(6)
    um_evolution.main()
    
    #Tables 1-4
    print("Now outputting data for tables")
    printtablabel(1)
    han_model_fit.main() 

    printtablabel("2 and 3")
    sigmasub.main() 
    scaling_external.main()
    
    printtablabel(4)
    scaling_fit.main()

if __name__ == "__main__":
    main()
