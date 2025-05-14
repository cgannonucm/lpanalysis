#!/usr/bin/env python
import spatial_3d
import spatial_ratio
import spatial_2d
import scaling_plot
import massfunction
import um_evolution


def main():
    printlabel = lambda i: print(f"Generating figure {i}")

    #Figures 1-6
    print("Generating figures")
    printlabel(1)
    spatial_3d.main()

    printlabel(2)
    spatial_ratio.main()

    printlabel(3)
    spatial_2d.main()

    printlabel(4)
    scaling_plot.main()

    printlabel(5)
    massfunction.main()

    printlabel(6)
    um_evolution.main()
    
    #Tables 1-4



if __name__ == "__main__":
    main()
