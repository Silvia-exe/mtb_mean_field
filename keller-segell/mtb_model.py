
import argparse
import numpy as np
from mtb_class import mtb
import mtb_plotter as mtb_plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bacterial simulation")
    parser.add_argument("--arg_file", default= 'parameters.yaml', help="Path to parameter YAML file", required=False)
    parser.add_argument('--O2_init', help = 'Path to .h5 file for O2 gradient initialization', required=False, default = None)

    args = parser.parse_args()

    run = mtb(args)

    #run.init_oxygen()
    #run.inoculum_center()

    #animate_colormaps(run)

    mtb_plt.plot_colormaps(run)
    run.run_save()
    mtb_plt.plot_colormaps(run)
  


    
