import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from matplotlib.ticker import MaxNLocator

"""In this file, there are 3 functions. Two plot efficancy, and the third identifies the specific value of efficiancy in a 2d histogram of your choice. The first makes 1d histograms, and the second makes 2d histograms"""

path = "/cluster/tufts/lblnu/nova/app/users/cullenms/generators/"

file = uproot.open(f"{path}efficiencies_computed.root")

general = ['mu_angle_mu_energy;1', 'mu_angle_q0;1', 'mu_energy_q0;1','mu_angle_mu_energy_q0;1', 'mu_angle;1', 'mu_energy;1', 'nu_energy;1', 'q0;1']

def efficancy_histogram(xmin = 0, xmax = 0.1, dif = 0.02):
    
    for i in range(len(full_selection)-4):
        selection_num = full_selection[i+4]
        hist_num = file[selection_num]
        selection_denom = no_reco[i+4]
        hist_denom = file[selection_denom]
        
        label_list = selection_num.split("/")
        label = label_list[1]
        
        values = hist_num.values()/hist_denom.values()
        edges = hist_num.axis().edges()

        fig, ax = plt.subplots(figsize=(9, 5))
      
        ax.stairs(values, edges, fill=True, color="#008080", label=[f"{label}"])
        plt.legend(loc='upper left')
        plt.title(f'{label} efficancy')
        plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/efficiancy_histograms/test/hist_{label}_efficancypng")
        plt.close()
        print(f"imput for {i}: {values}")
        print(f"interation {i} of making histogram from {len(full_selection)-4}")
    
    return 0

def efficancy_heatmaps():

    for i in range(3):
        selection = general[i]
        hist_num = file[selection]

        values = hist_num.values()
        x_edges = hist_num.axes[0].edges()
        y_edges = hist_num.axes[1].edges()
        print(f"first{i}")

        label_list = selection_num.split(";")
        label = label_list[0]

        fig, ax = plt.subplots(figsize=(9, 5))
        mesh = ax.pcolormesh(x_edges, y_edges, values.T, cmap="Blues")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.setp(ax.get_xticklabels(), rotation=45)

        fig.colorbar(mesh, ax=ax, label="counts")
        plt.title(f'{label}')
        plt.tight_layout()
        plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/efficiancy_histograms/test/heatmap_{label}.png")
        plt.close()
    return 0

def hist_lookup_2d(x_value, y_value, imput):
    if imput == "angle_energy":
        selection = 'mu_angle_mu_energy;1'
    elif imput == "angle_q0":
        selection = 'mu_angle_q0;1'
    elif imput == "energy_q0":
        selection = 'mu_energy_q0;1'
    else:
        print("Imput is not found in catagories. Please imput angle_energy, angle_q0, or energy_q0.")

    hist = file[selection]
    values = hist.values()
    x_edges = hist.axes[0].edges()
    y_edges = hist.axes[1].edges()
    
    x_idx = np.digitize(x_value, x_edges) - 1
    y_idx = np.digitize(y_value, y_edges) - 1
    
    if x_idx < 0 or x_idx >= values.shape[0] or y_idx < 0 or y_idx >= values.shape[1]:
        print("imput values are not in range.")
        print(f"the range of x is: ({x_edges[0], x_edges[-1]})")
        print(f"the range of y is: ({y_edges[0], y_edges[-1]})")
        return None
    
    return values[x_idx, y_idx]

