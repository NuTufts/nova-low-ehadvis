import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
from pathlib import Path

"""This file has one function which makes a  1d histogram of the interaction type for the imput generators"""

GENIEG1810j_path = "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only"
GENIEG21_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GENIEAR23_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GiBUU_path = "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/FHCDeriveFlux50MeVwidth_ppfx/numuCC_only/"

def histogram_int(value_protons, value_neutrons, value_pions, label, xmin=0, xmax=56, dif=1):
    # defining python list/library to store things inside the for loop
    array_list = []
    weight_list = []

    if label == "GENIEG21":
        path = GENIEG21_path
        var = "NEUT_int_type"
    elif label == "GENIEG1810j":
        path = GENIEG1810j_path
        var = "NEUT_int_type"
    elif label == "GENIEAR23":
        path = GENIEAR23_path
        var = "NEUT_int_type"
    elif label == "GiBUU":
        path = GiBUU_path
        var = "Mode"

    folder = Path(path)

    # Find all CSV files in that folder dynamically
    for file_path in folder.glob("*.root"):        
        n_files = len(list(folder.glob("*.root")))
        print(f"Processing: {file_path.name} out of {n_files} files")
        f = uproot.open(file_path)
        tree = f["generator_data"]
        
        branches = [var,'EventWeight','GenScaleFactor']
        arr = tree.arrays(branches,
                     f'(Enu<4.5) & (flagCC==True) & (n_FS_Muons==1)',
                     library="np")
        array = arr[var]
        weight = arr['EventWeight'] * arr['GenScaleFactor']

        weight_list.append(weight)
        array_list.append(array)

    # def full array of variable
    final_array = np.concatenate(array_list, axis=0)
    weight_array = np.concatenate(weight_list, axis=0)/(n_files*dif)
    bins = np.arange(xmin,xmax,dif)

    mask = (final_array >= xmin) & (final_array < xmax)
    filtered_array = final_array[mask]
    filtered_weights = weight_array[mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    _, bins, patches = plt.hist(filtered_array,
                                bins=bins, weights=filtered_weights, color=["#0D95DE"], label=[f'{var}'])

    centers = (bins[:-1] + bins[1:]) / 2

    labels = [f"{x:.1f}" for x in bins[:-1]]   # use 1-decimal labels, not int()

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation = 60, fontsize=8)

    plt.xlim([bins[0],bins[-1]])

    plt.title(f'histogram of {var} inclusive {label}')

    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper left')

    plt.title(f'histogram of {var} {value_protons}p{value_neutrons}n0pi_{label}')
    plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/Interaction_histograms/{var}_{value_protons}p{value_neutrons}n{value_pions}pi_{label}.png")
    plt.show()
    return final_array
