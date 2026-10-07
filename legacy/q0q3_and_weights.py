import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
from pathlib import Path

"""This file has two functions. One makes a 2d histogram of q0 vs q3. The second makes a table of the total weights for each generator, and has not been updated recently, so you should look it over before using it.
This is also the file that I'm trying to add in the efficiancies first, so there might be some weird edits in the plot_q0q3 function"""

GENIEG1810j_path = "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only"
GENIEG21_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GENIEAR23_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GiBUU_path = "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/FHCDeriveFlux50MeVwidth_ppfx/numuCC_only/"



def hist_lookup_2d(x_value, y_value): 
    """x value is the Enu and the y value is q0"""

    # defining the selection and file
    path = "/cluster/tufts/lblnu/nova/app/users/cullenms/generators/"
    file = uproot.open(f"{path}efficiencies_computed.root")
    selection = 'mu_angle_mu_energy_q0;1'
    
    hist = file[selection]
    values = hist.values()

    # finding the "edges" of the histogram
    x_edges = hist.axes[0].edges()
    y_edges = hist.axes[1].edges()
    
    x_value = np.asarray(x_value)
    y_value = np.asarray(y_value)

    # making the histogram into a table you can draw values from 
    x_idx = np.digitize(x_value, x_edges) - 1
    y_idx = np.digitize(y_value, y_edges) - 1

    # making sure imputs are in range

    in_range = (
        (x_idx >= 0) & (x_idx < values.shape[0]) &
        (y_idx >= 0) & (y_idx < values.shape[1])
    )
    
    if not np.all(in_range):
        n_bad = np.sum(~in_range)
        print(f"{n_bad} input value(s) are not in range.")
        print(f"the range of x is: ({x_edges[0], x_edges[-1]})")
        print(f"the range of y is: ({y_edges[0], y_edges[-1]})")

    # clip so out-of-range indices don't cause an IndexError, then mask those entries with NaN afterward
    x_idx_clipped = np.clip(x_idx, 0, values.shape[0] - 1)
    y_idx_clipped = np.clip(y_idx, 0, values.shape[1] - 1)

    result = values[x_idx_clipped, y_idx_clipped].astype(float)
    result_withNan = np.where(in_range, result, np.nan)
    
    return result_withNan

def plot_q0q3(value_proton, value_neutron, value_pion, label, xmin=0, xmax =3, ymin = 0, ymax =2.5):
    muon_E = []
    neutrino_E = []
    weight_list = []
    neutrino_p = []
    muon_p = []

    if label == "GENIEG21":
        path = GENIEG21_path
    elif label == "GENIEG1810j":
        path = GENIEG1810j_path
    elif label == "GENIEAR23":
        path = GENIEAR23_path
    elif label == "GiBUU":
        path = GiBUU_path
    
    folder = Path(path)

    # Find all root files in the folder and iterate over them
    for file_path in folder.glob("*.root"):        
        n_files = len(list(folder.glob("*.root")))
        print(f"Processing: {file_path.name} out of {n_files} files")
        f = uproot.open(file_path)
        tree = f["generator_data"]

        # listing the values in ttree to be extracted
        branches = ['Enu','FS_Muons','EventWeight','GenScaleFactor']
        arrs = tree.arrays(branches, f"(Enu<4.5) & (flagCC==True) &  (n_FS_Muons==1) & (n_FS_Protons=={value_proton}) & (n_FS_Neutrons=={value_neutron}) & (n_FS_PiPs==0)& (n_FS_PiMs==0)& (n_FS_Pi0s==0)", library="ak")

        """finding neutrino and muon energy (what is needed for q0)"""
        # finding neutrino energy
        np_Enu = ak.to_numpy(arrs['Enu'])
        neutrino_E.append(np_Enu)

        # finding muon energy 
        E_muon_per_muon = arrs['FS_Muons'][..., 0]
        E_muon_total = ak.sum(E_muon_per_muon, axis=1)
        muon_E.append(ak.to_numpy(E_muon_total))
                    
        """finding neutrino and muon momentum (needed for q3)"""
        # finding muon momentum
        px_muons = ak.to_numpy(ak.sum(arrs['FS_Muons'][..., 1], axis=1))
        py_muons = ak.to_numpy(ak.sum(arrs['FS_Muons'][..., 2], axis=1))
        pz_muons = ak.to_numpy(ak.sum(arrs['FS_Muons'][..., 3], axis=1))
        array_muon = np.column_stack((px_muons,py_muons,pz_muons))
        muon_p.append(array_muon)

        # finding neutrino momentum
        array_neutrino = ak.to_numpy(arrs['Enu'])            
        zeros = np.zeros((array_neutrino.shape[0], 2))
        neutrino_vectors = np.column_stack((zeros, array_neutrino))  # shape: (n_events, 3)
        neutrino_p.append(neutrino_vectors)

        """finding weights needed for histogram"""
        # making an array out of the weights
        EventWeight = ak.to_numpy(arrs['EventWeight'])
        GenScale = ak.to_numpy(arrs['GenScaleFactor'])

        # finding the efficancy (with imputs of Enu and q0)
        q0_temp = np_Enu - ak.to_numpy(E_muon_total)
        efficancy = hist_lookup_2d(np_Enu, q0_temp)

        weight = EventWeight * GenScale * efficancy
        weight_list.append(weight)        

    # concatenating the lists made in the for loop        
    final_muon = np.concatenate(muon_E)
    final_nu = np.concatenate(neutrino_E)
    final_neutrino_p = np.concatenate(neutrino_p)
    final_muon_p = np.concatenate(muon_p)

    q3_vector = final_neutrino_p - final_muon_p
    
    # defining the needed arrays for the histogram
    q3_array = np.sqrt(q3_vector[:,0]**2+q3_vector[:,1]**2+q3_vector[:,2]**2)
    q0_array = final_nu - final_muon
    
    # plotting the histogram
    x = q3_array
    y = q0_array
    counts, xedges, yedges, image =  plt.hist2d(x, y, bins=100,
                range=[[xmin,xmax], [ymin,ymax]])
    bin_width = (xedges[1]-xedges[0])*(yedges[1]-yedges[0])

    plt.hist2d(x, y, bins=100,
                range=[[xmin,xmax], [ymin,ymax]])

    # correcting the weights so that the graph is scaled by cross section
    final_weights = np.concatenate(weight_list)/(n_files*bin_width)
    for_weights = np.concatenate(weight_list)/n_files

    # plotting the 2d histogram
    plt.hist2d(x, y, bins=100, 
                range=[[xmin,xmax], [ymin,ymax]],
                weights=final_weights, cmap='Blues')
    plt.colorbar()
    plt.xlabel("q3 (GeV)")
    plt.ylabel("q0 (GeV)")
    plt.title(f"q0vq3 {value_proton}p{value_neutron}n{value_pion}pi {label} with efficancy")
    plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/with_efficancy/q0q3/q0q3_efficancy_{value_proton}p{value_neutron}n{value_pion}pi_CC_{label}.png")
    plt.show()
    plt.close()

    # return for making a table of the final weights to ensure they are scaled correctly
    sum_weights = np.sum(for_weights)
    return sum_weights

def weights_table():
    weights_1P0N_GENIEAR23 = plot_q0q3(1, 0, "test", path_to_GENIEAR23, 0, 3, 0,3)
    weights_1P1N_GENIEAR23 = plot_q0q3(1, 1, "test", path_to_GENIEAR23, 0, 3, 0,3)
    weights_2P0N_GENIEAR23 = plot_q0q3(2, 0, "test", path_to_GENIEAR23, 0, 3, 0,3)
    weights_0P0N_GENIEAR23 = plot_q0q3(0, 0, "test", path_to_GENIEAR23, 0, 3, 0,3)

    weights_1P0N_GENIEG21 = plot_q0q3(1, 0, "test", path_to_GENIEG21, 0, 3, 0,3)
    weights_1P1N_GENIEG21 = plot_q0q3(1, 1, "test", path_to_GENIEG21, 0, 3, 0,3)
    weights_2P0N_GENIEG21 = plot_q0q3(2, 0, "test", path_to_GENIEG21, 0, 3, 0,3)
    weights_0P0N_GENIEG21= plot_q0q3(0, 0, "test", path_to_GENIEG21, 0, 3, 0,3)

    weights_1P0N_GiBUU = plot_q0q3(1, 0, "test", path_to_GiBUU, 0, 3, 0,3)
    weights_1P1N_GiBUU = plot_q0q3(1, 1, "test", path_to_GiBUU, 0, 3, 0,3)
    weights_2P0N_GiBUU = plot_q0q3(2, 0, "test", path_to_GiBUU, 0, 3, 0,3)
    weights_0P0N_GiBUU = plot_q0q3(0, 0, "test", path_to_GiBUU, 0, 3, 0,3)

    data = [["1P0N", weights_1P0N_GENIEAR23, weights_1P0N_GENIEG21, weights_1P0N_GiBUU], ["1P1N", weights_1P1N_GENIEAR23, weights_1P1N_GENIEG21, weights_1P1N_GiBUU], 
            ["2P0N", weights_2P0N_GENIEAR23, weights_2P0N_GENIEG21, weights_2P0N_GiBUU], ["0P0N", weights_0P0N_GENIEAR23, weights_0P0N_GENIEG21, weights_0P0N_GiBUU]]
    columns = ("type", "weights GENIEAR23", "weights GENIEG21", "weights GIBUU")
    
    # making the table 
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')

    table = ax.table(
    cellText=data,
    colLabels=columns,
    edges="closed",
    loc="center",
    cellLoc="center")
 
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(columns))))
    table.scale(1, 1.8)
 
    plt.title("Cross-Section-Scaled Weights by Topology and Generator", fontsize=12, pad=15)
    plt.tight_layout()

    plt.show()
    plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/correctly_weighted_plots/Enu_restricted/table/table_weights.png")
    plt.close()
    return 0



         



     
     

    

