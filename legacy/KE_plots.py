import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
from pathlib import Path

"""This file makes 2 plots: a 1d KE histogram (which right now will not actually save the figure made, that line is commented out), and a KE Comparison between generators"""

#GENIEG1810j_path = "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only"
#GENIEG21_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
#GENIEAR23_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
#GiBUU_path = "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/FHCDeriveFlux50MeVwidth_ppfx/numuCC_only/"

GENIEG1810j_path = "./small_sim/G18/"
GENIEG21_path = "./small_sim/G21/"
GENIEAR23_path = "./small_sim/AR23/"
GiBUU_path = "./small_sim/GiBUU/"

def KE(value_proton, value_neutron, value_pions, label, variable, xmin=0, xmax=3, dif=0.2, proton_mass = 0.938, neutron_mass = 0.940): 
    array_proton = []
    array_neutron = []
    weight_list = []
    # if there are 2 protons (else just remain empty list)
    array_proton1 = []
    array_proton2 = []

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
        cuts = f"(Enu<4.5) & (flagCC==True) &  (n_FS_Muons==1)& (n_FS_Protons=={value_proton}) & (n_FS_Neutrons>=1) & (n_FS_PiPs==0)& (n_FS_PiMs==0)& (n_FS_Pi0s==0)"       

        # finding proton energy
        if value_proton==1: 
            Proton_tree_array = tree.arrays(['FS_Protons'],
                                cuts,
                                library="ak") #this is not a 4 vector, but returns dict
            
            E_proton_per_proton = Proton_tree_array['FS_Protons'][..., 0]
            # sum over the proton axis (axis=1) within each event
            E_proton_total = ak.sum(E_proton_per_proton, axis=1)

            array_proton.append(ak.to_numpy(E_proton_total))

        if value_proton==2: 
            Proton_tree_array = tree.arrays(['FS_Protons'],
                                cuts,
                                library="ak") #this is not a 4 vector, but returns dict
            E_proton_per_proton = Proton_tree_array['FS_Protons'][..., 0]
            
            # total KE of both protons
            E_proton_total = ak.sum(E_proton_per_proton, axis=1)
            array_proton.append(ak.to_numpy(E_proton_total))

            #KE of leading proton
            E_p1 = E_proton_per_proton[...,0]
            E_p2 = E_proton_per_proton[...,1]

            array_proton1.append(ak.to_numpy(E_p1))
            array_proton2.append(ak.to_numpy(E_p2))


        #finding neutron energy
        Neutron_tree_array = tree.arrays(['FS_Neutrons'],
                                cuts,
                                library="ak") #this is not a 4 vector, but returns dict
        E_neutron_per_neutron = Neutron_tree_array['FS_Neutrons'][..., 0]  # energy component

        # total KE of neutrons per event
        neutron_counts = ak.num(Neutron_tree_array['FS_Neutrons'], axis=1)

        E_neutron_total = ak.sum(E_neutron_per_neutron, axis=1)
        if value_neutron == 0:
            KE_neutron_total = 0
        else:
            KE_neutron_total = ak.to_numpy(E_neutron_total) - neutron_mass * ak.to_numpy(neutron_counts)
            array_neutron.append(KE_neutron_total)     

        """finding weights needed for histogram"""
        # making an array out of the weights
        EventWeight = tree.arrays(['EventWeight'], 
                                      cuts,
                                      library='np')
        GenScale = tree.arrays(['GenScaleFactor'],
                                    cuts,
                                     library='np')
        weight = EventWeight['EventWeight'] * GenScale['GenScaleFactor']
        weight_list.append(weight)       

    # building the arrays outside of the for loop
    final_weights = np.concatenate(weight_list)/(n_files*dif)

    if value_proton == 1: 
        proton_energy = np.concatenate(array_proton)
        KE_proton = proton_energy - proton_mass

        KE_p1 = 0
        KE_p2 = 0
        KE_tot = 0
    elif value_proton == 2:
        proton1_energy = np.concatenate(array_proton1)
        proton2_energy = np.concatenate(array_proton2)
        total_pE = np.concatenate(array_proton)

        KE_p1 = proton1_energy - proton_mass
        KE_p2 = proton2_energy - proton_mass
        KE_proton = total_pE - 2*proton_mass
    else:
        KE_proton = np.array([])
        KE_p1 = np.array([])
        KE_p2 = np.array([])
        KE_tot = np.array([])
    
    if value_neutron == 0:
        KE_neutron = np.array([])
    else:
        KE_neutron = np.concatenate(array_neutron)

    options = {
    "proton": KE_proton,
    "neutron": KE_neutron,
    "p1": KE_p1,
    "p2": KE_p2}

    if variable not in options:
        raise ValueError(f"variable must be one of {list(options.keys())}, got '{variable}'")

    # defining the array that will be plotted

    final_array = options[variable]

    # filterig the array so that it fits in the plot boundries 
    bins = np.arange(xmin,xmax,dif)
    mask = (final_array >= xmin) & (final_array < xmax)
    filtered_array = final_array[mask]

    # making the histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    _, bins, patches = plt.hist([filtered_array],
                            bins=bins, weights = final_weights[mask], color=["#003807"], label=[f'KE {label}'])

    centers = (bins[:-1] + bins[1:]) / 2

    labels = [f"{x:.1f}" for x in bins[:-1]]   # use 1-decimal labels, not int()

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation = 45)

    plt.xlim([bins[0],bins[-1]])

    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.title(f'histogram of P{value_proton}N{value_neutron}P{value_pions} {variable} {label} 0PiCC_Enu<4.5')
    # uncomment this line if actually want to save the 1d histogram
    plt.savefig(f"./plots/histogram_KE/histogram_KE_N{value_neutron}P{value_proton}Pi{value_pions}_{variable}_0MuCC_EnuLess4Half_{label}.png")
    plt.close()
    return final_array, final_weights, bins

def plot_KE_comparison(value_proton, value_neutron, value_pions, variable, xlabel="KE (GeV)", ylabel="flux average differential cross section (cm^2/GeV)"):
    #defining colors of specific generators used in histogram
    GENERATOR_COLORS = {
    "GENIEAR23": "b",
    "GiBUU": "g",
    "GENIEG21": "r",
    "GENIEG1810j": "y"}

    #reconstructing arrays for different generators using the KE function above
    model_outputs = {
    "genieG21_out": KE(value_proton, value_neutron, value_pions, "GENIEG21", variable, xmin=0, xmax=.4, dif=0.02),
    "genieAR23_out": KE(value_proton, value_neutron, value_pions, "GENIEAR23", variable, xmin=0, xmax=.4, dif=0.02),
    "gibuu_out": KE(value_proton, value_neutron, value_pions, "GiBUU", variable, xmin=0, xmax=.4, dif=0.02),
    "genieG1810j": KE(value_proton, value_neutron, value_pions, "GENIEG1810j", variable, xmin = 0, xmax=.4, dif=0.02)
    }
    #making the graph
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = None
    for name, (values, weights, model_bins) in model_outputs.items():
        if bins is None:
            bins = model_bins
        elif not np.allclose(bins, model_bins):
            raise ValueError(f"Bin edges for '{name}' don't match the others — "
                              f"call KE() with the same xmin/xmax/dif for all models.")

        counts, _ = np.histogram(values, bins=bins, weights=weights)
        color = GENERATOR_COLORS[name] if name in GENERATOR_COLORS else None
        ax.step(bins, np.append(counts, counts[-1]), where='post',
                linewidth=1.8, color=color, label=name)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(f"KE comparison between generators {variable} {value_proton}p{value_neutron}n{value_pions}pi", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/GenCompair/GenCompair_{variable}_{value_proton}p{value_neutron}n{value_pions}pi.png")
    plt.close()
    return 0
