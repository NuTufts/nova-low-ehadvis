import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
from pathlib import Path

"""This file makes a 1D histogram of the avalible energy for a certian topography"""

GENIEG1810j_path = "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only"
GENIEG21_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GENIEAR23_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GiBUU_path = "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/FHCDeriveFlux50MeVwidth_ppfx/numuCC_only/"

def E_avail(value_proton, value_neutron, pions, label, xmin = 0, xmax = 2.5, dif = 0.2):
    Pi0_E = []
    Gamma_E = []
    Electron_E = []
    weight_list = []

    Proton_KE = []
    PiP_KE = []
    PiM_KE = []
    AntiProton_KE = [] # need to add mass to this
    
    # defining the path the files are in
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
        # finding the catagories in the ttree needed for this cacluation
        branches = ['FS_Pi0s','FS_Gammas','FS_Electrons','FS_Protons',
            'FS_Antiprotons','FS_PiPs','FS_PiMs','EventWeight','GenScaleFactor']
        arrs = tree.arrays(branches, f"(Enu<4.5) & (flagCC==True)", library="ak")
        """finding Pi0, Gamma (photos), and electron energies"""
        #Pi0 energy
        E_pi0_per_pi0 = arrs['FS_Pi0s'][..., 0]
        E_pi0_total = ak.sum(E_pi0_per_pi0, axis=1)
        Pi0_E.append(ak.to_numpy(E_pi0_total))

        #Gamma (photo) energy
        E_G_per_G = arrs['FS_Gammas'][..., 0]
        E_G_total = ak.sum(E_G_per_G, axis=1)
        Gamma_E.append(ak.to_numpy(E_G_total))

        #Electron energy
        E_elect_per_elect = arrs['FS_Electrons'][..., 0]
        E_electron_total = ak.sum(E_elect_per_elect, axis=1)
        Electron_E.append(ak.to_numpy(E_electron_total))

        """Finding Proton, PiP, PiM, and Antiproton KE"""        
        #defining variables needed for KE of these particles
        mProton = mAntiProton = 0.938 #GeV/c^2
        mPiP = mPiM = 0.140 #GeV/c^2

        #finding the KE of the Proton
        E_pro_per_pro = arrs['FS_Protons'][..., 0]
        proton_counts = ak.num(arrs["FS_Protons"], axis =1)

        E_proton_total = ak.sum(E_pro_per_pro, axis=1)
        KE_proton_total = ak.to_numpy(E_proton_total) - mProton * ak.to_numpy(proton_counts)
        Proton_KE.append(KE_proton_total)

        #finding the KE of the AntiProton
        E_apro_per_apro = arrs['FS_Antiprotons'][..., 0]
        antiproton_counts = ak.num(arrs['FS_Antiprotons'], axis=1)
        E_antiproton_total = ak.sum(E_apro_per_apro, axis=1)

        antiproton_counts_np = ak.to_numpy(antiproton_counts)
        KE_antiproton_total = ak.to_numpy(E_antiproton_total) - mAntiProton * antiproton_counts_np

        # add rest mass back only for events that actually have an antiproton
        has_antiproton = antiproton_counts_np > 0
        KE_antiproton_total = np.where(has_antiproton, KE_antiproton_total + mAntiProton, KE_antiproton_total)

        AntiProton_KE.append(KE_antiproton_total)

        #finding the KE of the PiP
        E_PP_per_PP = arrs['FS_PiPs'][..., 0]
        PP_counts = ak.num(arrs["FS_PiPs"], axis = 1)

        E_PiP_total = ak.sum(E_PP_per_PP, axis=1)
        KE_PiP_total = ak.to_numpy(E_PiP_total) - mPiP * ak.to_numpy(PP_counts)
        PiP_KE.append(KE_PiP_total)

        #finding the KE of the PiM
        E_PM_per_PM = arrs['FS_PiMs'][..., 0]
        PM_counts = ak.num(arrs["FS_PiMs"], axis = 1)

        E_PiM_total = ak.sum(E_PM_per_PM, axis=1)
        KE_PiM_total = ak.to_numpy(E_PiM_total) - mPiM *ak.to_numpy(PM_counts)
        PiM_KE.append(KE_PiM_total)

        """finding weights needed for histogram"""
        # making an array out of the weights
        EventWeight = ak.to_numpy(arrs['EventWeight'])
        GenScale = ak.to_numpy(arrs['GenScaleFactor'])
        weight = EventWeight * GenScale
        weight_list.append(weight)        
    
    print("done processing files")
    # assembling final energies into np array
    final_Pi0 = np.concatenate(Pi0_E)
    final_Gamma = np.concatenate(Gamma_E)
    final_Electron = np.concatenate(Electron_E)
    print(f"Pi:{len(final_Pi0)}, Gamma:{len(final_Gamma)}, Electron:{len(final_Electron)}")

    # assembling final KE into np array
    final_Proton = np.concatenate(Proton_KE)
    final_PiP = np.concatenate(PiP_KE)
    final_PiM = np.concatenate(PiM_KE)
    final_antiproton = np.concatenate(AntiProton_KE) # added mass in for loop for antiprotons (make sure correct)
    print(f"Proton{len(final_Proton)}, PiP:{len(final_PiP)}, PiM:{len(final_PiM)}, Antiproton:{len(final_antiproton)}")

    final_weights = np.concatenate(weight_list)/(n_files*dif)

    # E_avail = KE_proton + KE_PiP + KE_PiM + KE_antiproton + E_Pi0 + E_Gamma + E_Electron
    if pions == 0:
        E_avail = (final_Gamma + final_Electron
                  + final_Proton + final_antiproton)
    else:
        E_avail = (final_Pi0 + final_Gamma + final_Electron
                  + final_Proton + final_PiP + final_PiM
                  + final_antiproton)
    
    """plotting the histogrma for E_avail"""
    final_array = E_avail

    bins = np.arange(xmin,xmax,dif)
    #filterig the array
    mask = (final_array >= xmin) & (final_array < xmax)
    filtered_array = final_array[mask]
    filtered_weights = final_weights[mask]
    fig, ax = plt.subplots(figsize=(9, 5))
    centers = (bins[:-1] + bins[1:]) / 2

    labels = [f"{x:.1f}" for x in bins[:-1]]

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation = 45)
    ax.set_xlabel("Energy (GeV)", fontsize=12)
    ax.set_ylabel("flux average differential cross section (cm^2/GeV)", fontsize=12)

    _, bins, patches = plt.hist([filtered_array],
                            bins=bins, weights = filtered_weights, color=["#380000"], label=['E_avail'])

    plt.xlim([bins[0],bins[-1]])
    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.title(f'histogram of E_avail inclusive {label}')
    plt.savefig(f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/E_avail/E_avail_inclusive_{label}.png")
    plt.show()
    plt.close()
    return 0
