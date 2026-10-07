import uproot
import numpy as np
import dataframe_image as dfi
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
from pathlib import Path

"""This file makes a table of the interaction types stored in the ttree file"""

GENIEG1810j_path = "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only"
GENIEG21_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GENIEAR23_path = "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/FHCDeriveFlux50MeVwidth_ppfx/numu_only/"
GiBUU_path = "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/FHCDeriveFlux50MeVwidth_ppfx/numuCC_only/"

def interaction_type(value_protons, value_neutrons, pions, label):
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

    # Find all root files in that folder and iterate over them
    for file_path in folder.glob("*.root"):        
        n_files = len(list(folder.glob("*.root")))
        print(f"Processing: {file_path.name} out of {n_files} files")
        f = uproot.open(file_path)
        tree = f["generator_data"]
        
        branches = [var,'EventWeight','GenScaleFactor']
        arr = tree.arrays(branches,
                     f'(Enu>4.5) & (flagCC==True) & (n_FS_Muons==1)',
                     library="np")
        array = arr[var]
        weight = arr['EventWeight'] * arr['GenScaleFactor']

        weight_list.append(weight)
        array_list.append(array)

    # def full array of variable
    weight_array = np.concatenate(weight_list, axis=0)/(n_files)
    final_array = np.concatenate(array_list, axis=0)

    # defining the general labels
    interaction_category = {
    # CC Elastic
    1: "CC Elastic", 2: "CC Elastic",
    # CC RES 1pi
    11: "CC RES 1pi", 12: "CC RES 1pi", 13: "CC RES 1pi",
    15: "CC RES 1pi", 16: "CC RES 1pi",
    # CC RES 1gamma
    17: "CC RES 1gamma",
    # CC Single K
    18: "CC Single K", 19: "CC Single K", 20: "CC Single K",
    # CC Multi-pi
    21: "CC Multi-pi",
    # CC RES 1eta
    22: "CC RES 1eta",
    # CC RES Single K (strange baryon)
    23: "CC RES Single K",
    # CC DIS
    26: "CC DIS",

    # NC RES 1pi
    31: "NC RES 1pi", 32: "NC RES 1pi", 33: "NC RES 1pi",
    34: "NC RES 1pi", 35: "NC RES 1pi", 36: "NC RES 1pi",
    # NC RES 1gamma
    38: "NC RES 1gamma", 39: "NC RES 1gamma",
    # NC Multi-pi
    41: "NC Multi-pi",
    # NC RES 1eta
    42: "NC RES 1eta", 43: "NC RES 1eta",
    # NC Single K
    44: "NC Single K", 45: "NC Single K",
    # NC DIS
    46: "NC DIS",
    # NC Elastic
    51: "NC Elastic", 52: "NC Elastic",

    # Antineutrino CC Elastic
    -1: "CCbar Elastic", -2: "CCbar Elastic",
    # Antineutrino CC RES 1pi
    -11: "CCbar RES 1pi", -12: "CCbar RES 1pi", -13: "CCbar RES 1pi",
    -15: "CCbar RES 1pi", -16: "CCbar RES 1pi",
    # Antineutrino CC RES 1gamma
    -17: "CCbar RES 1gamma",
    # Antineutrino CC Multi-pi
    -21: "CCbar Multi-pi",
    # Antineutrino CC RES 1eta
    -22: "CCbar RES 1eta",
    # Antineutrino CC RES Single K
    -23: "CCbar RES Single K",
    # Antineutrino CC DIS
    -26: "CCbar DIS",

    # Antineutrino NC RES 1pi
    -31: "NCbar RES 1pi", -32: "NCbar RES 1pi", -33: "NCbar RES 1pi",
    -34: "NCbar RES 1pi", -35: "NCbar RES 1pi", -36: "NCbar RES 1pi",
    # Antineutrino NC RES 1gamma
    -38: "NCbar RES 1gamma", -39: "NCbar RES 1gamma",
    # Antineutrino NC Multi-pi
    -41: "NCbar Multi-pi",
    # Antineutrino NC RES 1eta
    -42: "NCbar RES 1eta", -43: "NCbar RES 1eta",
    # Antineutrino NC Single K
    -44: "NCbar Single K", -45: "NCbar Single K",
    # Antineutrino NC DIS
    -46: "NCbar DIS",
    # Antineutrino NC Elastic
    -51: "NCbar Elastic", -52: "NCbar Elastic"}

    # defining the interaction labels
    interaction_labels = {
    1: "CC Elastic (NEU,N→L-,P)",
    2: "CC Elastic (NEU,N+X→L-,P+X)",
    11: "CC RES 1pi (NEU,P→L-,P,PI+)",
    12: "CC RES 1pi (NEU,N→L-,P,PI0)",
    13: "CC RES 1pi (NEU,N→L-,N,PI+)",
    15: "CC RES 1pi diffractive",
    16: "CC RES 1pi coherent (O16)",
    17: "CC RES 1gamma",
    18: "CC Single K (NEU,N→L-,N,K+)",
    19: "CC Single K (NEU,N→L-,P,K0)",
    20: "CC Single K (NEU,N→L-,P,K+)",
    21: "CC Multi-pi (1.3 to 2.0 GeV)",
    22: "CC RES 1eta",
    23: "CC RES Single K (Lambda)",
    26: "CC DIS (W above 2.0)",
    31: "NC RES 1pi (N→N,PI0)",
    32: "NC RES 1pi (P→P,PI0)",
    33: "NC RES 1pi (N→P,PI-)",
    34: "NC RES 1pi (P→N,PI+)",
    35: "NC RES 1pi diffractive",
    36: "NC RES 1pi coherent (O16)",
    38: "NC RES 1gamma (N)",
    39: "NC RES 1gamma (P)",
    41: "NC Multi-pi (1.3 to 2.0 GeV)",
    42: "NC RES 1eta (N)",
    43: "NC RES 1eta (P)",
    44: "NC Single K (Lambda,K0)",
    45: "NC Single K (Lambda,K+)",
    46: "NC DIS (W above 2.0)",
    51: "NC Elastic (P→P)",
    52: "NC Elastic (N→N)",
    -1: "CCbar Elastic",
    -2: "CCbar Elastic (+X)",
    -11: "CCbar RES 1pi",
    -12: "CCbar RES 1pi",
    -13: "CCbar RES 1pi",
    -15: "CCbar RES 1pi diffractive",
    -16: "CCbar RES 1pi coherent",
    -17: "CCbar RES 1gamma",
    -21: "CCbar Multi-pi",
    -22: "CCbar RES 1eta",
    -23: "CCbar Single K",
    -26: "CCbar DIS",
    -31: "NCbar RES 1pi",
    -32: "NCbar RES 1pi",
    -33: "NCbar RES 1pi",
    -34: "NCbar RES 1pi",
    -35: "NCbar RES 1pi diffractive",
    -36: "NCbar RES 1pi coherent",
    -38: "NCbar RES 1gamma",
    -39: "NCbar RES 1gamma",
    -41: "NCbar Multi-pi",
    -42: "NCbar RES 1eta",
    -43: "NCbar RES 1eta",
    -44: "NCbar Single K",
    -45: "NCbar Single K",
    -46: "NCbar DIS",
    -51: "NCbar Elastic",
    -52: "NCbar Elastic"}

    # defining the impputs that will be used to make the table
    codes, counts = np.unique(final_array, return_counts=True)
    df_events = pd.DataFrame({"code": final_array, "weight": weight_array})
    weighted = df_events.groupby("code")["weight"].sum().reset_index()

    # making the pandas data table
    df_raw = pd.DataFrame({
    "code": codes,
    "label": [interaction_labels.get(c, f"UNKNOWN ({c})") for c in codes],
    "catagory": [interaction_category.get(c, f"UNKNOWN ({c})") for c in codes],
    "flux average cross section": weighted["weight"]
    })

    df = df_raw.style.format({"flux average cross section": "{:.3e}"}).set_caption(f"Interaction type High Energy {value_protons}p{value_neutrons}n{pions}pi {label}")

    # Saving the pandas data table as a PNG
    dfi.export(df, f"/cluster/tufts/lblnu/nova/app/users/zfulto01/plots/presentation_plots/Interaction_tables/IntType_{value_protons}p{value_neutrons}n{pions}pi_{label}_highenergy.png", table_conversion="matplotlib")
    return df_raw
