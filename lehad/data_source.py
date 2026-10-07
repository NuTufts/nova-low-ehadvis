
from enum import Enum
from pathlib import Path

"""
Unified place to get generator prediction filepaths.
"""

# Shorthand tags for encoding generator configurations.
class GeneratorType(Enum):
  G18 = 1 # v3.00.06_G18_10j_00_000.
  G21 = 2 # v3.04.00_G21_11a_00_000.
  AR23 = 3 # v3.04.00_AR23_20i_00_000.
  GiBUU = 4 # R2025_03_Defaultparams2025.

# Use me to choose a flux prediction.
class BeamType(Enum):
  FHC = 1 # Forward Horn Current using the DeriveFlux50MeVwidth_ppfx flux prediction.
  RHC = 2 # Reverse Horn Current using the DeriveFlux50MeVwidth_ppfx flux prediction.

# Use me to choose a target neutrino type.
class NeutrinoType(Enum):
  Numu = 1
  Nue = 2
  Numubar = 3
  Nuebar = 4

# Map GeneratorType -> primary directory containing files.
# There is a consistent subdirectory structure splitting predictions on horn current and neutrino type.
DIRECTORIES_BY_GENERATOR_TYPE = {
  GeneratorType.G18 : "/cluster/tufts/lblnu/nova/data/generators/v3.00.06_G18_10j_00_000/",
  GeneratorType.G21 : "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_G21_11a_00_000/",
  GeneratorType.AR23 : "/cluster/tufts/lblnu/nova/data/generators/v3.04.00_AR23_20i_00_000/",
  GeneratorType.GiBUU : "/cluster/tufts/lblnu/nova/data/generators/R2025_03_Defaultparams2025/",
}

# Map BeamType -> subdirectory name.
SUBDIRECTORIES_BY_BEAM_TYPE = {
  BeamType.FHC : "FHCDeriveFlux50MeVwidth_ppfx",
  BeamType.RHC : "RHCDeriveFlux50MeVwidth_ppfx",
}

# Map NeutrinoType -> subdirectory name.
SUBDIRECTORIES_BY_NEUTRINO_TYPE = {
  NeutrinoType.Numu : "numu_only",
  NeutrinoType.Nue : "nue_only",
  NeutrinoType.Numubar : "numubar_only",
  NeutrinoType.Nuebar : "nuebar_only",
}

# Get filepaths for this generator type.
# Param generator_type: Expect an element of the GeneratorType enum.
# Param beam_type: Expect an element of the BeamType enum.
# Param neutrino_type: Expect an element of the NeutrinoType enum.
# Returns a glob of filepaths that represent the prediction from this generator.
def get_generator_filepaths(generator_type,beam_type,neutrino_type):
  directory = DIRECTORIES_BY_GENERATOR_TYPE[generator_type] + "/" \
            + SUBDIRECTORIES_BY_BEAM_TYPE[beam_type] + "/" \
            + SUBDIRECTORIES_BY_NEUTRINO_TYPE[neutrino_type]
  return list(Path(directory).glob("*.root"))

