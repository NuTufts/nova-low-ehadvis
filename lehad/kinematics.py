
import awkward as ak

# Map branch name to the mass in GeV of the particle correpsonding to that branch.
PARTICLE_MASSES_BY_BRANCH_NAME = {
  'FS_Protons' : 0.938,
  'FS_Neutrons' : 0.940
}

# Compute for each event the sum of kinetic energies for a particle species.
# Pass the "events" dict mapping branch name -> branch.
# Pass the name of a branch: e.g. 'FS_Protons' or 'FS_Neutrons'.
def ke_sum(events, branch_name):
  particle_energies = events[branch_name][:,:,0]
  kinetic_energies = particle_energies[...] - PARTICLE_MASSES_BY_BRANCH_NAME[branch_name]
  return ak.sum(kinetic_energies,axis=-1)
  
# Convenience method for summing proton KE.
def proton_ke_sum(events): return ke_sum(events,'FS_Protons')
# Convenience method for summing neutron KE.
def neutron_ke_sum(events): return ke_sum(events,'FS_Neutrons')

