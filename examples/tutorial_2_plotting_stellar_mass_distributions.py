import pynbody

import particle_tagging as ptag 
import matplotlib.pyplot as plt 

# time in gyr 
time_to_plot = 14 

# DMO = dark matter only 
# load in the particle data for the DMO main halo 
particle_data_DMO = pynbody.load('Particle_data/Halo1459_DMO')

particle_data_DMO.physical_units()

main_halo_DMO = particle_data_DMO.halos()[1]

# loading in particle data for the HYDRO main halo 
particle_data_HYDRO = pynbody.load('Particle_data/Halo1459_fiducial')

particle_data_HYDRO.physical_units()

main_halo_HYDRO = particle_data_HYDRO.halos()[1]

# Plotting 2D tagged stellar mass distribution Vs the @D stellar mass distribution in the HYDRO sim 
ptag.analysis.plotting.plot_tagged_vs_hydro_mass_dist(main_halo_DMO, main_halo_HYDRO, 'file_with_tagged_particles.csv', time_to_plot, plot_type='2D Mass Distribution') 
plt.clf()

# Plotting 1D tagged stellar mass distribution Vs the @D stellar mass distribution in the HYDRO sim 
ptag.analysis.plotting.plot_tagged_vs_hydro_mass_dist(main_halo_DMO, main_halo_HYDRO, 'file_with_tagged_particles.csv', time_to_plot, plot_type='1D Mass Distribution')
plt.clf()

# Alternatively, to plot the stellar mass enclosed over radial distance  
ptag.analysis.plotting.plot_tagged_vs_hydro_mass_dist(main_halo_DMO, main_halo_HYDRO, 'file_with_tagged_particles.csv', time_to_plot, plot_type='1D Mass Enclosed') 
plt.clf()

