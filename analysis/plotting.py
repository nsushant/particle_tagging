import pynbody
import pandas as pd
import numpy as np
import darklight 
import matplotlib.pyplot as plt
import sys
import tangos
import os
import matplotlib.style
import matplotlib as mpl
import seaborn as sns
from particle_tagging_package.tagging.utils import *

mpl.rcParams.update({'text.usetex': False})


def edge_plot_tagged_vs_hydro_mass_dist(name_of_DMO_simulation,name_of_HYDRO_simulation, file_with_tagged_particles, time_to_plot, plot_type='2D Mass Distribution'):
    
    tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
    pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
    
    # finding mass distribution of tagged particles in DMO simulation 

    DMOsim, main_halo, halonums, outputs = load_tangos_data(name_of_DMO_simulation,0)

    times_tangos = main_halo.calculate_for_progenitors('t()')[0][::-1]
    
    output_number = np.where(times_tangos <= time_to_plot)[0][-1]
  
    d = pd.read_csv(file_with_tagged_particles)

    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()

    tagged_iords = dt.index.values
    tagged_m = dt['mstar'].values

    s = pynbody.load(os.path.join(pynbody_path,name_of_DMO_simulation,outputs[output_number]))
    s.physical_units()
  

    h = s.halos()[1]
    
    pynbody.analysis.halo.center(h)

    r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')

    selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]


    idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

    selected_masses = [tagged_m[i] for i in idxs_m]

    data_all_tagged = pd.DataFrame({'x':selected_parts['x'],'y':selected_parts['y'], 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = pd.DataFrame({'r':np.sqrt(selected_parts['x']**2+selected_parts['y']**2+selected_parts['z']**2), 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

    dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)

    
    # plotting hydro particles 

    sim, HYDRO_main_halo, HYDRO_halonums, outputs_HYDRO = load_tangos_data(name_of_HYDRO_simulation,0)
    
    times_tangos_HYDRO = HYDRO_main_halo.calculate_for_progenitors('t()')[0][::-1]
    
    output_number_HYDRO = np.where(times_tangos_hydro <= time_to_plot)[0][-1]
 
    s_hydro = pynbody.load(os.path.join(pynbody_path,name_of_HYDRO_simulation,outputs_HYDRO[output_number_HYDRO]))
    
    s_hydro.physical_units()
    
    h_hydro = s_hydro.halos()[1]
    
    #reff_hydro = dt_ahf_hydro['reff'].values
    
    pynbody.analysis.halo.center(h_hydro)
    
    stars = h_hydro.st
    
    data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})
        
    
    if plot_type == '2D Mass Distribution':

      plt.gca().set_box_aspect(1)
      
      sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,cmap="viridis",levels=5)
      sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='masses',fill=False,color='white',levels=5)
       
      plt.xlim(4,-4)
      plt.ylim(4,-4)

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
  
    if plot_type == '1D Mass Distribution':
      
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
    
      plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel('Mass in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')
    
    return 


def plot_tagged_vs_hydro_mass_dist(DMO_halo_particles, HYDRO_halo_particles, file_with_tagged_particles, time_to_plot, plot_type='2D Mass Distribution'):

    d = pd.read_csv(file_with_tagged_particles)

    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()

    tagged_iords = dt.index.values
    tagged_m = dt['mstar'].values
    
    h = DMO_halo_particles
    
    pynbody.analysis.halo.center(h)

    r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')

    selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]


    idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

    selected_masses = [tagged_m[i] for i in idxs_m]

    data_all_tagged = pd.DataFrame({'x':selected_parts['x'],'y':selected_parts['y'], 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = pd.DataFrame({'r':np.sqrt(selected_parts['x']**2+selected_parts['y']**2+selected_parts['z']**2), 'masses':np.asarray(selected_masses)})


    dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

    dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)

    
    # plotting hydro particles 
    
    h_hydro = HYDRO_halo_particles
    
    pynbody.analysis.halo.center(h_hydro)
    
    stars = h_hydro.st
    
    data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})
        
    
    if plot_type == '2D Mass Distribution':

      plt.gca().set_box_aspect(1)
      
      sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,cmap="viridis",levels=5)
      sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='masses',fill=False,color='white',levels=5)
       
      plt.xlim(4,-4)
      plt.ylim(4,-4)

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
  
    if plot_type == '1D Mass Distribution':
      
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
    
      plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel('Mass in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')
    
    return 



def edge_plot_tagged_vs_hydro_angmom_dist():
    
    return 



def plot_tagged_vs_hydro_angmom_dist():
    
    return 




