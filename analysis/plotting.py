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
from particle_tagging.tagging.utils import *
from particle_tagging.analysis.calculate import *

mpl.rcParams.update({'text.usetex': False})

pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]


def edge_plot_tagged_vs_hydro_mass_dist(name_of_DMO_simulation, name_of_HYDRO_simulation, file_with_tagged_particles, time_to_plot, plot_type='2D Mass Distribution',label=None):
    
    tangos_path     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
    pynbody_path    = '/vol/ph/astro_data/shared/morkney/EDGE/'
    
    # finding mass distribution of tagged particles in DMO simulation 

    split = name_of_DMO_simulation.split('_')

    halo_shortname = split[0]

    tangos.core.init_db(tangos_path+halo_shortname+'.db') 

    DMOsim = tangos.get_simulation(name_of_DMO_simulation)
    
    t_all,red_all,main_halo, halonums, outputs = load_indexing_data(DMOsim,1)

    times_tangos = main_halo.calculate_for_progenitors('t()')[0][::-1]

    time_to_plot = times_tangos[np.argmin(abs(times_tangos-time_to_plot))]
    
    output_number = np.where(times_tangos <= time_to_plot)[0][-1]
  
    d = pd.read_csv(file_with_tagged_particles)

    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()

    tagged_iords = dt.index.values
    tagged_m = dt['mstar'].values

    s = pynbody.load(os.path.join(pynbody_path,name_of_DMO_simulation,outputs[output_number]))
    s.physical_units()
  

    h = s.halos()[int(halonums[output_number] - 1)]
    
    pynbody.analysis.halo.center(h.dm)
    #pynbody.analysis.angmom.faceon(h.dm)
    r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')

    selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]


    idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

    selected_masses = [tagged_m[i] for i in idxs_m]
    
    '''
    st_ages = calc_ages(d[ d['t'] <= time_to_plot ],time_to_plot)

    grouped_first = d[ d['t'] <= time_to_plot ].groupby(['iords']).first()
    
    ages_df = pd.DataFrame({'ages':st_ages , 'iords':grouped_first.index.values}).groupby(['iords']).last()

    ordered_ages = np.asarray([ ages_df.loc[part_id]['ages'] for part_id in selected_parts['iord'] ])
    
    lums = calc_luminosity(ordered_ages)
                                                                    
    '''
    data_all_tagged = pd.DataFrame({ 'x':selected_parts['x'], 'y':selected_parts['y'], 'masses':np.asarray(selected_masses) })
    #, 'lums':lums, 'ages':ordered_ages })


    dataframe_for_hist = pd.DataFrame({'r':selected_parts['r'], 'masses': np.asarray(selected_masses) })
    #, 'ages': np.asarray(ordered_ages)})

    print(data_all_tagged.head())
    
    dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

    dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)

    
    # plotting hydro particles 

    sim = tangos.get_simulation(name_of_HYDRO_simulation)
    t_all_H,red_all_h, HYDRO_main_halo, HYDRO_halonums, outputs_HYDRO = load_indexing_data(sim,1)

    
    times_tangos_HYDRO = HYDRO_main_halo.calculate_for_progenitors('t()')[0][::-1]

    halonums_hydro = HYDRO_main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
    
    output_number_HYDRO = np.where(times_tangos_HYDRO <= time_to_plot)[0][-1]
 
    s_hydro = pynbody.load(os.path.join(pynbody_path,name_of_HYDRO_simulation,outputs_HYDRO[output_number_HYDRO]))
    
    s_hydro.physical_units()
    
    h_hydro = s_hydro.halos()[int(halonums_hydro[output_number_HYDRO] - 1)]
    
    print(s_hydro[0].st,s_hydro[1].st)
    
    pynbody.analysis.halo.center(h_hydro)

    #pynbody.analysis.angmom.faceon(h_hydro.st)

    r200_hydro = pynbody.analysis.halo.virial_radius(h_hydro, overden=200, r_max=None, rho_def='critical')

    stars = s_hydro.st[get_dist(s_hydro.st['pos'])<=r200_hydro]
    
    data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})
    print(data_all_stars.head())
    
    if plot_type == '2D Mass Distribution':

      plt.gca().set_box_aspect(1)
      
      sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,levels=5,cmap="viridis")#,cbar=True) #label='Tagged Stellar Mass')
      sns.kdeplot(data = data_all_stars, x ='x',y='y', weights='masses',fill=False,levels=5,color='white')#,cbar=True)
       
      plt.xlim(4,-4)
      plt.ylim(4,-4)
      
      #plt.colorbar()
      
      plt.title(str(name_of_HYDRO_simulation))
  
    if plot_type == '1D Mass Enclosed':
      
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values, label=label)
    
      #plt.xlim(0,1)
      #plt.xlim(0.1,None)
      plt.ylim(300,None)
      
      #plt.yscale('log')
      #plt.xscale('log')

      plt.title(str(name_of_HYDRO_simulation))
      
      plt.ylabel(' $M_{star}(<r)$ in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')

    if plot_type == '1D Mass Distribution':
      
      plt.hist(dataframe_for_hist['r'].values,weights=dataframe_for_hist['masses'].values,bins=15)
    
      #plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation))
      
      plt.ylabel('Stellar Mass in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')

    if plot_type == '1D Ages Hist':
        plt.hist(dataframe_for_hist['ages'].values,histtype='step',weights=dataframe_for_hist['masses'].values)

        #plt.xlim(0,1)
        
        plt.title(str(name_of_HYDRO_simulation))
        
        plt.title('Stellar Age in Gyr')
                                    

    if plot_type == 'Median Age Vs Radius':

        bins = np.arange(0,1,1/10)
        
        dataframe_for_hist['r_bins'] = pd.cut(dataframe_for_hist['r'],bins)

        df_grp = dataframe_for_hist.groupby(['r_bins']).median()

        df_r_vals = [i.left for i in df_grp.index.values]
        
        plt.bar(df_r_vals,df_grp['ages'])

        plt.xlabel('Radial Dist. in kpc')

        plt.ylabel('Median stellar age Gyr')
        
    '''
    if plot_type == '2D Luminosity Distribution':
        plt.gca().set_box_aspect(1)

        sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='lums',fill=True,cmap="viridis",levels=5)
        sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='lums',fill=False,color='white',levels=5)
        
        plt.xlim(4,-4)
        plt.ylim(4,-4)
        
        plt.title(str(name_of_HYDRO_simulation))
    '''                                         
    
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
  
    if plot_type == '1D Mass Enclosed':
        
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
      plt.xlim(0,1)
      
      plt.yscale('log')
      plt.xscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel(' $M_{star}(<r)$ in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')

    if plot_type == '1D Mass Distribution':
      
      plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
    
      plt.xlim(0,1)

      plt.yscale('log')

      plt.title(str(name_of_HYDRO_simulation)+"t = "+str(time_to_plot))
      
      plt.ylabel('Stellar Mass in $M_{\odot}$')
      plt.xlabel('Radial Distance in Kpc')
    
    return 




def edge_plot_tagged_vs_hydro_angmom_dist():
    print('not yet implemented')
    return 



def plot_tagged_vs_hydro_angmom_dist(DMO_halo_particles,HYDRO_halo_particles,file_with_tagged_particles,time_to_plot):
    
    h = HYDRO_halo_particles
    s = h.st 
    # centering on most massive halo                                                                                                                                                                                                                               
    
    pynbody.analysis.halo.center(h)
    r200_HYDRO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')
    
    stars = s.st[ np.sqrt(s.st['x']**2+s.st['y']**2+s.st['z']**2) <= r200_HYDRO ] 
                                                                                                                                                                                                                  
    rdists = np.sqrt(stars['x']**2+stars['y']**2+stars['z']**2)
    
    jstars = np.sqrt(stars['j'][:,0]**2+stars['j'][:,1]**2+stars['j'][:,2]**2)
    
    #create dataframe                                                                                                                                                                                                                                              
    
    df = pd.DataFrame({'r':rdists,'j':jstars, 'mass':stars['mass']})
    
    s_tagged = DMO_halo_particles

    # Stellar Mass weighted median at each radial distance  for tagged particles                                                                                                                                                                                   
    
    d = pd.read_csv(file_with_tagged_particles)
    dt = d[ d['t'] <= time_to_plot ].groupby(['iords']).last()
    
    tagged_iords = dt.index.values
    
    tagged_particles = s_tagged[np.where(np.isin(s_tagged['iord'],tagged_iords)==True)]

    #print(tagged_stars[0]['iord'], np.where(tagged_masses.index.values == tagged_stars[0]['iord']) )                                                                                                                                                              
    
    #print(tagged_masses.loc[tagged_stars[0]['iord']])                                                                                                                                                                                                             
    
    jtagged = np.sqrt(tagged_particles['j'][:,0]**2+tagged_particles['j'][:,1]**2+tagged_particles['j'][:,2]**2)
    
    rtagged = np.sqrt(tagged_particles['x']**2+tagged_particles['y']**2+tagged_particles['z']**2)
    
    #print(tagged_stars['j'].units)                                                                                                                                                                                                                                
    mstar_tagged = [dt.loc[i]['mstar'] for i in tagged_particles['iord']]
    
    dftagged = pd.DataFrame({'r':rtagged,'j':jtagged, 'mass':mstar_tagged })
                                                                                                                                                         
    plt.hist(np.asarray(jtagged),weights= np.asarray(mstar_tagged),histtype='step',label="angmom tagging")                                                                                                                                                        
    plt.hist(np.asarray(jstars),weights= np.asarray(stars['mass']),histtype='step',label="Hydro Sim")                                                                                                                                                             
    plt.yscale('log')
    
    return 





