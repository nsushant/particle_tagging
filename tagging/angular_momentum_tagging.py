
import csv
import os
import pynbody
import tangos
import numpy as np
from numpy import sqrt

import gc

from os import listdir
from os.path import *
import sys

import numpy as np 
import pandas as pd 

from darklight import DarkLight

import tangos 

from tangos.examples.mergers import * 

from numpy import sqrt
import random
import pynbody
from .utils import *

def rank_order_particles_by_angmom(DMOparticles, hDMO):
    
    '''
    Inputs: 

    DMOparticles - Particle data (angular momenta and positions) 
    hDMO - Tangos halo object for the main halo
    
    
    Returns: 
    
    a list of particle IDs ordered by their corresponding angular momenta.
    
    '''
    
    print('this is how many DMOparticles were passed',len(DMOparticles))
    
    print('r200',hDMO['r200c'])

    particles_in_r200 = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= hDMO['r200c']]
    
    softening_length = pynbody.array.SimArray(np.ones(len(particles_in_r200))*10.0, units='pc', sim=None)
    
    angular_momenta = get_dist(particles_in_r200['j'])

    #values arranged in ascending order
    sorted_indicies = np.argsort(angular_momenta.flatten())

    particles_ordered_by_angmom = np.asarray(particles_in_r200['iord'])[sorted_indicies] if sorted_indicies.shape[0] != 0 else np.array([]) 
   
    return np.asarray(particles_ordered_by_angmom)




def assign_stars_to_particles(snapshot_stellar_mass,particles_sorted_by_angmom,tagging_fraction,selected_particles = [np.array([]),np.array([])]):
    
    '''

    Tags the lowest angular momenta dark matter particles of a halo with stellar mass. 

    Inputs: 
    
    snapshot_stellar_mass - stellar mass to be tagged in given snapshot 
    particles_sorted_by_angmom - list of particle dark matter IDs sorted by their angular momenta. 
    tagging_fraction - defines the size of the free paramter used to perform tagging 
    selected_particles - particle IDs of previously selected/tagged particles
    
    
    Returns: 
    
    selected_particles is a 2d array with rows = 2, cols = num of particles  
    
    selected_particles[0] = iords
    selected_particles[1] = stellar mass

    updates_to_arrays = array updates that need to be written to an output file 
    
    '''
    
    size_of_tagging_fraction = int(particles_sorted_by_angmom.shape[0]*tagging_fraction)
    
    particles_in_tagging_fraction = particles_sorted_by_angmom[:size_of_tagging_fraction]
    
    #dividing stellar mass evenly over all the particles in the most bound fraction 

    print('assigning stellar mass')
    
    stellar_mass_assigned = float(snapshot_stellar_mass/len(list(particles_in_tagging_fraction))) if len(list(particles_in_tagging_fraction))>0 else 0
    
    #check if particles have been selected before 
    
    idxs_previously_selected = np.where(np.isin(selected_particles[0],particles_in_tagging_fraction)==True)
    
    selected_particles[1] = np.where(np.isin(selected_particles[0],particles_in_tagging_fraction)==True,selected_particles[1]+stellar_mass_assigned,selected_particles[1]) 
    
    #if not selected previously, add to array
    
    idxs_not_previously_selected = np.where(np.isin(particles_in_tagging_fraction,selected_particles[0])==False)

    how_many_not_previously_selected = particles_in_tagging_fraction[idxs_not_previously_selected].shape[0]
    
    selected_particles_new_iords = np.append(selected_particles[0],particles_in_tagging_fraction[idxs_not_previously_selected])
    
    selected_particles_new_masses = np.append(selected_particles[1],np.repeat(stellar_mass_assigned,how_many_not_previously_selected))

    
    selected_particles = np.array([selected_particles_new_iords,selected_particles_new_masses])

    array_iords = np.append(selected_particles[0][idxs_previously_selected], particles_in_tagging_fraction[idxs_not_previously_selected])

    array_masses = np.append(selected_particles[1][idxs_previously_selected],np.repeat(stellar_mass_assigned,how_many_not_previously_selected))

    updates_to_arrays = np.array([array_iords,array_masses])
    
    return selected_particles,updates_to_arrays
    


def tag(DMOparticles, hDMO, snapshot_stellar_mass,free_param_value = 0.01, previously_tagged_particles = [np.array([]),np.array([])]):

    '''
    
    Given the dark matter particles and the associated tangos halo object, the function performs particle tagging based on angular momentum 

    Inputs:

    DMOparticles - Particle data (angular momenta, positions, IDs) 
    hDMO - Tangos halo object of main halo 
    snapshot_stellar_mass - stellar mass to be tagged in current snapshot 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    previously_tagged_particles - particle IDs of any previously tagged particles 

    Returns: 
    
    selected_particles is a 2d array with rows = 2, cols = num of particles  
    
    selected_particles[0] = iords
    selected_particles[1] = stellar mass

    updates_to_arrays = array updates that need to be written to an output file 
    
    
    '''
    particles_ordered_by_angmom = rank_order_particles_by_angmom(DMOparticles, hDMO)

    return assign_stars_to_particles(snapshot_stellar_mass,particles_sorted_by_angmom, free_param_value, selected_particles = previously_tagged_particles)
    

# under construction
def angmom_tag_over_full_sim(DMOsim, free_param_value = 0.01, pynbody_path  = None, occupation_frac = 'all' ,particle_storage_filename=None, AHF_centers_file=None, mergers = True):

    '''

    Given a tangos simulation, the function performs angular momentum based tagging over the full simulation. 

    Inputs: 

    DMOsim - tangos simulation 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    pynbody_path - path to particle data 
    occupation_frac - One of 'nadler20' , 'all' , 'edge1' or 'edgert' (controls the occupation regime followed by darklight)
    mergers - Whether to include merging/accreting halos or not. 
    
    Returns: 
    
    dataframe with tagged particle masses at given times, redshifts and associated particle IDs  
    
    '''
    
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

    # path to particle data 
    DMOname = DMOsim.path
    # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
    # load_tangos_data is a part of the 'utils.py' file in the tagging dir, it loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
    
    t_all, red_all, main_halo,halonums,outputs = load_indexing_data(DMOsim,1)
    
    # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers = False, poccupied=occupation_frac)

    #calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
    
    if ( len(red_all) != len(outputs) ) : 

        print('output array length does not match redshift and time arrays')
    
    # group_mergers groups all merging objects by redshift.
    # this array gets stored in hmerge_added in the form => len = no. of unique zmerges, 
    # elements = all the hmerges of halos merging at each zmerge
    
    hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

    ##################################################### SECOND LOOP ###############################################################
    
    selected_particles = np.array([[np.nan],[np.nan]])
    mstars_total_darklight_l = [] 
    
    # number of stars left over after selection (per iteration)
    leftover=0

    # total stellar mass selected 
    mstar_selected_total = 0

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids = np.array([])

    # if an AHF centering file is provided use the centers stroed within it
    AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_file != None else None

    tagged_iords_to_write = np.array([])
    tagged_types_to_write = np.array([])
    tagged_mstars_to_write = np.array([])
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    
    # looping over all snapshots  
    for i in range(len(outputs)):
        gc.collect()
        
        # was particle data loaded in (insitu) 
        decision=False

        # was particle data loaded in (accreted) 
        decision2=False
        decl = False
    
        print('Current snapshot -->',outputs[i])
    
        # loading in the main halo object at this snapshot from tangos 
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))

        # value of redshift at the current timestep 
        z_val = red_all[i]
                
        # time in gyr
        t_val = t_all[i]

        # 't' is the darklight time array 
        # idrz is thus the index of the mstar value calculated at the closest time to that of the snap
        idrz = np.argmin(abs(t - t_val))

        # index of previous snap's mstar value in darklight array
        idrz_previous = np.argmin(abs(t - t_all[i-1])) if idrz>0 else None 

        # current snap's darklight calculated stellar mass 
        msn = mstar_s_insitu[idrz]              

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            # else msp = previous snap's mstar value
            elif idrz_previous >= 0:
                msp = mstar_s_insitu[idrz_previous]
        else:
            print('There is no stellar mass at current timestep')
            continue

                                                                    
        #calculate the difference in mass between the two mstar's
        mass_select = int(msn-msp)
        print('stellar mass to be tagged in this snap -->',mass_select)

        # if stellar mass is to be tagged then load in particle data 
    
        if mass_select>0:
            
            decision=True
            
            # try to load in the data from this snapshot
            
            try:
                simfn = join(pynbody_path,outputs[i])
                
                print(simfn)
                print('loading in DMO particles')
                
                DMOparticles = pynbody.load(simfn)
                # once the data from the snapshot has been loaded, .physical_units()
                # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                DMOparticles.physical_units()
                
                #print('total energy  ---------------------------------------------------->',DMOparticles['te'])
                print('loaded data insitu')
            
            # where this data isn't available, notify the user.
            except Exception as e:
                print(e)
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
   
            print('mass_select:',mass_select)
            #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
            
            try:
                hDMO['r200c']
            except:
                print("Couldn't load in the R200 at timestep:" , i)
                continue
            
            print('the time is:',t_all[i])
        
            subhalo_iords = np.array([])
            
            if AHF_centers_file == None:
                print(int(halonums[i])-1)
                h = DMOparticles.halos()[int(halonums[i])-1]

            elif AHF_centers_file != None:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                
                h = DMOparticles.halos()[int(AHF_crossref)] 
                    
                children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                
                child_str_l = children_ahf[0][1:-1].split()

                children_ahf_int = list(map(float, child_str_l))                    
            
                halo_catalogue = DMOparticles.halos()
            
                subhalo_iords = np.array([])
                
                for ch in children_ahf_int:
                    
                    subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(ch)].dm['iord'])
                

                c = 0                  
                
                                                                                                                                        
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
            

            pynbody.analysis.halo.center(h)

            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        
            try:                                                                                                                                                                                              
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
                                                                                                                                                                                                              
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
                                                                                                                                                                            
            
            
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            
            DMOparticles_insitu_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]

            #print('angular_momentum: ', DMOparticles["j"])
            
            DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]

            #DMOparticles_insitu_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],accreted_only_particle_ids))]
            
            particles_sorted_by_angmom = rank_order_particles_by_angmom( DMOparticles_insitu_only, hDMO)
            
            if particles_sorted_by_angmom.shape[0] == 0:
                continue
            
            selected_particles,array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(free_param_value),selected_particles = selected_particles)
            #halonums_indexing+=1
            
            
            print('writing insitu particles to output file')
            
            tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write[0])
            tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('insitu',len(array_to_write[0])))
            tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write[1])
            ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write[0])))
            zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write[0])))

            
            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
            
            #pynbody.analysis.halo.center(h,mode='hyb').revert()

            del DMOparticles_insitu_only
            
            #print('moving onto mergers loop')
            #get mergers ----------------------------------------------------------------------------------------------------------------
            # check whether current the snapshot has a the redshift just before the merger occurs.
        
        if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):
                
            decision2 = False if decision==True else True

            decl=False
            
            t_id = int(np.where(z_set_vals==red_all[i+1])[0][0])

            #print('chosen merger particles ----------------------------------------------',len(chosen_merger_particles))
            #loop over the merging halos and collect particles from each of them
        
            #mstars_total_darklight = np.array([])
            DMO_particles = 0 
            
            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:',hDM)
                
                if (occupation_frac != 'all'):
                    try:
                        prob_occupied = calculate_poccupied(hDM,occupation_frac)

                    except Exception as e:
                        print(e)
                        print("poccupied couldn't be calculated")
                        continue
                    
                    if (np.random.random() > prob_occupied):
                        print('Skipped')
                        continue
                #angmom_tag_over_full_sim(hDM, free_param_value = 0.01, pynbody_path  = pynbody_path, occupation_frac = 'all', mergers = True)

                
                try:
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,poccupied=occupation_frac,mergers=True)
                    print(len(t_2))
                    print(mstar_merging)
                except Exception as e :
                    print(e)
                    print('there are no darklight stars')
                    #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                
                    continue
        
        
                if len(mstar_merging)==0:
                    #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                    continue

                mass_select_merge= mstar_merging[-1]
                #mstars_total_darklight = np.append(mstars_total_darklight,mass_select_merge)

                print(mass_select_merge)
                if int(mass_select_merge)<1:
                    leftover+=mstar_merging[-1]
                    continue
                
                simfn = join(pynbody_path, outputs[i])

                if float(mass_select_merge) >0 and decision2==True:
                    # try to load in the data from this snapshot
                    try:
                        DMOparticles = pynbody.load(simfn)
                        DMOparticles.physical_units()
                        print('loaded data in mergers')
                    # where this data isn't available, notify the user.
                    except:
                        print('--> DMO particle data exists but failed to read it, skipping!')
                        continue
                    decision2 = False
                    decl=True
             
                if int(mass_select_merge) > 0:

                    try:
                        h_merge = DMOparticles.halos()[int(hDM.calculate('halo_number()'))-1]
                        pynbody.analysis.halo.center(h_merge,mode='hyb')
                        r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                    
                    except Exception as ex:
                        print('centering data unavailable, skipping',ex)
                        continue
                                                                                                           
               
                    print('mass_select:',mass_select_merge)
                    #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
                    print('sorting accreted particles by TE')
                    #print(rank_order_particles_by_te(z_val, DMOparticles, hDM,'accreted'), 'output')
                    DMOparticles_acc_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn_acc] 

                    #DMOparticles_acc_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],insitu_only_particle_ids))]
                                            
                    try:
                        accreted_particles_sorted_by_angmom = rank_order_particles_by_angmom(DMOparticles_acc_only, hDM)
                    except:
                        continue
                    
        
                    print('assinging stars to accreted particles')

                    selected_particles,array_to_write_accreted = assign_stars_to_particles(mass_select_merge,accreted_particles_sorted_by_angmom,float(free_param_value),selected_particles = selected_particles)
                    
                    

                    tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write_accreted[0])
                    tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('accreted',len(array_to_write_accreted[0])))
                    tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write_accreted[1])
                    ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write_accreted[0])))
                    zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write_accreted[0])))

        
                    accreted_only_particle_ids = np.append(accreted_only_particle_ids,np.asarray(array_to_write_accreted[0]))
                    print('writing accreted particles to output file')
          
                    del DMOparticles_acc_only
        
                  
                            
        if decision==True or decl==True:
            del DMOparticles
    
    
        print("Done with iteration",i)

        df_tagged_particles = pd.DataFrame({'iords':tagged_iords_to_write, 'mstar':tagged_mstars_to_write,'t':ts_to_write,'z':zs_to_write,'type':tagged_types_to_write})

        if particle_storage_filename != None:
            df_tagged_particles.to_csv(particle_storage_filename)
            
    return df_tagged_particles


def angmom_tag_over_full_sim_recursive(DMOsim, halonumber, free_param_value = 0.01, pynbody_path  = None, occupation_frac = 'all' ,particle_storage_filename=None, AHF_centers_file=None, mergers = True):

    '''

    Given a tangos simulation, the function performs angular momentum based tagging over the full simulation. 

    Inputs: 

    DMOsim - tangos simulation 
    free_param_value - specifies the size of the 'tagging fraction' when tagging dm particles with stellar mass (bigger values correspond to a larger spread of angmom.)
    pynbody_path - path to particle data 
    occupation_frac - One of 'nadler20' , 'all' , 'edge1' or 'edgert' (controls the occupation regime followed by darklight)
    mergers - Whether to include merging/accreting halos or not. 
    
    Returns: 
    
    dataframe with tagged particle masses at given times, redshifts and associated particle IDs  
    
    '''
    
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

    # path to particle data 
    DMOname = DMOsim.path
    # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
    # load_tangos_data is a part of the 'utils.py' file in the tagging dir, it loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
    
    t_all, red_all, main_halo,halonums,outputs = load_indexing_data(DMOsim,halonumber)
    
    # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
    t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers = False, poccupied=occupation_frac)

    #calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
    
    if ( len(red_all) != len(outputs) ) : 

        print('output array length does not match redshift and time arrays')
    
    # group_mergers groups all merging objects by redshift.
    # this array gets stored in hmerge_added in the form => len = no. of unique zmerges, 
    # elements = all the hmerges of halos merging at each zmerge
    
    hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

    ##################################################### SECOND LOOP ###############################################################
    
    selected_particles = np.array([[np.nan],[np.nan]])
    mstars_total_darklight_l = [] 
    
    # number of stars left over after selection (per iteration)
    leftover=0

    # total stellar mass selected 
    mstar_selected_total = 0

    accreted_only_particle_ids = np.array([])
    insitu_only_particle_ids = np.array([])

    # if an AHF centering file is provided use the centers stroed within it
    AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_file != None else None

    tagged_iords_to_write = np.array([])
    tagged_types_to_write = np.array([])
    tagged_mstars_to_write = np.array([])
    ts_to_write = np.array([])
    zs_to_write = np.array([])
    acc_halo_path_tagged = np.array([])
    
    # looping over all snapshots  
    for i in range(len(outputs)):
        gc.collect()
        
        # was particle data loaded in (insitu) 
        decision=False

        # was particle data loaded in (accreted) 
        decision2=False
        decl = False
    
        print('Current snapshot -->',outputs[i])
    
        # loading in the main halo object at this snapshot from tangos 
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))

        # value of redshift at the current timestep 
        z_val = red_all[i]
                
        # time in gyr
        t_val = t_all[i]

        # 't' is the darklight time array 
        # idrz is thus the index of the mstar value calculated at the closest time to that of the snap
        idrz = np.argmin(abs(t - t_val))

        # index of previous snap's mstar value in darklight array
        idrz_previous = np.argmin(abs(t - t_all[i-1])) if idrz>0 else None 

        # current snap's darklight calculated stellar mass 
        msn = mstar_s_insitu[idrz]              

        # msp = previous snap's darklight calculated stellar mass 
        if msn != 0:
            # if there wasn't a previous snap msp = 0 
            
            if idrz_previous==None:
                msp = 0
                
            # else msp = previous snap's mstar value
            elif idrz_previous >= 0:
                msp = mstar_s_insitu[idrz_previous]
        else:
            print('There is no stellar mass at current timestep')
            continue

                                                                    
        #calculate the difference in mass between the two mstar's
        mass_select = int(msn-msp)
        print('stellar mass to be tagged in this snap -->',mass_select)

        # if stellar mass is to be tagged then load in particle data 
    
        if mass_select>0:
            
            decision=True
            
            # try to load in the data from this snapshot
            
            try:
                simfn = join(pynbody_path,outputs[i])
                
                print(simfn)
                print('loading in DMO particles')
                
                DMOparticles = pynbody.load(simfn)
                # once the data from the snapshot has been loaded, .physical_units()
                # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                DMOparticles.physical_units()
                
                #print('total energy  ---------------------------------------------------->',DMOparticles['te'])
                print('loaded data insitu')
            
            # where this data isn't available, notify the user.
            except Exception as e:
                print(e)
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
   
            print('mass_select:',mass_select)
            #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
            
            try:
                hDMO['r200c']
            except:
                print("Couldn't load in the R200 at timestep:" , i)
                continue
            
            print('the time is:',t_all[i])
        
            subhalo_iords = np.array([])
            
            if AHF_centers_file == None:
                print(int(halonums[i])-1)
                h = DMOparticles.halos()[int(halonums[i])-1]

            elif AHF_centers_file != None:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                
                h = DMOparticles.halos()[int(AHF_crossref)] 
                    
                children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                
                child_str_l = children_ahf[0][1:-1].split()

                children_ahf_int = list(map(float, child_str_l))                    
            
                halo_catalogue = DMOparticles.halos()
            
                subhalo_iords = np.array([])
                
                for ch in children_ahf_int:
                    
                    subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(ch)].dm['iord'])
                

                c = 0                  
                
                                                                                                                                        
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
            

            pynbody.analysis.halo.center(h)

            pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        
            try:                                                                                                                                                                                              
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
                                                                                                                                                                                                              
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
                                                                                                                                                                            
            
            
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            
            DMOparticles_insitu_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]

            #print('angular_momentum: ', DMOparticles["j"])
            
            DMOparticles_insitu_only = DMOparticles_insitu_only[np.logical_not(np.isin(DMOparticles_insitu_only['iord'],subhalo_iords))]

            #DMOparticles_insitu_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],accreted_only_particle_ids))]
            
            particles_sorted_by_angmom = rank_order_particles_by_angmom( DMOparticles_insitu_only, hDMO)
            
            if particles_sorted_by_angmom.shape[0] == 0:
                continue
            
            selected_particles,array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(free_param_value),selected_particles = selected_particles)
            #halonums_indexing+=1
            
            
            print('writing insitu particles to output file')
            
            tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write[0])
            tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('insitu',len(array_to_write[0])))
            tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write[1])
            ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write[0])))
            zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write[0])))

            
            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
            
            #pynbody.analysis.halo.center(h,mode='hyb').revert()

            del DMOparticles_insitu_only
            
            #print('moving onto mergers loop')
            #get mergers ----------------------------------------------------------------------------------------------------------------
            # check whether current the snapshot has a the redshift just before the merger occurs.
        
        if (((i+1 < len(red_all)) and (red_all[i+1] in z_set_vals)) and (mergers == True)):
                
            decision2 = False if decision==True else True

            decl=False
            
            t_id = int(np.where(z_set_vals==red_all[i+1])[0][0])

            #print('chosen merger particles ----------------------------------------------',len(chosen_merger_particles))
            #loop over the merging halos and collect particles from each of them
        
            #mstars_total_darklight = np.array([])
            DMO_particles = 0 
            
            for hDM in hmerge_added[t_id][0]:
                gc.collect()
                print('halo:',hDM)
                
                if (occupation_frac != 'all'):
                    try:
                        prob_occupied = calculate_poccupied(hDM,occupation_frac)

                    except Exception as e:
                        print(e)
                        print("poccupied couldn't be calculated")
                        continue
                    
                    if (np.random.random() > prob_occupied):
                        print('Skipped')
                        continue

                
                acc_halo_path = hDM.calculate_for_progenitors('path()')
                halonumber_hDM = hDM.calculate_for_progenitors('halo_number()')[0][0]

                # if halo has not been tagged on before, we want to perform tagging over its full lifetime (upto the current snap)
                if ( len(np.where(np.isin(acc_halo_path,acc_halo_path_tagged) == True)[0]) > 0 ):
                    acc_halo_path_tagged = np.append(acc_halo_path_tagged,acc_halo_path[0][0])
                    
                    df_tagged_acc = angmom_tag_over_full_sim_recursive(hDM, halonumber_hDM, free_param_value = 0.01,pynbody_path = pynbody_path )
                    tagged_iords_to_write = np.append(tagged_iords_to_write,df_tagged_acc['iords'].values)
                    tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('accreted',len(df_tagged_acc['iords'].values)))
                    tagged_mstars_to_write = np.append(tagged_mstars_to_write,df_tagged_acc['mstar'].values)
                    ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(df_tagged_acc['iords'].values)))
                    zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(df_tagged_acc['iords'].values)))
                    
                    selected_particles[1] = np.where(np.isin(selected_particles[0],df_tagged_acc['iords'].values)==True,selected_particles[1]+df_tagged_acc['mstar'].values,selected_particles[1])

                    particles_not_previously_tagged = df_tagged_acc[np.logical_not(np.isin(df_tagged_acc['iords'].values,selected_particles[0]))]

                    selected_particles_new_iords  = np.append(selected_particles[0] , particles_not_previously_tagged['iords'])
                    selected_particles_new_mstar  = np.append(selected_particles[1] , particles_not_previously_tagged['mstar'])

                    selected_particles = np.array([selected_particles_new_iords,selected_particles_new_mstar])
                    accreted_only_particle_ids = np.append(accreted_only_particle_ids,df_tagged_acc['iords'].values)

                    
                    
                else:
                    
                    
                    try:
                        t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,poccupied=occupation_frac,mergers=True)
                        print(len(t_2))
                        print(mstar_merging)
                    except Exception as e :
                        print(e)
                        print('there are no darklight stars')
                        #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                    
                        continue
            
            
                    if len(mstar_merging)==0:
                        #mstars_total_darklight = np.append(mstars_total_darklight,0.0)
                        continue
    
                    mass_select_merge= mstar_merging[-1]
                    #mstars_total_darklight = np.append(mstars_total_darklight,mass_select_merge)
    
                    print(mass_select_merge)
                    if int(mass_select_merge)<1:
                        leftover+=mstar_merging[-1]
                        continue
                    
                    simfn = join(pynbody_path, outputs[i])
    
                    if float(mass_select_merge) >0 and decision2==True:
                        # try to load in the data from this snapshot
                        try:
                            DMOparticles = pynbody.load(simfn)
                            DMOparticles.physical_units()
                            print('loaded data in mergers')
                        # where this data isn't available, notify the user.
                        except:
                            print('--> DMO particle data exists but failed to read it, skipping!')
                            continue
                        decision2 = False
                        decl=True
                 
                    if int(mass_select_merge) > 0:
    
                        try:
                            h_merge = DMOparticles.halos()[int(hDM.calculate('halo_number()'))-1]
                            pynbody.analysis.halo.center(h_merge,mode='hyb')
                            r200c_pyn_acc = pynbody.analysis.halo.virial_radius(h_merge.d, overden=200, r_max=None, rho_def='critical')
                        
                        except Exception as ex:
                            print('centering data unavailable, skipping',ex)
                            continue
                                                                                                               
                   
                        print('mass_select:',mass_select_merge)
                        #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
                        print('sorting accreted particles by TE')
                        #print(rank_order_particles_by_te(z_val, DMOparticles, hDM,'accreted'), 'output')
                        DMOparticles_acc_only = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn_acc] 
    
                        #DMOparticles_acc_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],insitu_only_particle_ids))]
                                                
                        try:
                            accreted_particles_sorted_by_angmom = rank_order_particles_by_angmom(DMOparticles_acc_only, hDM)
                        except:
                            continue
                        
            
                        print('assinging stars to accreted particles')
    
                        selected_particles,array_to_write_accreted = assign_stars_to_particles(mass_select_merge,accreted_particles_sorted_by_angmom,float(free_param_value),selected_particles = selected_particles)
                        
                        
    
                        tagged_iords_to_write = np.append(tagged_iords_to_write,array_to_write_accreted[0])
                        tagged_types_to_write = np.append(tagged_types_to_write,np.repeat('accreted',len(array_to_write_accreted[0])))
                        tagged_mstars_to_write = np.append(tagged_mstars_to_write,array_to_write_accreted[1])
                        ts_to_write = np.append(ts_to_write,np.repeat(t_all[i],len(array_to_write_accreted[0])))
                        zs_to_write = np.append(zs_to_write,np.repeat(red_all[i],len(array_to_write_accreted[0])))
    
            
                        accreted_only_particle_ids = np.append(accreted_only_particle_ids,np.asarray(array_to_write_accreted[0]))
                        print('writing accreted particles to output file')
              
                        del DMOparticles_acc_only
        
                  
                            
        if decision==True or decl==True:
            del DMOparticles
    
    
        print("Done with iteration",i)

        df_tagged_particles = pd.DataFrame({'iords':tagged_iords_to_write, 'mstar':tagged_mstars_to_write,'t':ts_to_write,'z':zs_to_write,'type':tagged_types_to_write})

        if particle_storage_filename != None:
            df_tagged_particles.to_csv(particle_storage_filename)
            
    return df_tagged_particles







def angmom_calculate_reffs_over_full_sim(DMOsim, data_particles_tagged, pynbody_path  = None , AHF_centers_file = None):

    '''

    Given a tangos simulation, the function performs angular momentum based tagging over the full simulation. 

    Inputs: 

    DMOsim - tangos simulation 
    pynbody_path - path to particle data 
    data_particles_tagged - dataframe containing tagged particle data (tagged mstar, particle IDs, tagging times)
    
    Returns: 
    
    dataframe with half-mass radii calculated using tagged particles. 
    
    '''

    
    
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                    
    sims = [str(sim_name)]

    DMOname = DMOsim.path
    
    t_all, red_all, main_halo,halonums,outputs = load_indexing_data(DMOsim,1)
    print(outputs)
    

    #load in the two files containing the particle data
    if ( len(red_all) != len(outputs) ) : 
        print('output array length does not match redshift and time arrays')
 

    data_t = np.asarray(data_particles_tagged['t'].values)
    
    stored_reff = np.array([])
    stored_reff_acc = np.array([])
    stored_reff_z = np.array([])
    stored_time = np.array([])
    kravtsov_r = np.array([])
    stored_reff_tot = np.array([])
    KE_energy = np.array([])
    PE_energy = np.array([])

    AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_supplied == True else None
            
    for i in range(len(outputs)):

        gc.collect()

        
        if len(np.where(data_t <= float(t_all[i]))) == 0:
            continue

        
        dt_all = data_particles_tagged[data_particles_tagged['t']<=t_all[i]]

        data_grouped = dt_all.groupby(['iords']).last()

        selected_iords_tot = data_grouped.index.values

        data_insitu = data_grouped[data_grouped['type'] == 'insitu']
        
        selected_iords_insitu_only = data_insitu.index.values
        
        if selected_iords_tot.shape[0]==0:
            continue
        
        mstars_at_current_time = data_grouped['mstar'].values
        
        half_mass = float(mstars_at_current_time.sum())/2
        
        print(half_mass)
        
        #get the main halo object at the given timestep if its not available then inform the user.

       
        hDMO = tangos.get_halo(DMOname+'/'+outputs[i]+'/halo_'+str(halonums[i]))
            
        print(hDMO)
            
        #for  the given path,entry,snapshot at given index generate a string that includes them
        simfn = join(pynbody_path,outputs[i])
        
        # try to load in the data from this snapshot
        try:  DMOparticles = pynbody.load(simfn)

        # where this data isn't available, notify the user.
        except:
            print('--> DMO particle data exists but failed to read it, skipping!')
            continue
        
        # once the data from the snapshot has been loaded, .physical_units()
        # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
        DMOparticles.physical_units()

        

        try:
            if AHF_centers_file==None:
                h = DMOparticles.halos()[int(halonums[i])-1]
                
            elif AHF_centers_file != None:
                pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                
                
                AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                    
                h = DMOparticles.halos()[int(AHF_crossref)] 
                        
                children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                        
                child_str_l = children_ahf[0][1:-1].split()

                children_ahf_int = list(map(float, child_str_l))

                    
                #pynbody.analysis.halo.center(h)
                    
                #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                
                
                halo_catalogue = DMOparticles.halos()
                
                subhalo_iords = np.array([])
                    
                for i in children_ahf_int:
                            
                    subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(i)].dm['iord'])
                                                                                                                                             
                h = h[np.logical_not(np.isin(h['iord'],subhalo_iords))] if len(subhalo_iords) >0 else h
                

                
            pynbody.analysis.halo.center(h)
            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

        except:
            print('centering data unavailable')
            continue


        try:
            r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')

        except:
            print('could not calculate R200c')
            continue
        DMOparticles = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ]
        
        particle_selection_reff_tot = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else []

        particles_only_insitu = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_insitu_only)] if len(selected_iords_insitu_only) > 0 else []

        
        if (len(particle_selection_reff_tot))==0:
            print('skipped!')
            continue
        else:

            dfnew = data_particles_tagged[data_particles_tagged['t']<=t_all[i]].groupby(['iords']).last()
    
            masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

            masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                
            cen_stars = calc_3D_cm(particles_only_insitu,masses_insitu)
            
            particle_selection_reff_tot['pos'] -= cen_stars
            
            masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

            #particle_selection_reff_tot['pos'] -= cen_stars 

            distances =  np.sqrt(particle_selection_reff_tot['x']**2 + particle_selection_reff_tot['y']**2 + particle_selection_reff_tot['z']**2)

            #caculate the center of mass using all the tagged particles
            #cen_of_mass = center_on_tagged(distances,masses)
            
                        
            idxs_distances_sorted = np.argsort(distances)

            sorted_distances = np.sort(distances)

            distance_ordered_iords = np.asarray(particle_selection_reff_tot['iord'][idxs_distances_sorted])
            
            print('array lengths',len(set(distance_ordered_iords)),len(distance_ordered_iords))

            sorted_massess = [dfnew.loc[n]['mstar'] for n in distance_ordered_iords]
            
            cumilative_sum = np.cumsum(sorted_massess)

            R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]
            #print(cumilative_sum)
            
            halfmass_radius = []

            stored_reff_z = np.append(stored_reff_z,red_all[i])
            stored_time = np.append(stored_time, t_all[i])
               
            stored_reff = np.append(stored_reff,float(R_half))
            kravtsov = hDMO['r200c']*0.02
            kravtsov_r = np.append(kravtsov_r,kravtsov)

            particle_selection_reff_tot['pos'] += cen_stars

            print('halfmass radius:',R_half)
            print('Kravtsov_radius:',kravtsov)
            
        
  

    print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

    df_reff = pd.DataFrame({'reff':stored_reff,'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
    
    
    return df_reff

    
    
    
    
    
