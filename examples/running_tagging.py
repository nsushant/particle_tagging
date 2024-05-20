
'''
from particle_tagging_package.angular_momentum_tagging_module import *
import sys


# Uncomment code block to run tagging using EDGE specific functions 


if len(sys.argv) == 5:
    haloname = str(sys.argv[1])
    occupation_fraction = str(sys.argv[2])
    name_of_p_file = str(sys.argv[3])
    script_mode = str(sys.argv[4])

    
if len(sys.argv) == 5:
    haloname = str(sys.argv[1])
    name_of_p_file = str(sys.argv[2])
    name_of_reff_file = str(sys.argv[3])
    script_mode = str(sys.argv[4])

fmb_percentage = 0.01    


if script_mode == 'tagging': 
    tag_particles(haloname,occupation_fraction,fmb_percentage,name_of_p_file,AHF_centers_file=None,mergers = True,AHF_centers_supplied=False)

if script_mode == 'reff calculation':
    calculate_reffs(haloname, name_of_p_file,name_of_reff_file,AHF_centers_file=None,from_file =True,from_dataframe=False,save_to_file=True,AHF_centers_supplied=False)

'''


# for more general use 

import csv
import os
import pynbody
import tangos
import numpy as np
from numpy import sqrt
from darklight import DarkLight
import darklight
from os import listdir
from os.path import *
import gc
import random
import sys
import pandas as pd
from .functions_for_angular_momentum_tagging import *

pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
#paths to tangos databases
tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
tangos_path_chimera  = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'

#paths to pynbody particle data 
pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
pynbody_path_chimera = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
pynbody_edge_gm =  '/vol/ph/astro_data2/shared/morkney/EDGE_GM/'

# script inputs
simname = 'Halo1459_DMO'
occupation_fraction = 'all'
particle_storage_filename = 'particles_tagged_with_angular_momentum_'+simname+'.csv'


DMOname = simname 
split = simname.split('_')
shortname = split[0][4:]
halonum = shortname[:]

# set the correct paths to data files
tangos_path  = tangos_path_edge
pynbody_path = pynbody_path_edge 

# fetch tangos data 
tangos.core.init_db(tangos_path+'Halo'+halonum+'.db')
DMOsim = tangos.get_simulation(simname)

main_halo = DMOsim.timesteps[-1].halos[0]
halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
times_valid = main_halo.calculate_for_progenitors('t()')[0][::-1]
        
outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])
times_tangos = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

valid_outputs = outputs[np.isin(times_tangos,times_valid)]

#get darklight stellar masses for each snapshot
t_darklight,z_darklight,vsmooth,sfh_insitu,mstar_snap_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers = False, poccupied=occupation_fraction)

# get list of merging objects stored in tangos merger trees  
# zmerge = merger redshifts 
# qmerge = merger ratios (main halo DM mass / merging halo DM mass, small qmerge = large halo w.r.t. main halo) 
# hmerge = halo objects of merging halos 

zmerge, qmerge, hmerge = tangos.examples.mergers.get_mergers_of_major_progenitor(main_halo)

# group_mergers groups all merging objects by redshift.
# this array gets stored in hmerge_added in the form => len = no. of unique zmerges, 
# elements = all the hmerges of halos merging at each zmerge
hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

# The redshifts and times (Gyr) of all snapshots of the given simulation from the tangos database
red_all = main_halo.calculate_for_progenitors('z()')[0][::-1]
      
t_all = main_halo.calculate_for_progenitors('t()')[0][::-1]
     
if ( len(red_all) != len(outputs) ) : 
    print('output array length does not match redshift and time arrays')


##################################################### SECOND LOOP ###############################################################
        
selected_particles = np.array([[np.nan],[np.nan]])
mstars_total_darklight_l = [] 
        
# number of stars left over after selection (per iteration)
leftover=0

# total stellar mass selected 
mstar_selected_total = 0

accreted_only_particle_ids = np.array([])
insitu_only_particle_ids = np.array([])


with open(particle_storage_filename, 'a') as particle_storage_file:
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
            
            # load in pynbody data from this snapshot
            
            try:
                simfn = join(pynbody_path,DMOname,outputs[i])
                
                print(simfn)
                print('loading in DMO particles')
                
                DMOparticles = pynbody.load(simfn)
                # once the data from the snapshot has been loaded, .physical_units()
                # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                DMOparticles.physical_units()
                
               
                print('loaded data insitu')
            
            # where this data isn't available, notify the user.
            except Exception as e:
                print(e)
                print('--> DMO particle data exists but failed to read it, skipping!')
                continue
   
            print('mass_select:',mass_select)
            #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
            
            print('the time is:',t_all[i])

            # get pynbody particle data for main halo 
            h = DMOparticles.halos()[int(halonums[i])-1]
            
            # center on main halo 
            pynbody.analysis.halo.center(h)
        
            try:                                                                                                                                                                                              
                r200c_pyn = pynbody.analysis.halo.virial_radius(h.d, overden=200, r_max=None, rho_def='critical')                                                                                             
                                                                                                                                                                                                              
            except:                                                                                                                                                                                           
                print('could not calculate R200c')                                                                                                                                                            
                continue                                                                                                                                                                                      
                                                                                                                                                                            
            DMOparticles_only_main = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ]

            DMOparticles_only_main = DMOparticles_only_main[np.logical_not(np.isin(DMOparticles_only_main['iord'],subhalo_iords))]
            
            particles_sorted_by_angmom = rank_order_particles_by_angmom(z_val, DMOparticles_only_main, hDMO, centering=False)
            
            if particles_sorted_by_angmom.shape[0] == 0:
                continue
            
            selected_particles,array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(fmb_percentage),selected_particles)
           
            writer = csv.writer(particle_storage_file)
            print('writing insitu particles to output file')


            insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
            
            for particle_ids,stellar_masses in zip(array_to_write[0],array_to_write[1]):
                writer.writerow([particle_ids,stellar_masses,t_all[i],red_all[i],'insitu'])
            print('insitu selection done')
            
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
            
                if (occupation_fraction != 'all'):
                    try:
                        prob_occupied = calculate_poccupied(hDM,occupation_fraction)

                    except Exception as e:
                        print(e)
                        print("poccupied couldn't be calculated")
                        continue
                    
                    if (np.random.random() > prob_occupied):
                        print('Skipped')
                        continue
                
                try:
                    t_2,redshift_2,vsmooth_2,sfh_in2,mstar_in2,mstar_merging = DarkLight(hDM,DMO=True,poccupied=occupation_fraction,mergers=True)
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
                
                simfn = join(pynbody_path,DMOname,outputs[i])

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
                        accreted_particles_sorted_by_angmom = rank_order_particles_by_angmom(red_all[i], DMOparticles_acc_only, hDM, centering=False)
                    except:
                        continue
        
                    print('assinging stars to accreted particles')

                    selected_particles,array_to_write_accreted = assign_stars_to_particles(mass_select_merge,accreted_particles_sorted_by_angmom,float(fmb_percentage),selected_particles)
                    
                    writer = csv.writer(particle_storage_file)
        
                    accreted_only_particle_ids = np.append(accreted_only_particle_ids,np.asarray(array_to_write_accreted[0]))
                    print('writing accreted particles to output file')
                    #pynbody.analysis.halo.center(h_merge,mode='hyb').revert()
                     
                    for particle_ids,stellar_masses in zip(array_to_write_accreted[0],array_to_write_accreted[1]):
                        writer.writerow([particle_ids,stellar_masses,t_all[i],red_all[i],'accreted'])

                    #pynbody.analysis.halo.center(h_merge,mode='hyb').revert()
          
                    del DMOparticles_acc_only
        
                    
                            
        if decision==True or decl==True:
            del DMOparticles
    



  
    
