
# parent = pynbody_analysis.py created 2021.08.11 by Stacy Kim

# selection script = created 2021.08.21 by Sushanta Nigudkar 


#import tracemalloc
#from memory_profiler import profile
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
from tangos.examples.mergers import *     
import random
import sys
import pandas as pd
from .functions_for_angular_momentum_tagging import *

def get_child_iords(halo,halo_catalog,DMOstate='fiducial'):

    '''
    
    Given a halo object from an AHF (Amiga's Halo Finder)
    halo catalogue, the function returns a list of dark matter and star particle id's  
    of particles belonging to 'child' or sub-halo of the main halo. 
    
    '''
    children_dm = np.array([])

    children_st = np.array([])

    sub_halonums = np.array([])

    if (np.isin('children',list(halo.properties.keys())) == True) :

        children_halonums = halo.properties['children']

        sub_halonums = np.append(sub_halonums,children_halonums)

        #print(children_halonums)                                                                                                                                                                                                                              


        for child in children_halonums:

            if (len(halo_catalog[child].dm['iord']) > 0):

                children_dm = np.append(children_dm,halo_catalog[child].dm['iord'])



            if DMO_state == 'fiducial':

                if (len(halo_catalog[child].st['iord']) > 0 ):

                    children_st = np.append(children_st,halo_catalog[child].st['iord'])

            if (np.isin('children',list(halo_catalog[child].properties.keys())) == True) :

                dm_2nd_gen,st_2nd_gen,sub_halonums_2nd_gen = get_child_iords(halo_catalog[child],halo_catalog,DMOstate)

                children_dm = np.append(children_dm,dm_2nd_gen)
                children_st = np.append(children_st,st_2nd_gen)
                sub_halonums = np.append(sub_halonums,sub_halonums_2nd_gen)
            #else:                                                                                                                                                                                                                                             
            #    print("there were no star or dark-matter iord arrays")                                                                                                                                                                                        

    #else:                                                                                                                                                                                                                                                     
    #    print("did not find children in halo properties list")                                                                                                                                                                                                

    return children_dm,children_st,sub_halonums



pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

def tag_particles(sim_name,occupation_fraction,fmb_percentage,particle_storage_filename,AHF_centers_file=None,mergers = True,AHF_centers_supplied=False):
    '''
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
    #used paths
    tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
    tangos_path_chimera  = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
    pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
    pynbody_path_chimera = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
    pynbody_edge_gm =  '/vol/ph/astro_data2/shared/morkney/EDGE_GM/'

    
    Halos Available

    'Halo383_fiducial'
    'Halo383_fiducial_late',   'Halo383_fiducial_288', 'Halo383_fiducial_early' 'Halo383_Massive'
    'Halo600_fiducial','Halo600_fiducial_later_mergers','Halo1445_fiducial'
    'Halo605_fiducial','Halo624_fiducial','Halo624_fiducial_higher_finalmass','Halo1459_fiducial',
    'Halo605_fiducial','Halo1459_fiducial_Mreionx02'#, 'Halo1459_fiducial_Mreionx03', 'Halo1459_fiducial_Mreionx12','Halo600_RT', 'Halo605_RT', 'Halo624_RT',
    'Halo1445_RT','Halo1459_RT'

    '''
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
    sims = [str(sim_name)]

    # open the file in the write mode
    with open(particle_storage_filename, 'w') as particle_storage_file:
        # create the csv writer
        writer = csv.writer(particle_storage_file)
        header = ['iords','mstar','t','z','type']
        # write a row to the csv file
        writer.writerow(header)



        
    # iterating over all the simulations in the 'sims' list
    for isim,simname in enumerate(sims):

        print('==================================================')
        print(simname)

        # assign it a short name
        split = simname.split('_')
        DMOstate = split[1]
        shortname = split[0][4:]
        halonum = shortname[:]
        
        if len(split) > 2:
            if   halonum=='332': shortname += 'low'
            elif halonum=='383': shortname += 'late'
            elif halonum=='600': shortname += 'lm'
            elif halonum=='624': shortname += 'hm'
            elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
            elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
            elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
            else:
                print('unsupported simulation',simname,'! Not sure what shortname to give it. Aborting...')
                continue
        elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'

        DMOname = simname 
        
        # set the correct paths to data files
        '''
        if halonum == '383':
            tangos_path  = tangos_path_chimera
            pynbody_path = pynbody_path_chimera 
        else:
            tangos_path  = tangos_path_edge
            pynbody_path = pynbody_path_edge if halonum == shortname else pynbody_edge_gm
        '''

        pynbody_path = darklight.edge.get_pynbody_path(DMOname)
        
        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO simulation with name '+DMOname+' does not exist, skipping!')
            continue
        
        # listdir returns the list of entries in a given dir path (like ls on a dir)
        # isdir check if the given dir exists
        # join creates a string consisting of the path,name,entry in dir
        # once we have this string we check to see if the word 'output' is in this string (to grab only the output snapshots)
        
        snapshots = [ f for f in listdir(pynbody_path) if (isdir(join(pynbody_path,f)) and f[:6]=='output') ]
        
        #sort the list of snapshots in ascending order
        snapshots.sort()
        
        # load in the DMO sim to get particle data and get accurate halonums for the main halo in each snapshot
        # get_the_right_halonums loads in the tangos database 'DMOsim' and returns the main halos tangos object, outputs and halonums at all timesteps
        # here haloidx_at_end specifies the index associated with the main halo at the last snapshot in the tangos db's halo catalogue

        haloidx_at_end = 0
        
        DMOsim,main_halo,halonums,outputs = get_the_right_halonums(DMOname,haloidx_at_end)
        
        print('HALONUMS:---',len(halonums), "OUTPUTS---",len(snapshots))
        
        # Get stellar masses at each redshift using darklight for insitu tagging (mergers = False excludes accreted mass)
        t,redshift,vsmooth,sfh_insitu,mstar_s_insitu,mstar_total = DarkLight(main_halo,DMO=True,mergers = False, poccupied=occupation_fraction)

        #calculate when the mergers took place and grab all the tangos halo objects involved in the merger (zmerge = merger redshift, hmerge = merging halo objects,qmerge = merger ratio)
        zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(main_halo)
        
        # The redshifts and times (Gyr) of all snapshots of the given simulation from the tangos database
        red_all = np.array([ DMOsim.timesteps[i].__dict__['redshift'] for i in range(len(DMOsim.timesteps)) ])
        t_all = np.array([ DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

        # group_mergers groups all merging objects by redshift.
        # this array gets stored in hmerge_added in the form => len = no. of unique zmerges, 
        # elements = all the hmerges of halos merging at each zmerge
        hmerge_added, z_set_vals = group_mergers(zmerge,hmerge)

        
        print('dkl',np.array(mstar_s_insitu))
        print(len(snapshots),'snaps length')

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
        AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_supplied == True else None
        

        with open(particle_storage_filename, 'a') as particle_storage_file:
            
            # looping over all snapshots  
            for i in range(len(snapshots)):

                # finding out what index in the tangos output array matches the specific snapshot 
                
                idxout = np.asarray(np.where(outputs==snapshots[i])).flatten()
                                
                if idxout.shape[0] == 0 :
                    print('no matching output found')
                    continue
                else:
                    iout = idxout[0]
                    
                # was particle data loaded in (insitu) 
                decision=False

                # was particle data loaded in (accreted) 
                decision2=False
                decl = False
            
                print('Current snapshot -->',i)

                #get the halo objects at the given timestep if and inform the user if no halos are present.
                if len(DMOsim.timesteps[i].halos[:])==0:
                    print('No halos found in the tangos db at this timestep')
                    continue
            
                # loading in the main halo object at this snapshot from tangos 
                hDMO = tangos.get_halo(DMOname+'/'+snapshots[i]+'/halo_'+str(halonums[iout]))
        
                # value of redshift at the current timestep 
                z_val = red_all[iout]
                        
                # time in gyr
                t_val = t_all[iout]

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
                        simfn = join(pynbody_path,snapshots[i])
                        
                        print(simfn)
                        print('loading in DMO particles')
                        
                        DMOparticles = pynbody.load(simfn)
                        # once the data from the snapshot has been loaded, .physical_units()
                        # converts all array’s units to be consistent with the distance, velocity, mass basis units specified.
                        DMOparticles.physical_units()
                        
                        #print('total energy  ---------------------------------------------------->',DMOparticles['te'])
                        print('loaded data insitu')
                    
                    # where this data isn't available, notify the user.
                    except:
                        print('--> DMO particle data exists but failed to read it, skipping!')
                        continue
           
                    print('mass_select:',mass_select)
                    #print('total energy  ---------------------------------------------------->',DMOparticles.loadable_keys())
                    
                    iout = np.where(outputs==snapshots[i])[0][0]
                    
                    try:
                        hDMO['r200c']
                    except:
                        print("Couldn't load in the R200 at timestep:" , i)
                        continue
                    
                    print('the time is:',t_all[iout])
                
                    subhalo_iords = np.array([])
                    
                    if AHF_centers_supplied==False:
                        h = DMOparticles.halos()[int(halonums[iout])-1]

                    elif AHF_centers_supplied == True:
                        pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]
                        
                        AHF_crossref = AHF_centers[AHF_centers['i'] == i]['AHF catalogue id'].values[0]
                        
                        h = DMOparticles.halos()[int(AHF_crossref)] 
                            
                        children_ahf = AHF_centers[AHF_centers['i'] == i]['children'].values[0]
                        
                        child_str_l = children_ahf[0][1:-1].split()

                        children_ahf_int = list(map(float, child_str_l))                    
                    
                        halo_catalogue = DMOparticles.halos()
                    
                        subhalo_iords = np.array([])
                        
                        for i in children_ahf_int:
                            
                            subhalo_iords = np.append(subhalo_iords,halo_catalogue[int(i)].dm['iord'])
                        

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
                    
                    DMOparticles = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= r200c_pyn ] #hDMO['r200c']]

                    #print('angular_momentum: ', DMOparticles["j"])
                    
                    
                    DMOparticles_insitu_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],subhalo_iords))]

                    #DMOparticles_insitu_only = DMOparticles[np.logical_not(np.isin(DMOparticles['iord'],accreted_only_particle_ids))]
                    
                    
                    particles_sorted_by_angmom = rank_order_particles_by_angmom(z_val, DMOparticles_insitu_only, hDMO, centering=False)
                    
                    if particles_sorted_by_angmom.shape[0] == 0:
                        continue
                    
                    selected_particles,array_to_write = assign_stars_to_particles(mass_select,particles_sorted_by_angmom,float(fmb_percentage),selected_particles)
                    #halonums_indexing+=1
                    writer = csv.writer(particle_storage_file)
                    print('writing insitu particles to output file')


                    insitu_only_particle_ids = np.append(insitu_only_particle_ids,np.asarray(array_to_write[0]))
                    
                    for particle_ids,stellar_masses in zip(array_to_write[0],array_to_write[1]):
                        writer.writerow([particle_ids,stellar_masses,t_all[iout],red_all[iout],'insitu'])
                    print('insitu selection done')
                    
                    #pynbody.analysis.halo.center(h,mode='hyb').revert()
            
                    #print('moving onto mergers loop')
                    #get mergers ----------------------------------------------------------------------------------------------------------------
                    # check whether current the snapshot has a the redshift just before the merger occurs.
                
                idxout_next = np.asarray(np.where(outputs==snapshots[i+1])).flatten()
                if idxout_next.shape[0] == 0 :
                    print('no matching output found')
                    continue
                else:
                    iout_next = np.where(outputs==snapshots[i+1])[0][0]
                
                if (((iout_next<len(red_all)) and (red_all[iout_next] in z_set_vals)) and (mergers == True)):
                        
                    decision2 = False if decision==True else True

                    decl=False
                    
                    t_id = int(np.where(z_set_vals==red_all[iout_next])[0][0])

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
                        
                        simfn = join(pynbody_path,snapshots[i])

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
                            except:
                                print('centering data unavailable, skipping')
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
                                writer.writerow([particle_ids,stellar_masses,t_all[iout],red_all[iout],'accreted'])

                            #pynbody.analysis.halo.center(h_merge,mode='hyb').revert()
                  
                            
                                    
                if decision==True or decl==True:
                    del DMOparticles
            
            
                print("Done with iteration",i)
                
    return pd.read_csv(particle_storage_filename)


def calc_3D_cm(particles,masses):
    
    x_cm = sum(particles['x']*masses)/sum(masses)
        
    y_cm = sum(particles['y']*masses)/sum(masses)
    
    z_cm = sum(particles['z']*masses)/sum(masses)

    return np.asarray([x_cm,y_cm,z_cm])


def center_on_tagged(radial_dists,mass):
    masses = np.asarray(mass)
        
    return sum(radial_dists*masses)/sum(masses)




def calculate_reffs(sim_name, particles_tagged,reffs_fname,AHF_centers_file=None,from_file = False,from_dataframe=False,save_to_file=True,AHF_centers_supplied=False):
    #used paths
    tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
    tangos_path_chimera  = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
    pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
    pynbody_path_chimera = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
    pynbody_edge_gm =  '/vol/ph/astro_data2/shared/morkney/EDGE_GM/'

    '''

    'Halo383_fiducial'
    'Halo383_fiducial_late', 'Halo383_fiducial_288', 'Halo383_fiducial_early','Halo383_Massive',
    'Halo600_fiducial','Halo600_fiducial_later_mergers','Halo605_fiducial','Halo624_fiducial',
    'Halo624_fiducial_higher_finalmass','Halo1445_fiducial','Halo1445_fiducial','Halo1459_fiducial_Mreionx02', 'Halo1459_fiducial_Mreionx03','Halo1459_fiducial_Mreionx12','Halo600_RT', 'Halo605_RT', 'Halo624_RT',
    'Halo1445_RT','Halo1459_RT'

    '''
    pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
                    

    sims = [str(sim_name)]

    for isim,simname in enumerate(sims):

        print('==================================================')
        print(simname)
        
        # assign it a short name
        split = simname.split('_')
        shortname = split[0][4:]
        halonum = shortname[:]
        if len(split) > 2:
            if   halonum=='332': shortname += 'low'
            elif halonum=='383': shortname += 'late'
            elif halonum=='600': shortname += 'lm'
            elif halonum=='624': shortname += 'hm'
            elif halonum=='1459' and split[-1][-2:] == '02': shortname += 'mr02'
            elif halonum=='1459' and split[-1][-2:] == '03': shortname += 'mr03'
            elif halonum=='1459' and split[-1][-2:] == '12': shortname += 'mr12'
            else:
                print('unsupported simulation',simname,'! Not sure what shortname to give it. Aborting...')
                continue
        elif len(split)==2 and simname[-3:] == '_RT':  shortname += 'RT'
       # DMOname = 'Halo'+halonum+'_DMO' if split[-1]=='fiducial' else None
        #DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))

        if simname[-3] == 'x':
            DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+simname[-3:]

        else:
            DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:]))) #if split[1]=='fiducial' else None

                        
        # set the correct paths to data files
        if halonum == '383':
            tangos_path  = tangos_path_chimera
            pynbody_path = pynbody_path_chimera #if halonum == shortname else pynbody_edge_gm
        else:
            tangos_path  = tangos_path_edge
            pynbody_path = pynbody_path_edge if halonum == shortname else pynbody_edge_gm
        
        # get particle data at z=0 for DMO sims, if available
        if DMOname==None:
            print('--> DMO particle does not data exists, skipping!')
            continue
        # listdir returns the list of entries in a given dir path (like ls on a dir)
        # isdir check if the given dir exists
        # join creates a string consisting of the path,name,entry in dir
        # once we have this string we check to see if the word 'output' is in this string (to grab only the output snapshots)
        DMOsim = darklight.edge.load_tangos_data(DMOname)
        main_halo = DMOsim.timesteps[-1].halos[0]
        halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
        outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])[-len(halonums):]

        print(outputs)
        
        snapshots = [ f for f in listdir(pynbody_path+DMOname) if (isdir(join(pynbody_path,DMOname,f)) and f[:6]=='output') ]
        
        #sort the list of snapshots in ascending order

        snapshots.sort()

        
        red_all = np.array([DMOsim.timesteps[i].__dict__['redshift'] for i in range(len(DMOsim.timesteps)) ])
        t_all = np.array([DMOsim.timesteps[i].__dict__['time_gyr'] for i in range(len(DMOsim.timesteps)) ])

        #load in the two files containing the particle data

        data_particles = pd.read_csv(particles_tagged)

        #print('data parts',data_particles['t'])

        data_t = np.asarray(data_particles['t'].values)
        
        stored_reff = np.array([])
        stored_reff_acc = np.array([])
        stored_reff_z = np.array([])
        stored_time = np.array([])
        kravtsov_r = np.array([])
        stored_reff_tot = np.array([])
        KE_energy = np.array([])
        PE_energy = np.array([])

        AHF_centers = pd.read_csv(str(AHF_centers_file)) if AHF_centers_supplied == True else None
                
        for i in range(len(snapshots)):

            gc.collect()

            #print(data_t[i])

            #if i >= int(stop_run):
             #   print('skipped')
              #  continue
            
            if len(np.where(data_t <= float(t_all[i]))) == 0:
                continue

            
            dt_all = data_particles[data_particles['t']<=t_all[i]]


            data_grouped = dt_all.groupby(['iords']).last()
            
            #selected_iords_tot = np.unique(data_particles['iords'][data_particles['t']<=t_all[i]].values)

            selected_iords_tot = data_grouped.index.values

            data_insitu = data_grouped[data_grouped['type'] == 'insitu']
            
            selected_iords_insitu_only = data_insitu.index.values
            

            #selected_iords_insitu = np.unique(data_particles['iords'][data_particles['type']=='insitu'][data_particles['t']<=t_all[i]].values)
            
            #selected_iords_acc = np.unique(data_particles['iords'][data_particles['type']=='accreted'][data_particles['t']<=t_all[i]].values)


            if selected_iords_tot.shape[0]==0:
                continue
            
            #mstars_at_current_time = data_particles[data_particles['t'] <= t_all[i]].groupby(['iords']).last()['mstar']

            mstars_at_current_time = data_grouped['mstar'].values
            
            half_mass = float(mstars_at_current_time.sum())/2
            
            print(half_mass)
            #selected_iords_acc = np.array(data_particles['iords'][data_particles['z']>=red_all[i]][ data_particles['type']=='accreted'])
            #get the main halo object at the given timestep if its not available then inform the user.

            if len(DMOsim.timesteps[i].halos[:])==0:
                print('No halos!')
                continue

            elif len(np.where(outputs==snapshots[i])[0])>0 :
                
                print(np.where(outputs==snapshots[i])[0])
                
                iout = np.where(outputs==snapshots[i])[0][0]
                hDMO = tangos.get_halo(DMOname+'/'+snapshots[i]+'/halo_'+str(halonums[iout]))
                
                print(hDMO)
                
                #hDMO =DMOsim.timesteps[i].halos[0]

            else:
                print('Snap not found in outputs --------------------------------------- ')
                continue

            #for  the given path,entry,snapshot at given index generate a string that includes them
            simfn = join(pynbody_path,DMOname,snapshots[i])
            
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
                if AHF_centers_supplied==False:
                    h = DMOparticles.halos()[int(halonums[iout])-1]
                    
                elif AHF_centers_supplied == True:
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

            #pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
            
            '''                                                                    
            try:
                #DMOparticles['pos']-= hDMO['shrink_center']
                h = DMOparticles.halos()[int(halonums[iout])-1]
                pynbody.analysis.halo.center(h,mode='hyb')
                                        
            except:
                print('Tangos shrink center unavailable!')
                continue
            '''

            
            particle_selection_reff_tot = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_tot)] if len(selected_iords_tot)>0 else []

            particles_only_insitu = DMOparticles[np.isin(DMOparticles['iord'],selected_iords_insitu_only)] if len(selected_iords_insitu_only) > 0 else []

            print('m200 value---->',hDMO['M200c'])
            
            if (len(particle_selection_reff_tot))==0:
                print('skipped!')
                continue
            else:

                dfnew = data_particles[data_particles['t']<=t_all[i]].groupby(['iords']).last()

        
                masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

                masses_insitu = [data_insitu.loc[iord]['mstar'] for iord in particles_only_insitu['iord']]
                    
                cen_stars = calc_3D_cm(particles_only_insitu,masses_insitu)
                
                particle_selection_reff_tot['pos'] -= cen_stars
                
                #particle_selection_reff_tot['pos'] -= cen_stars

                # new cutoff calc begins 
                distances = np.sqrt(particle_selection_reff_tot['x']**2+particle_selection_reff_tot['y']**2 + particle_selection_reff_tot['z']**2)

                b = np.linspace(0,r200c_pyn,num=50)

                bins_digi = np.digitize(distances,bins=b)-1

                data_sum = pd.DataFrame({'bins':bins_digi,'masses':masses}).groupby(['bins']).sum()

                print('masses min',min(data_sum['masses']))
                print('masses max:', max(data_sum['masses'].values))

                #print('cutoffs:',b[data_sum.index.values[np.where(data_sum['masses'].values < max(data_sum['masses'].values)/100)]])

                if min(data_sum['masses'].values) > (max(data_sum['masses'].values)/100):
                    id_minima = data_sum.index.values[np.where(data_sum['masses'].values <= min(data_sum['masses']))]
                else:    
                    id_minima = data_sum.index.values[np.where(data_sum['masses'].values <= max(data_sum['masses'].values)/100)]

                m_cutoff = min(b[id_minima])
                # new cutoff calc ends 
                if (len(stored_reff)>0):
                    previous_halflight = stored_reff[-1]
                    particle_selection_reff_tot = particle_selection_reff_tot[np.sqrt(particle_selection_reff_tot['pos'][:,0]**2+particle_selection_reff_tot['pos'][:,1]**2+particle_selection_reff_tot['pos'][:,2]**2) <= (m_cutoff)]
                    
                masses = [dfnew.loc[n]['mstar'] for n in particle_selection_reff_tot['iord']]

                #cen_stars = calc_3D_cm(particle_selection_reff_tot,masses)

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

                '''
                for d in range(len(sorted_distances)):
                    if cumilative_sum[d] >= half_mass:
                        halfmass_radius.append(sorted_distances[d])
                '''     

                stored_reff_z = np.append(stored_reff_z,red_all[i])
                stored_time = np.append(stored_time, t_all[i])
                   
                stored_reff = np.append(stored_reff,float(R_half))
                kravtsov = hDMO['r200c']*0.02
                kravtsov_r = np.append(kravtsov_r,kravtsov)

                particle_selection_reff_tot['pos'] += cen_stars

                print('halfmass radius:',R_half)
                print('Kravtsov_radius:',kravtsov)
                
            
        #open('reffs_new23_'+halonum+'.csv','w').close()

        print('---------------------------------------------------------------writing output file --------------------------------------------------------------------')

        df_reff = pd.DataFrame({'reff':stored_reff,'z':stored_reff_z, 't':stored_time,'kravtsov':kravtsov_r})
        
        #df2_reff = pd.DataFrame({'z_tangos':ztngs, 't_tangos':ttngs,'reff_tangos':hlftngs})
        
        df_reff.to_csv(reffs_fname) if save_to_file==True else print('reffs not saved to file, to store values set save_to_file = True')
        #df2_reff.to_csv('reffs_new22_tangos'+halonum+'.csv')
        print('wrote', reffs_fname)
        
    return df_reff
