import numpy as np 
import pandas as pd 
import darklight  
from numpy import sqrt
import random
from .utils import *



def rhalf2D_dm(particles):
    #Calculate radius that encloses half the given particles.

    #Assumes each particle positions have been centered on main halo.  Adopts
    #same 'luminosity' for each particle.  Creates list of projected distances
    #from halo center (as seen from the z-axis), sorts this to get the distances
    #in increasing order, then choses the distance that encloses half the particles.

    rproj = np.sqrt(particles['x']**2 + particles['y']**2)
    rproj.sort()
    if round(len(particles)/2)>0:
        return rproj[ round(len(particles)/2) ]
    else:
        return rproj
                    

def get_mass(m,a,r1,r2):

    # calculates the mass enclosed at distances r1 and r2 
    # from the center of the main halo 
    # according to the plummer profile 

    x1 = m*(r1**3)/((r1**2+a**2)**(3.0/2.0))
    x2 = m*(r2**3)/((r2**2+a**2)**(3.0/2.0))
    return x2-x1



def plum_const(hDMO,z_val,insitu,r200):
    if insitu == 'insitu':
        return ((0.015*r200)/1.3) if z_val > 4 else ((10**(0.1*r200 - 4.2))/1.3)
    else:
        return ((0.015*r200)/1.3)

#prod_binned_df(z_val, msn,mass_select,chosen_parts,DMOparticles,hDMO,'insitu',a_coeff,r200)


def prod_binned_df(z_val, mstar, mass_select, chosen_parts, DMOparticles, hDMO,insitu,a_coeff,r200):

    print('this is how many',len(DMOparticles))
    
    # calculate 'a' to construct plummer profile 
    if insitu == 'insitu':
        a = plum_const(hDMO,z_val,'insitu',r200)
    elif insitu == 'accreted':
        a = plum_const(hDMO,z_val,'accreted',r200)
    

    print('appended to, a_coeff')
    #store 'a' for this snap 
    #a_coeff = np.append(a_coeff,a)
            
    # filter out particles outside the plummer tidal radius 'a'.
    particles_in_selection_radius = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= 15*a ]

    selection_mass = 1112

    iords_of_parts = list(particles_in_selection_radius['iord'])
    positions_of_parts = particles_in_selection_radius['pos']

    if chosen_parts.shape[0] >0:

        print('removal of duplicates took place')
        
        ignore_these = [iords_of_parts.index(i) for i in iords_of_parts if i in chosen_parts]
        
        iords_of_parts = np.delete(iords_of_parts,ignore_these)
        
        positions_of_parts = np.delete(positions_of_parts,ignore_these,axis=0)
    

    # get 3d pos vec magnitude array 
    R_dists = get_dist(positions_of_parts)

    #if insitu == 'insitu':
    #   num_of_bins = int(mstar/(1112*40)) if int(mstar/(1112*40))>5 else 5

    #elif insitu == 'accreted':
    #decide on the number of bins based on mstar 
    #   num_of_bins = int(mstar[-1]/(1112*40)) if int(mstar[-1]/(1112*40))>5 else 5 
    num_of_bins=5      
    #generate log-spaced bins 
    bins = np.logspace(np.log10(min(R_dists)),np.log10(max(R_dists)),num=num_of_bins,base=10)
    bins = np.insert(bins,0,0)
        
    # bin data (assigns bin numbers to data)
    binning = np.digitize(R_dists,bins,right=True)

    # put elements that don't fit anywhere, into the last bin 
    binning = np.where(binning==len(bins),len(bins)-1,binning)
            
    # generate left edges 
    bz_prev = binning-1
    bz_prev = np.where(bz_prev<0,0,bz_prev)


    binned_df = pd.DataFrame({'bin_right':bins[binning],'bin_left':bins[bz_prev], 'iords':iords_of_parts,'pos':R_dists, 'pos_x': positions_of_parts[:,0], 'pos_y': positions_of_parts[:,1],'pos_z': positions_of_parts[:,2]})

    binned_df = binned_df.drop_duplicates(subset=['iords'],keep='first')

    return binned_df, bins, a, a_coeff, selection_mass


def get_bins(bins, binned_df, M_0, a, a_coeff, msp, red_all, t_all, i, insitu,sm):
   

    ch_parts_2 = np.array([])
    out_num = np.array([])
    tgyr_of_choice = np.array([])

    r_of_choice = np.array([])
    p_typ = np.array([])
    a_storage = np.array([])
    m_storage = np.array([])
    for bx in bins[1:]:

        idx_b = np.array(np.where(bins==bx))
        b_left = np.array(idx_b) - 1
                
        #b = binned_df['iords'][binned_df['bin_right'] == bx][binned_df['bin_left'] == bins[b_left[0][0]]].reset_index()
        #po = binned_df[binned_df['bin_right']==bx][binned_df['bin_left']==bins[b_left[0][0]]][['pos_x','pos_y','pos_z']].reset_index()
        df_bin = binned_df[(binned_df['bin_right'] == bx) & (binned_df['bin_left'] == bins[b_left[0][0]])].reset_index()
        po = df_bin[['pos_x','pos_y','pos_z']]
        b = df_bin['iords']
        
        #chaged!!!!!!!
        if insitu == 'insitu':
            print('length of a_coeff after insitu engaged', len(a_coeff))
            if len(a_coeff)>1:
                mass_binned = get_mass(M_0,a,bins[b_left[0][0]],bx)
                
            else:
                print('a_coeff < 0 at this timestep')
                mass_binned = get_mass(M_0,a,bins[b_left[0][0]],bx)

        if insitu == 'accreted':
            print('accreted tagging engaged')
            mass_binned = get_mass(M_0,a,bins[np.array(idx_b) - 1],bx)

        if (mass_binned/1112)<=0:
            print('Decrease/ Zero!! mass_binned = ', mass_binned)
            continue
                
        print('binned mass',mass_binned)
        #print(sm)
        s = int(mass_binned/sm)+(1 if np.random.random() < (mass_binned % sm)/sm else 0)
        #print('This is how many particles-------------------------------------------------------------', int(mass_binned/sm),s,len(b.index))
                
        if s <= len(list(b.index)):
            print('route 1 ;  s < b.index length')
            print('Stellar mass in bin:', s*sm)
            choose_parts = random.sample(list(b.index),s)
            print('choose_out_of_this',b.index)
            print('these were chosen',choose_parts)
                    
        elif len(list(b.index)) < s :
            print('route 2 ; s-1 > b.index length')
            choose_parts = list(b.index)
            print(choose_parts)

        else:
            continue 
                       
        if len(choose_parts) != len(set(choose_parts)):
            print('unequality in choice', len(choose_parts),len(set(choose_parts)))
        else:
            print('All Unique iords are found ')
        
        if len(choose_parts)>0:
            ch_parts_2 = np.append(ch_parts_2,(b.iloc[choose_parts].values))
    
            out_num = np.append(out_num, np.repeat(i,len(choose_parts)))
            r_of_choice = np.append(r_of_choice, np.repeat(red_all[i],len(choose_parts)))
            tgyr_of_choice = np.append(tgyr_of_choice,np.repeat(t_all[i],len(choose_parts)))

            p_typ = np.append(p_typ, np.repeat(insitu,len(choose_parts)))

            a_storage = np.append(a_storage, np.repeat(a,len(choose_parts)))
            m_storage = np.append(m_storage,np.repeat(M_0,len(choose_parts)))
            

    return ch_parts_2,out_num,tgyr_of_choice,r_of_choice,p_typ,a_storage,m_storage
