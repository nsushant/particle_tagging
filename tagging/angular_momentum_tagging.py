import numpy as np 
import pandas as pd 
import darklight  
from numpy import sqrt
import random
import pynbody
from .utils import *

def rank_order_particles_by_angmom(DMOparticles, hDMO):
    
    print('this is how many DMOparticles were passed',len(DMOparticles))
    
    print('r200',hDMO['r200c'])

    particles_in_r200 = DMOparticles[sqrt(DMOparticles['pos'][:,0]**2 + DMOparticles['pos'][:,1]**2 + DMOparticles['pos'][:,2]**2) <= hDMO['r200c']]
    
    softening_length = pynbody.array.SimArray(np.ones(len(particles_in_r200))*10.0, units='pc', sim=None)
    
    angular_momenta = get_dist(particles_in_r200['j'])

    #values arranged in ascending order
    sorted_indicies = np.argsort(angular_momenta.flatten())

    particles_ordered_by_angmom = np.asarray(particles_in_r200['iord'])[sorted_indicies] if sorted_indicies.shape[0] != 0 else np.array([]) 
   
    return np.asarray(particles_ordered_by_angmom)




def assign_stars_to_particles(snapshot_stellar_mass,particles_sorted_by_angmom,most_bound_fraction,selected_particles = [np.array([]),np.array([])]):
    
    '''
    selected_particles is a 2d array with rows = 2, cols = num of particles  
    
    selected_particles[0] = iords
    selected_particles[1] = stellar mass
    
    '''
    
    size_of_most_bound_fraction = int(particles_sorted_by_angmom.shape[0]*most_bound_fraction)
    
    particles_in_most_bound_fraction = particles_sorted_by_angmom[:size_of_most_bound_fraction]
    
    #dividing stellar mass evenly over all the particles in the most bound fraction 

    print('assigning stellar mass')
    
    stellar_mass_assigned = float(snapshot_stellar_mass/len(list(particles_in_most_bound_fraction))) if len(list(particles_in_most_bound_fraction))>0 else 0
    
    #check if particles have been selected before 
    
    idxs_previously_selected = np.where(np.isin(selected_particles[0],particles_in_most_bound_fraction)==True)
    
    selected_particles[1] = np.where(np.isin(selected_particles[0],particles_in_most_bound_fraction)==True,selected_particles[1]+stellar_mass_assigned,selected_particles[1]) 
    
    #if not selected previously, add to array
    
    idxs_not_previously_selected = np.where(np.isin(particles_in_most_bound_fraction,selected_particles[0])==False)

    how_many_not_previously_selected = particles_in_most_bound_fraction[idxs_not_previously_selected].shape[0]
    
    selected_particles_new_iords = np.append(selected_particles[0],particles_in_most_bound_fraction[idxs_not_previously_selected])
    
    selected_particles_new_masses = np.append(selected_particles[1],np.repeat(stellar_mass_assigned,how_many_not_previously_selected))

    
    selected_particles = np.array([selected_particles_new_iords,selected_particles_new_masses])

    array_iords = np.append(selected_particles[0][idxs_previously_selected], particles_in_most_bound_fraction[idxs_not_previously_selected])

    array_masses = np.append(selected_particles[1][idxs_previously_selected],np.repeat(stellar_mass_assigned,how_many_not_previously_selected))

    updates_to_arrays = np.array([array_iords,array_masses])
    
    return selected_particles,updates_to_arrays
    


def tag(DMOparticles, hDMO, snapshot_stellar_mass, free_param_value, previously_tagged_particles = [np.array([]),np.array([])]):
    
    particles_ordered_by_angmom = rank_order_particles_by_angmom(DMOparticles, hDMO)

    return assign_stars_to_particles(snapshot_stellar_mass,particles_sorted_by_angmom, free_param_value, selected_particles = previously_tagged_particles)
    
     
    

    
    
    
    
    
