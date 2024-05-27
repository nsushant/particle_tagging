import numpy as np


def projected_halfmass_radius(particles, tagged_masses):

    particle_distances =  np.sqrt(particles['x']**2 + particles['y']**2 + particles['z']**2)
  
    idxs_distances_sorted = np.argsort(particle_distances)

    sorted_distances = particle_distances[idxs_distances_sorted]
                
    sorted_massess = tagged_masses[idxs_distances_sorted]
                
    cumilative_sum = np.cumsum(sorted_massess)

    R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]


    
    
      
    
