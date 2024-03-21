import pynbody
import darklight
import matplotlib.pyplot as plt
import sys
import numpy as np 

def calculate_r_vec(pos_array):

    return np.sqrt(pos_array[:,0]**2+pos_array[:,1]**2+pos_array[:,2]**2)

def plot_dm_contour(sim_halo,fname):

    dm_particles = sim_halo.dm

    r200 = pynbody.analysis.halo.virial_radius(sim_halo)

    dm_particles = dm_particles[calculate_r_vec(dm_particles['pos']) < r200]
    
    pynbody.plot.generic.gauss_kde(dm_particles['x'], dm_particles['y'],mass = dm_particles['mass'])

    plt.tight_layout()
    
    plt.savefig(fname)

    return print('wrote: ',fname)

