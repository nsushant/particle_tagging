

# import statements 

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

# specify preference of halo catalog
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
# Loading-in the tangos data 
path_to_tangos_db = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'

name_of_db = 'Halo1459.db'

## tangos database initialization
tangos.core.init_db(path_to_tangos_db+name_of_db)

## loading in the simulation data  
tangos_simulation = tangos.get_simulation(simname)
tangos_main_halo_object = tangos_simulation.timesteps[-1].halos[0]
halonums = tangos_main_halo_object.calculate_for_progenitors('halo_number()')[0][::-1]

output_names = np.array([tangos_simulation.timesteps[i].__dict__['extension'] for i in range(len(tangos_simulation.timesteps))])
# Alternatively the halonums and output names  can be supplied as arrays 


# get darklight stellar masses for each snapshot
t_darklight,z_darklight,vsmooth,sfh_insitu,mstar_snap_insitu,mstar_total = DarkLight(tangos_main_halo_object,DMO=True,mergers = False)

# alternatively, provide mstar (in units of solar masses) to tag
stellar_mass_to_assign = mstar_snap_insitu[-1]

# Loading-in the pynbody particle data and simulation snapshot 
path_to_pynbody_data = '/vol/ph/astro_data/shared/morkney/EDGE/'
name_of_simulation = 'Halo1459_DMO'

path_to_pynbody_simulation = join(path_to_pynbody_data,name_of_simulation,output_names[-1])
simulation_particles = pynbody.load(path_to_pynbody_simulation).dm
simulation_particles.physical_units()

##centering snapshot on main halo 
halo_catalogue = simulation_particles.halos()
main_halo_pynbody = halo_catalogue[int(halonums[-1])-1]
pynbody.analysis.halo.center(main_halo_pynbody)


# Perform tagging based on angular momentum 
main_halo_virial_radius = pynbody.analysis.halo.virial_radius(main_halo_pynbody.d, overden=200, r_max=None, rho_def='critical')                                                                                             
particles_in_virial_radius  = simulation_particles[sqrt(simulation_particles['pos'][:,0]**2 + simulation_particles['pos'][:,1]**2 + simulation_particles['pos'][:,2]**2) <= main_halo_virial_radius ]

free_parameter_value = 0.01

particles_sorted_by_angmom = rank_order_particles_by_angmom(particles_in_virial_radius, tangos_main_halo_object)
selected_particles,array_to_write = assign_stars_to_particles(stellar_mass_to_assign,particles_sorted_by_angmom, free_parameter_value,np.array([]))


# Analysis and plotting Functions 



  
    
