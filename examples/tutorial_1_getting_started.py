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
#from particle_tagging_package.tagging.angular_momentum_tagging import *

import particle_tagging_package.tagging as ptag 


# specify preference of halo catalog
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
# Loading-in the tangos data 
## tangos database initialization
tangos.core.init_db('/vol/ph/astro_data/shared/morkney/EDGE/tangos/Halo1459.db')

## loading in the simulation data  
tangos_simulation = tangos.get_simulation('Halo1459_DMO')

'''
tangos_main_halo_object = tangos_simulation.timesteps[-1].halos[0]

## Alternatively supply halonums, output names as arrays
halonums = tangos_main_halo_object.calculate_for_progenitors('halo_number()')[0][::-1]
output_names = np.array([tangos_simulation.timesteps[i].__dict__['extension'] for i in range(len(tangos_simulation.timesteps))])
 
# get stellar mass to tag using darklight 
## alternatively, provide mstar (in units of solar masses) 
t_darklight,z_darklight,vsmooth,sfh_insitu,mstar_snap_insitu,mstar_total = DarkLight(tangos_main_halo_object,DMO=True,mergers = False)
stellar_mass_to_assign = mstar_snap_insitu[-1]


# Loading-in the pynbody particle data for the last simulation snapshot 
simulation_particles = pynbody.load(join('/vol/ph/astro_data/shared/morkney/EDGE/Halo1459_DMO',output_names[-1]))
simulation_particles.physical_units()

##centering snapshot on main halo 
main_halo_pynbody = simulation_particles.halos()[int(halonums[-1])-1]
pynbody.analysis.halo.center(main_halo_pynbody)

main_halo_virial_radius = pynbody.analysis.halo.virial_radius(main_halo_pynbody.d, overden=200, r_max=None, rho_def='critical')                                                                                             
particles_in_virial_radius  = simulation_particles[sqrt(simulation_particles['pos'][:,0]**2 + simulation_particles['pos'][:,1]**2 + simulation_particles['pos'][:,2]**2) <= main_halo_virial_radius ]

# Perform tagging based on angular momentum for a single snapshot  

## here the script array = [ all selected particle ids, masses tagged ] , array_to_write_to_output_file =  [ newly selected particle ids, masses tagged ]
## script_array = array_to_write_to_output_file for single snapshot tagging but different when running over >1 snapshot 
script_array,array_to_write_to_output_file = ptag.angular_momentum_tagging.tag(particles_in_virial_radius, tangos_main_halo_object, stellar_mass_to_assign, free_param_value = 0.01)

#particles_sorted_by_angmom = rank_order_particles_by_angmom(particles_in_virial_radius, tangos_main_halo_object)
#selected_particles,array_to_write = assign_stars_to_particles(stellar_mass_to_assign,particles_sorted_by_angmom, free_parameter_value,[np.array([]),np.array([])])
'''

df_tagged_particles = ptag.angular_momentum_tagging.tag_over_full_sim(tangos_simulation)







  
    
