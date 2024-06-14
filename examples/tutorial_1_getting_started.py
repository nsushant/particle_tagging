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

import particle_tagging_package as ptag 


# specify preference of halo catalog
pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]
    
# Loading-in the tangos data 
## tangos database initialization
tangos.core.init_db('/vol/ph/astro_data/shared/morkney/EDGE/tangos/Halo1459.db')

## loading in the simulation data  
tangos_simulation = tangos.get_simulation('Halo1459_DMO')

#df_tagged_particles = ptag.tag_particles(tangos_simulation, tagging_method = 'angular momentum')
df_tagged_particles = ptag.tag_particles(tangos_simulation, tagging_method = 'spatial')

print(df_tagged_particles.head())
# change df_tagged_particles = ptag.tag_over_full_sim(tangos_simulation, method = 'angular momentum')

# ptag.analysis.plotting, ptag.analysis.calculate 




  
    
