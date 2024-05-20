
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
from tangos.examples.mergers import *
import random
import sys
import pandas as pd
from .functions_for_angular_momentum_tagging import *


  
    
