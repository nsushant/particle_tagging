#import darklight
import pynbody
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import tangos
import os
import matplotlib.style
import matplotlib as mpl

mpl.style.use('dark_background')


# user inputs 

if len(sys.argv[:]) == 6:
    p_file_name = str(sys.argv[1])

    time_requested = float(sys.argv[2])

    sim_name = str(sys.argv[3])

    save_to_file = str(sys.argv[4])

    ahf_cen_file = str(sys.argv[5])
else:
    print('Usage: [particle file] [time to plot] [simname] [filename for image]')
    exit()
    
# finding output_num of time corresponding to user input 


tangos_path_edge     = '/vol/ph/astro_data/shared/morkney/EDGE/tangos/'
tangos_path_chimera  = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'
pynbody_path_edge    = '/vol/ph/astro_data/shared/morkney/EDGE/'
pynbody_path_chimera = '/vol/ph/astro_data/shared/etaylor/CHIMERA/'                
pynbody_edge_gm =  '/vol/ph/astro_data2/shared/morkney/EDGE_GM/'


split = sim_name.split('_')
DMOstate = split[1]
shortname = split[0][4:]
halonum = shortname[:]


                    
if halonum == '383':
    tangos_path  = tangos_path_chimera
    pynbody_path = pynbody_path_chimera
else:
    tangos_path  = tangos_path_edge
    pynbody_path = pynbody_path_edge if halonum == shortname else pynbody_edge_gm
                                    


if sim_name[-3] == 'x':
    DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+sim_name[-3:]

else:
    DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))
                        


tangos.core.init_db(tangos_path+'Halo'+halonum+'.db')
s_tangos = tangos.get_simulation(sim_name)

num_of_halos_in_output = [ len(s_tangos.timesteps[i].halos[:]) for i in range(len(s_tangos.timesteps)) ]

valid_ids = np.where(np.asarray(num_of_halos_in_output) > 0)[0]

t_tangos = s_tangos.timesteps[-1].halos[0].calculate_for_progenitors('t()')[::-1]

time_index = np.where(np.asarray(t_tangos) >= time_requested)[0][0]

# because the array 'valid_ids' is 0 indexed and the output array isn't
output_number = int(valid_ids[time_index]+1)
#loading in the tagged particles 
d = pd.read_csv(p_file_name)
d_ahf = pd.read_csv(ahf_cen_file)

dt = d[d['t'] <= time_requested]

dt_ahf = d_ahf[d_ahf['i'] == int(output_number-1)]
tagged_iords = dt['iords'].values
tagged_m = dt['mstar'].values

if sim_name[-3] == 'x':
    simpath = '/vol/ph/astro_data2/shared/morkney/EDGE_GM/{0}/'.format(sim_name)
else:
    simpath = '/vol/ph/astro_data2/shared/morkney/EDGE/{0}/'.format(sim_name)
    
simfn = os.path.join(simpath,'output_'+str(output_number).zfill(5))

s = pynbody.load(simfn)
s.physical_units()
print(s.halos())

h = s.halos()[int(dt_ahf['AHF catalogue id'].values)]

pynbody.analysis.halo.center(h)


selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]


print(tagged_iords,selected_parts['iord'])

idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

selected_masses = [tagged_m[i] for i in idxs_m]


tagged_acc = dt[dt['type']=='accreted']['iords'].values
acc_m = dt[dt['type']=='accreted']['mstar'].values
selected_parts_acc = h.dm[np.isin(h.dm['iord'],tagged_acc)]
idxs_m_acc = [np.where(tagged_acc == i)[0][0] for i in selected_parts_acc['iord']]

selected_masses_acc = [tagged_acc[i] for i in idxs_m_acc]

#plotting

#pynbody.plot.generic.gauss_kde(h.dm['x'],h.dm['y'],mass=h.dm['mass'])

#plt.scatter(h.st['x'],h.st['y'],s=0.00001)

plt.scatter(selected_parts_acc['x'],selected_parts_acc['y'],s=0.001,alpha=(np.asarray(selected_masses_acc)/max(selected_masses_acc)),facecolor='red', label='acc') 
plt.scatter(selected_parts['x'],selected_parts['y'],s=0.001,alpha=(np.asarray(selected_masses)/max(selected_masses)),color='yellow', label='insitu')
#plt.scatter(selected_parts_acc['x'],selected_parts_acc['y'],s=0.001,alpha=(np.asarray(selected_masses_acc)/max(selected_masses_acc)),facecolor='red', label='acc')
plt.legend(frameon=False)

#plt.colorbar()
#plt.tight_layout()
#plt.title('8Gyrs (DMO)')

plt.savefig(str(save_to_file))

