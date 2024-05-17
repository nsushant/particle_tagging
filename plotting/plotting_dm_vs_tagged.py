#import darklight
import pynbody
import pandas as pd
import numpy as np
import darklight
import matplotlib.pyplot as plt
import sys
import tangos
import os
import matplotlib.style
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Circle
import gc

mpl.rcParams.update({'text.usetex': False})
mpl.style.use('dark_background')



def calc_3D_cm(particles,masses):
    
    x_cm = sum(particles['x']*masses)/sum(masses)

    y_cm = sum(particles['y']*masses)/sum(masses)

    z_cm = sum(particles['z']*masses)/sum(masses)

    return np.asarray([x_cm,y_cm,z_cm])




def calculate_output_number(sim_name,time_requested):
    s_tangos = darklight.edge.load_tangos_data(sim_name)

    num_of_halos_in_output = [ len(s_tangos.timesteps[i].halos[:]) for i in range(len(s_tangos.timesteps)) ]

    valid_ids = np.where(np.asarray(num_of_halos_in_output) > 0)[0]

    t_tangos = s_tangos.timesteps[-1].halos[0].calculate_for_progenitors('t()')[0][::-1]

    time_index = np.where(np.asarray(t_tangos) <= time_requested)[0][-1]
    
    # because the array 'valid_ids' is 0 indexed and the output array isn't
    output_number = int(valid_ids[time_index]+1)

    return output_number 



# user inputs 

'''
if len(sys.argv[:]) == 8:
    p_file_name = str(sys.argv[1])

    p_file_name_hydro = str(sys.argv[2])
    
    time_requested = float(sys.argv[3])

    sim_name = str(sys.argv[4])

    sim_name_hydro = str(sys.argv[4])

    save_to_file = str(sys.argv[5])

    ahf_cen_file = str(sys.argv[6])

    ahf_cen_file_hydro = str(sys.argv[7])

else:
    print('Usage: [particle file] [time to plot] [simname] [filename for image]')
    exit()
'''

#DMOsim = darklight.edge.load_tangos_data(DMOname)
#main_halo = DMOsim.timesteps[-1].halos[0]
#halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
#outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])[-len(halonums):]
                                


#pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

p_file_name = "angular_momentum_tagging_runs/particles/Halo1459_DMO.csv"
#'angular_momentum_tagging_runs/particles/ahf_corrected/Halo1459_DMO_Mreionx02_1p.csv'        

time_requested = 14

sim_name = 'Halo1459_DMO'

sim_name_hydro = 'Halo1459_fiducial'

save_to_file = 'dm_vs_tagged_plots/Halo1459_DMO.pdf'

#ahf_cen_file = 'AHF_cen_files/DMO/Halo1459_DMO_Mreionx02.csv'

#ahf_cen_file_hydro = 'AHF_cen_files/Halo1459_fiducial_Mreionx02.csv'

calculated_reffs = 'angular_momentum_tagging_runs/reffs/Halo1459_DMO.csv'

# finding output_num of time corresponding to user input 


DMOsim = darklight.edge.load_tangos_data(sim_name)
main_halo = DMOsim.timesteps[-1].halos[0]
halonums = main_halo.calculate_for_progenitors('halo_number()')[0][::-1]
outputs = np.array([DMOsim.timesteps[i].__dict__['extension'] for i in range(len(DMOsim.timesteps))])[-len(halonums):]







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
                                    

'''
if sim_name[-3] == 'x':
    DMOname = 'Halo'+halonum+'_DMO_'+'Mreion'+sim_name[-3:]

else:
    DMOname = 'Halo'+halonum+'_DMO' + ('' if len(split)==2 else ('_' +  '_'.join(split[2:])))
'''                     

DMOname = sim_name

# because the array 'valid_ids' is 0 indexed and the output array isn't
#output_number = calculate_output_number(sim_name,time_requested)

#print('ouput:----->',output_number)
#loading in the tagged particles 

d = pd.read_csv(p_file_name)

#d_ahf = pd.read_csv(ahf_cen_file)

dt = d[d['t'] <= time_requested].groupby(['iords']).last()

#print(dt.head(),dt.index.values)

#dt_ahf = d_ahf[d_ahf['i'] == int(output_number-1)]
tagged_iords = dt.index.values
tagged_m = dt['mstar'].values



df_reffs = pd.read_csv(calculated_reffs)

reff_calculated = df_reffs[df_reffs['t'] <= time_requested].iloc[-1]['reff']
prev_reff = df_reffs[df_reffs['t'] <= time_requested].iloc[-2]['reff']
# plotting tagged DMO particles


s = pynbody.load(os.path.join(pynbody_path,DMOname,outputs[-1]))
#darklight.edge.load_pynbody_data(sim_name,output=output_number)
s.physical_units()

print(s.halos())

#h = s.halos()[int(dt_ahf['AHF catalogue id'].values)]

h = s.halos()[1]
pynbody.analysis.halo.center(h)

r200_DMO = pynbody.analysis.halo.virial_radius(h, overden=200, r_max=None, rho_def='critical')

selected_parts = h.dm[np.isin(h.dm['iord'],tagged_iords)]

#selected_parts = selected_parts[np.sqrt(selected_parts['pos'][:,0]**2+selected_parts['pos'][:,1]**2+selected_parts['pos'][:,2]**2)] 

print(tagged_iords,selected_parts['iord'])

idxs_m = [np.where(tagged_iords == i)[0][0] for i in selected_parts['iord']]

selected_masses = [tagged_m[i] for i in idxs_m]


tagged_acc = dt[dt['type']=='accreted'].index.values

acc_m = dt[dt['type']=='accreted']['mstar'].values

selected_parts_acc = h.dm[np.isin(h.dm['iord'],tagged_acc)]
idxs_m_acc = [np.where(tagged_acc == i)[0][0] for i in selected_parts_acc['iord']]

selected_masses_acc = [acc_m[i] for i in idxs_m_acc]

print('max selected masses ',max(selected_masses))

data_all_tagged = pd.DataFrame({'x':selected_parts['x'],'y':selected_parts['y'], 'masses':np.asarray(selected_masses)})




dataframe_for_hist = pd.DataFrame({'r':np.sqrt(selected_parts['x']**2+selected_parts['y']**2+selected_parts['z']**2), 'masses':np.asarray(selected_masses)})


dataframe_for_hist = dataframe_for_hist.sort_values(by=['r'])

dataframe_for_hist['m_enclosed'] = np.cumsum(dataframe_for_hist['masses'].values)




data_only_acc_tagged = pd.DataFrame({'x':selected_parts_acc['x'],'y':selected_parts_acc['y'],'masses':np.asarray(selected_masses_acc)})



tagged_insitu = dt[dt['type']=='insitu'].index.values

insitu_m = dt[dt['type']=='insitu']['mstar'].values

selected_parts_insitu = h.dm[np.isin(h.dm['iord'],tagged_insitu)]
idxs_m_insitu = [np.where(tagged_insitu == i)[0][0] for i in selected_parts_insitu['iord']]

selected_masses_insitu = [insitu_m[i] for i in idxs_m_insitu]


data_only_insitu =pd.DataFrame({'x':selected_parts_insitu['x'],'y':selected_parts_insitu['y'],'masses':np.asarray(selected_masses_insitu)}) 

cen_stars = calc_3D_cm(selected_parts_insitu,selected_masses_insitu)


# plotting hydro particles 
output_number_hydro = calculate_output_number(sim_name_hydro,time_requested)

s_hydro = darklight.edge.load_pynbody_data(sim_name_hydro)

s_hydro.physical_units()

#d_ahf_hydro = pd.read_csv(ahf_cen_file_hydro)

#dt_ahf_hydro = d_ahf_hydro[d_ahf_hydro['i'] == int(output_number_hydro-1)]


#print('halonum hydro:',int(dt_ahf_hydro['AHF catalogue id'].values))


h_hydro = s_hydro.halos()[1]

#reff_hydro = dt_ahf_hydro['reff'].values


pynbody.analysis.halo.center(h_hydro)

stars = h_hydro.st


# new cutoff testing
distances = np.sqrt(data_all_tagged['x']**2+data_all_tagged['y']**2)

sorted_idxs = np.argsort(distances)


distances = distances[sorted_idxs]


masses = np.asarray(data_all_tagged['masses'].values)[sorted_idxs]


b = np.linspace(0,r200_DMO,num=50)


print(stars['mass'],'<--mass array')


bins_digi = np.digitize(distances,bins=b)-1

data_sum = pd.DataFrame({'bins':bins_digi,'masses':masses}).groupby(['bins']).sum()

print('masses max:', max(data_sum['masses'].values))

print('cutoffs:',b[data_sum.index.values[np.where(data_sum['masses'].values < max(data_sum['masses'].values)/100)]])

'''
dtm = np.diff(data_sum['masses'])

loc_minima = np.array([])

id_minima = data_sum.index.values[(loc_minima+1)]
'''

#id_minima = data_sum.index.values[np.where(data_sum['masses'].values < max(data_sum['masses'].values)/100)]

#m_cutoff = min(b[id_minima]) 

data_all_stars = pd.DataFrame({'x':stars['x'],'y':stars['y'], 'masses':np.asarray(stars['mass'])})

#plt.hist(dataframe_for_hist['r'].values,bins=50,weights=dataframe_for_hist['masses'].values)

#print(data_all_stars)

#plotting

#pynbody.plot.generic.gauss_kde(h.dm['x'],h.dm['y'],mass=h.dm['mass'])

#plt.scatter(h.st['x'],h.st['y'],s=0.00001)


#plt.scatter(selected_parts_acc['x'],selected_parts_acc['y'],s=0.001,alpha=(np.asarray(selected_masses_acc)/max(selected_masses)),color='red', label='acc') 

#correct
#plt.scatter(selected_parts['x'],selected_parts['y'],s=0.001,alpha=(np.asarray(selected_masses)/max(selected_masses)),color='white', label='insitu')


#plt.scatter(data_all_stars['x'],data_all_stars['y'],s=0.001,color='red')  



plt.gca().set_box_aspect(1)

sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=True,cmap="viridis")


#,cbar=True)

#sns.kdeplot(data = data_only_acc_tagged, x='x',y='y',weights ='masses',fill=True,cmap="mako")

#sns.kdeplot(data = data_all_tagged, x='x',y='y',weights='masses',fill=False)

sns.kdeplot(data = data_all_stars,x ='x',y='y', weights='masses',fill=False,color='white')


#plt.scatter(data_all_stars['x'],data_all_stars['y'],s=0.001,alpha=data_all_stars['masses'].values/max(data_all_stars['masses'].values),color='red')


circle_reff = Circle(xy=(0,0), radius=reff_calculated, fill=False,color="white")

#circle_hydro_reff = Circle(xy=(0,0), radius=(reff_hydro), fill=False,color="blue")



#cutoff_m = Circle(xy=(0,0),radius=m_cutoff, fill=False,color="blue")


plt.title('Rhalf = '+str(round(reff_calculated,3)))
#plt.gca().add_patch(circle_reff)

#ax.add_patch(circle_hydro_reff)

#ax.add_patch(cutoff_m)

#ax.add_patch(circle_cutoff)
plt.xlim(4,-4)
plt.ylim(4,-4)



'''
plt.plot(dataframe_for_hist['r'].values,dataframe_for_hist['m_enclosed'].values)
#plt.hist(dataframe_for_hist['r'].values,bins=50,weights=dataframe_for_hist['masses'].values)
plt.yscale('log')

plt.xlim(0,1)


plt.yscale('log')

plt.xlim(0,r200_DMO)


plt.title('Halo1459')
plt.ylabel('Mass in $M_{\odot}$')
plt.xlabel('Radial Distance in Kpc')
'''

#plt.colorbar()
#sns.kdeplot(data = data_only_acc_tagged, x='x',y='y',weights='masses',fill=True)


#plt.scatter(cen_stars[0],cen_stars[1],s=0.001,label='Stellar Center')



#plt.scatter(selected_parts_acc['x'],selected_parts_acc['y'],s=0.001,alpha=(np.asarray(selected_masses_acc)/max(selected_masses_acc)),facecolor='red', label='acc')

#plt.colorbar()
#plt.tight_layout()
#plt.title('8Gyrs (DMO)')


plt.savefig(str(save_to_file))

