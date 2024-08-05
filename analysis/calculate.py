import numpy as np
import pynbody
import pynbody.filt as f

def projected_halfmass_radius(particles, tagged_masses):
    '''
    inputs: 

    particles - pynbody particle data 

    tagged_masses - stellar masses tagged 

    
    Output: 
    
    R_half - 2D stellar projected half-mass radius 

    '''
    particle_distances =  np.sqrt(particles['x']**2 + particles['y']**2 + particles['z']**2)
  
    idxs_distances_sorted = np.argsort(particle_distances)

    sorted_distances = particle_distances[idxs_distances_sorted]
                
    sorted_massess = tagged_masses[idxs_distances_sorted]
                
    cumilative_sum = np.cumsum(sorted_massess)

    R_half = sorted_distances[np.where(cumilative_sum >= (cumilative_sum[-1]/2))[0][0]]

    return R_half



def calc_luminosity(particle_ages,masses):


    # Assumes ages are supplied in units of gyr 
    
    ages_st = particle_ages*10**(9)

    # Assumes stars have a metallicity of -2 (1/100 L_sun)
    metals = np.ones(len(particle_ages))*-2

    ## calculating v-band mags

    lums = np.load(pynbody.analysis.luminosity._cmd_lum_file)

    age_star = pynbody.array.SimArray(ages_st , units = 'yr')
    
    # get values off grid to minmax
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(lums['ages'])
    metals[np.where(metals < np.min(lums['mets']))] = np.min(lums['mets'])
    metals[np.where(metals > np.max(lums['mets']))] = np.max(lums['mets'])

    age_grid = np.log10(lums['ages'])
    #age_grid = lums['ages']
    met_grid = lums['mets']
    mag_grid = lums['v']

    output_mags = pynbody.analysis.interpolate.interpolate2d(metals,np.log10(age_star), met_grid, age_grid, mag_grid)
    #output_mags = pynbody.analysis.interpolate.interpolate2d(metals,age_star, met_grid, age_grid, mag_grid)

    # calculating luminosities

    vals = output_mags - 2.5 * np.log10(masses)
    
    v_mag_sun = 4.8

    lum_msol = 10.0 ** ((v_mag_sun - vals)*0.4)

    #print('luminosity (Msol): ',list(lum_msol))
    
    return lum_msol


def calc_ages(tagged_particle_df, t_current):

    tform = tagged_particle_df.groupby(['iords']).first()['t'].values

    age = t_current-tform

    return age 




def produce_lums_grouped(df,present_iords,t_snap):

    #tagging_info_for_each_particle = df.groupby(['iords'])

    ages = t_snap - df['t'].values 
    masses_st =df['mstar'].values

    lum_vals = calc_luminosity(ages,masses_st)

    df['lums'] = lum_vals

    lums_df = df.groupby(['iords']).sum()['lums']

    lums_for_part = np.asarray([lums_df.loc[iord] for iord in present_iords])
    '''
    for particle_id_tag in present_iords:

        group = tagging_info_for_each_particle.get_group(particle_id_tag)

        group.sort_values(by="t",ascending=True,inplace=True)
        
        masses_particle = group['mstar'] - group['mstar'].shift(1)

        masses_particle.iloc[0] = group['mstar'].iloc[0] 

        #mass_form = np.append(mass_form,masses_particle)
        tgrp = t_snap - group['t'].values
        #ages_from_tag = np.append(ages_from_tag,tgrp)
        lum_grp = np.sum(np.asarray(calc_luminosity(tgrp,masses_particle)))
        lums_for_part = np.append(lums_for_part,lum_grp)
    '''

    
    return lums_for_part 


def calc_tot_lum(particle_ages,masses):
    

    lums = calc_luminosity(particle_ages,masses)

    halo_luminosity = np.sum(lums)
    
    return halo_luminosity 


def calc_halflight(sim,lum_for_each_iord,band='v',cylindrical=False):

    '''
    Assumes ordering of ages_st is the same as sim_particles
    '''
    
    half_l = np.sum(lum_for_each_iord) * 0.5

    if cylindrical:
        coord = 'rxy'
    else:
        coord = 'r'

    max_high_r = np.max(sim.dm[coord])
    test_r = 0.5 * max_high_r
    
    testrf = f.LowPass(coord, test_r)
    min_low_r = 0.0

    #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
    #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
    
    test_l = np.sum(lum_for_each_iord[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])])
    
    it = 0

    while ((np.abs(test_l - half_l) / half_l) > 0.01):
        it = it + 1
        if (it > 20):
            break

        if (test_l > half_l):
            test_r = 0.5 * (min_low_r + test_r)
        else:
            test_r = (test_r + max_high_r) * 0.5

        testrf = f.LowPass(coord, test_r)
        #chosen_particle_ages = ages_st[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        #chosen_particle_masses = masses[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])]
        test_l = np.sum(lum_for_each_iord[np.isin(sim.dm['iord'],sim.dm[testrf]['iord'])])
            
        if (test_l > half_l):
            max_high_r = test_r
        else:
            min_low_r = test_r
    
    return test_r


                                                                                                                                                        
