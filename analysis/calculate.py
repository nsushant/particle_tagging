import numpy as np
import pynbody

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



def calc_luminosity(particle_ages):


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

    v_mag_sun = 4.8

    lum_msol = 10.0 ** ((v_mag_sun - output_mags)*0.4)

    #print('luminosity (Msol): ',list(lum_msol))
    
    return lum_msol


def calc_ages(tagged_particle_df, t_current):

    tform = tagged_particle_df.groupby(['iords']).first()['t'].values

    age = t_current-tform

    return age 
