import .spatial_tagging as spt
import .angular_momentum_tagging as angt


def tag_particles(DMOsim, path_to_particle_data = '/vol/ph/astro_data/shared/morkney/EDGE/', tagging_method = 'angular momentum', free_param_val = 0.01, include_mergers = True, darklight_occupation_frac = 'all' ):
    
    if tagging_method == 'angular momentum':
      
      df_tagged = angt.angmom_tag_over_full_sim(DMOsim, fmb_percentage = free_param_val, pynbody_path  = path_to_particle_data, occupation_frac = darklight_occupation_frac, mergers = include_mergers)
      

    if tagging_method == 'spatial' : 

      df_tagged = spt.spatial_tag_over_full_sim(DMOsim, pynbody_path  = path_to_particle_data, occupation_frac = darklight_occupation_frac, particle_storage_filename=None, mergers= include_mergers)


    return df_tagged

