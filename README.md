This is a particle tagging package that assigns stellar mass to particles in dark matter only simulations. The primary aim of the project is to accurately reproduce the sizes and stellar mass distributions of dwarf galaxies. 
Two tagging methods have been developed 

1.  Angular momentum based tagging - associates stellar mass with dark matter particles using their angular momenta. (particle dark matter mass != stellar mass tagged)
2.  Spatial tagging - distributes stellar mass across the galaxy based on a Plummer Profile. (particle dark matter mass = stellar mass tagged)

The package has the following dependencies in its current form. 

1. Darklight
2. Pynbody
3. Tangos
4. Numpy
5. Pandas
6. OS
7. csv
8. random
9. Seaborn (only for plotting scripts)
10. Matplotlib (only for plotting scripts)


An example script is available in the 'examples' folder and demonstrates how the modules are used to tag particles in a simulation from the EDGE suite. This is a version of the angular momentum tagging script that has been 
reduced to the barebones. In addition to the above packages, the tagging methods require tangos databases (including merger trees) , pynbody particle data and currently use either AHF or HOP halo catalogues. 
