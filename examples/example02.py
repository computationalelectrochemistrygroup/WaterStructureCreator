#!/usr/bin/env python
# coding: utf-8

from ase.visualize import view
import numpy as np

from pymatgen.io.ase import AseAtomsAdaptor

from waterstructureCreator.load_dictionaries import import_slab_structures, import_bulk_structures
from waterstructureCreator.construct_supercells import get_independent_Supercell_Matrices
from waterstructureCreator.check_polarity import get_structure_polarity
from waterstructureCreator.create_slabs import get_export_subslabs_clean, get_bulk_and_slab
from waterstructureCreator.create_combined_slab import get_export_adsorbed_water_SS, get_isotropic_SS
from get_adsorbed_water_split import *


# Example: Static water structures

# Creation of supercell scaling matrices
# While pymatgen internally allows to scan through various surface terminations and supercells, it makes more sense in the present case to fix a priori a certain supercell geometry for any (hkl) surface. Here the requirement is not that we get perfect fit with water structures but we rather want to use for a given (hkl) surface the most isotropic one, as this allows a maximum distance between periodic images and is thus the most sensible approach to choose supercell geometries, that are then only at a later stage evaluated w.r.t. to fitting 2D water supercells.

# Here we calculate the transformation 
# matrices for 12 atoms per layer.
Natoms_perlayer = 12
SuperCellSizes = [Natoms_perlayer, Natoms_perlayer]
allredm = get_independent_Supercell_Matrices(SuperCellSizes)


for i in allredm.keys():
    print('For',i, 'atoms per layer, we found ',len(allredm[i]['independent']),'independent matrices')


# # Water films
# Here we import the water structures, which were previously calculated by means of DFT.
# For each polymorph, 20 structures have been computed. 


films = import_slab_structures(filename="metalbulk_waterfilms/h2o_export.json")
print("Water phases:", films.keys())
print("=++++++++++++++++++++++++++++++++=")

print("Each water phase then has {} different possible low energy lattice constants.".format(len(films['p3b4o1'])) )
print("These are")
for l in films['p3b4o1']:
    print("cell lengths: ", np.round(l[0][0],3))


# # Create interface Structures

# Read bulk structures, which were calculated 
# previously by means of DFT.
bulk_structs = import_bulk_structures(filename="metalbulk_waterfilms/bulk_export.json")
print("Materials:", bulk_structs.keys())
print("=++++++++++++++++++++++++++++++++=")

# Miller index and number of layers for surfaces.
interface_specs = [[[1,0,0], "(100)", 3],                    [[1,1,0] ,"(110)", 4],                     [[1,1,1], "(111)", 6]]

# Create a slab with selected miller indices.
# In this example, we calculate the primitive 
# slab for the Pt(111) surface.  
oriented_primitive_bulk_o, primitive_slab = get_bulk_and_slab(bulk_structs['Pt'],
                                                              miller=interface_specs[2][0],
                                                              layers=interface_specs[2][2],
                                                              vacuum=17.)

# Visualize
view(AseAtomsAdaptor.get_atoms(primitive_slab))


# Find the isotropic supercell.
best, allresults_sorted  = get_isotropic_SS(primitive_slab, allredm, Natoms_perlayer)
trans_matrix = allresults_sorted[0][-1]
# Create the isotropic supercell 
# from the transformation matrix.
isotropic_cell = create_slab_from_matrix(trans_matrix,primitive_slab)
# Visualize
view(AseAtomsAdaptor.get_atoms(isotropic_cell[1]))


# Settings for placing the water adlayer
limitxylen = 36. # limit on any cell length in Angstrom, in order to ignore directly all very assymmetric cells
horizontaldist = 2. # the distance with which we scan the different lateral arrangements in the unitcell
waterdistance = 2.3 # The distance to the surface where we want to place the waters

# Results are shown in a pandas DataFrame.
# Pymatgen structures can be exported.
df = wateronsurface(films,isotropic_cell,horizontaldist,waterdistance,limitxylen)

df.query('H2O == 8.0')

view(AseAtomsAdaptor.get_atoms(df.iloc[4].struct))



