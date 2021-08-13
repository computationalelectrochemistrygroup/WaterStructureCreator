import copy
import os
import pickle

from ase.visualize import view
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.analysis.waterstructureCreator.load_dictionaries import (
    import_slab_structures, import_bulk_structures)

from pymatgen.analysis.waterstructureCreator.construct_supercells import (
    get_independent_Supercell_Matrices)

from pymatgen.analysis.waterstructureCreator.check_polarity import (
    get_structure_polarity)

from pymatgen.analysis.waterstructureCreator.create_slabs import (
    get_export_subslabs_clean)

from pymatgen.analysis.waterstructureCreator.create_combined_slab import (
    get_export_adsorbed_water_SS)


SuperCellSizes = [6, 12]

allredm = get_independent_Supercell_Matrices(SuperCellSizes)

#Note there is a function in 
# pymatgen.analysis.substrate_analyzer
# def gen_sl_transform_matricies(area_multiple)
# that creates all superlattice transformations
# I didnt use it as I don't really get it.
# It seems there is only superlattices with positive first
# row entries possible, why
# Now I am just not sure if these are really all
# I am not seeing how this function in combination with
# def reduce_vectors(a, b) gives all possible superlattices
# to be checked maybe with random matrices
# Question is reduce_vectors(reduce_vectors(Superlattices)*Basis)
# all there is with the pymatgen functionality?

os.makedirs('pickle_data', exist_ok=True)
with open("pickle_data/all_independent_supercells_{}-{}.pickle".format(SuperCellSizes[0], SuperCellSizes[1]), "wb") as o:
    pickle.dump(allredm,o) 
    
    
with open("pickle_data/all_independent_supercells_{}-{}.pickle".format(SuperCellSizes[0], SuperCellSizes[1]), "rb") as o:
    allredm = pickle.load(o) 
    
films = import_slab_structures(filename="h2o_export.json")

bulk_structs = import_bulk_structures(filename="bulk_export.json")

print("Materials:", bulk_structs.keys())
print("=++++++++++++++++++++++++++++++++=")
print("Water phases:", films.keys())
print("=++++++++++++++++++++++++++++++++=")

print("Each water phase then has {} different possible low energy lattice constants.".format(len(films['p3b4o1'])) )
print("These are")
for l in films['p3b4o1']:
    print("cell lengths: ", np.round(l[0][0],3))
    
    
# This is where we will place the structures
folder = "created_interface_structures_supercells_{}-{}/".format(SuperCellSizes[0], SuperCellSizes[1])
if not os.path.exists(os.getcwd() + "/" + folder):
    os.mkdir(os.getcwd() + "/" + folder)

# Are water 2D phases polar or not. IF polar both orientations will be placed onto the surface later
films_polarity = get_structure_polarity(films)  

# Supercell sizes we are interested in
SuperCellSizes = [6, 12] 
target_ss =  list(np.arange(SuperCellSizes[0], \
                            SuperCellSizes[1] + 1))   


limitxylen = 36. # limit on any cell length in Angstrom, in order to ignore directly all very assymmetric cells
horizontaldist = 2. # the distance with which we scan the different lateral arrangements in the unitcell
waterdistance = 2.3 # The distance to the surface where we want to place the waters


# all results will be placed in this dictionary
all_singlesided_structures = {}
          

# Here we choose which surfaces and how thick we want the cells to be.
# pymatgen has problems with thickness in dhkl units. This works fine
# This will create 100 slabs with 6 atomic layers, 110 with 8 layers and 111 with 6 layers
# later in the process we will cut away the lower half of the slab for prerelaxations and
# this gives us 3,4,3 layers in asymmetric cells sizes 
interface_specs = [ [[1,0,0], "(100)", 3], \
                   [[1,1,0] ,"(110)", 4],  \
                   [[1,1,1], "(111)", 6] ]
#names
allNs= {i[1]: target_ss for i in interface_specs} 
print(allNs)
#raise


for material in ['Pt']: # bulk_structs.keys(), Careful you need always conventional cells! for hkl to be correct
    bulk_struct = bulk_structs[material]
    print(material)
    all_singlesided_structures[material] = {}
    if True:
        all_primitive_slabs = [] 
        # First from conventional bulk structure construct the slabs (primitive slabs)
        # include the oriented primitive bulk structures
        # calculation of bulk references in similar geom makes convergence good (interface energy)
        
        for miller, millername, lays in interface_specs:
            if True:
                all_singlesided_structures[material][millername] = {}
                all_singlesided_structures[material][millername][lays] = {}

                if True:  
                    cleanslab_dict, primitive_slab = \
                    get_export_subslabs_clean(bulk_struct,\
                                          miller, millername,lays, material, folder, vacuum = 17)
                    print("primitive", millername, primitive_slab.lattice.matrix)
                    #view(AseAtomsAdaptor.get_atoms(structure=primitive_slab))
                    #raise SystemExit('sfsbn')
                    
                    all_singlesided_structures[material][millername][lays]["clean"] = cleanslab_dict

                    for target_N in allNs[millername]:
                        print('target_N', target_N)
                   
                        for i in range(2):
                            print('######################')
                        print("Analysing surface {} for {} target surface atoms".format(miller,target_N))
                        for i in range(2):
                            print('######################')
                        returned_SS_infos = get_export_adsorbed_water_SS(target_N, primitive_slab, \
                                                 allredm, films, all_singlesided_structures, \
                                                 material, folder, millername, lays,\
                                                     horizontaldist, waterdistance,\
                                                     limitxylen, \
                                                     films_polarity)
                        #raise Systemerror('Stop')
                        all_singlesided_structures[material][millername][lays][target_N]["all_supercells_info"] = returned_SS_infos

                        
with open("pickle_data/all_singlesided_structures_target_cells_larger_{}-{}.pickle".format(SuperCellSizes[0], SuperCellSizes[1]), "wb") as o:
    pickle.dump(all_singlesided_structures, o)
                
                
            
             
