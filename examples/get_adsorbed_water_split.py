import copy, os, pickle
import numpy as np
import logging
import pandas as pd
import warnings
#warnings (deprecated syntax in pymatgen) are ignored
warnings.filterwarnings('ignore')
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from waterstructureCreator.modified_pymatgen.interface_mod import InterfaceBuilder_mod
from waterstructureCreator.modified_pymatgen.substrate_analyzer_mod import ZSLGenerator_mod, reduce_vectors

from waterstructureCreator.construct_supercells import get_independent_Supercell_Matrices

from waterstructureCreator.check_polarity import get_structure_polarity

from waterstructureCreator.create_slabs import get_bulk_and_slab
from waterstructureCreator.create_slabs import export_structure, create_onesided_analysestruct, get_symmtrafo

from waterstructureCreator.create_combined_slab import get_isotropic_SS, export_lattices, create_all_interfaces

from waterstructureCreator.get_isotropic_supercell import get_maximally_isotropic_cell

from waterstructureCreator.load_dictionaries import *


def create_slab_from_matrix(matrix,primitive_slab):
    """
    create pymatgen structure from the given transformation
    matrix and the primitive slab.

    Parameters
    ----------
    matrix : 2x2 array
        matrix for constructing the surface.
    primitive_slab : pymatgen structure
        primitive slab of the low index surface .

    Returns
    -------
    2D tuple
        3x3 array(transformtion matrix), third array [0,0,1].
        pymatgen structure for the given transformation matrix,
        number of atoms per layer are fixed with the matrix.
    """
    em = np.zeros((3, 3))
    em[:2, :2] = np.array(matrix)
    em[2][2] = 1
    sst = SupercellTransformation(em)
    primitive_slab = sst.apply_transformation(primitive_slab)
    return em,Structure(primitive_slab.lattice, primitive_slab.species,
                     [c.coords for c in primitive_slab.sites], coords_are_cartesian=True,to_unit_cell=True)

def match_surf(primitive_s,water_struct,polarity,water_polymorph,horizontaldist,distz,limitxylen,data_frame):
    """
    core of the program, substrate surfaces are matched to
    the precalculated water films.
    several thresholds are enforced for an appropiate matching.
    if not matching is possible, the program will continue with other structure.
    in case of other exception, the program will give a warning.

    Parameters
    ----------
    primitive_s : 2D-tuple
        3x3 array (transformtion matrix), third array [0,0,1]
        pymatgen structure
    water_struct : 2D-list
        list with information of the water film e.g. lattice
        pymatgen structure of the water film for a given polymoph and lattice
    polarity : bool
        polarity of the polymorph
    water_polymorph : string
        name of the polymorph
    horizontaldist : float
        the distance with which the different lateral
        arrangements are scaned in the unitcell.
    distz : float
        the distance to the surface where the water
        layer is placed.
    limitxylen : float
        limit on any cell length in Angstrom,
        to ignore directly all very assymmetric cells
    data_frame : pandas dataframe
        6 columns dataframe, which are:
        'metal','polymorph', 'water_lattice','number','H2O','struct' .

    Returns
    -------
    type
        update the pandas dataframe

    """
    number = len(data_frame)
    filmstruct = water_struct[-1].copy()
    primitive_slab = primitive_s[1]
    ifb = InterfaceBuilder_mod(primitive_slab, filmstruct)
    #area of substrat
    slabsurface = np.linalg.norm(np.cross(primitive_slab.lattice.matrix[0], primitive_slab.lattice.matrix[1]))

    try:
        ifb.get_oriented_slabs(lowest=True, film_millers=[[0, 0, 1]],substrate_millers=[[0, 0, 1]],\
                               film_layers=1,substrate_layers=1,\
                               zslgen=ZSLGenerator_mod(max_area_ratio_tol=0.10,\
                                                       max_area=slabsurface*3, max_length_tol=0.05,\
                                                       max_angle_tol=0.05),\
                               reorient=False, primitive=False, film_bonds={('H', 'O'): 3, ('O', 'O'): 5},\
                               substrate_bonds={(primitive_slab.species[0],\
                                                 primitive_slab.species[0]): 4})
        #condition
        area_thres = abs((ifb.matches[0]['match_area']-slabsurface)/(slabsurface)) < 0.1
        if area_thres:
            ifb.combine_slabs_nico([distz, 0, 0],polarity)#no idea what it does?
            interfacelist, interfacelist_tags, newstruct_sub, newstruct_film, supercelldict = \
            create_all_interfaces(ifb, horizontaldist=horizontaldist, distz=distz, vacuum=17,polarity=polarity, primitive_SS_trafo=primitive_s[0])

            lattlen = [np.linalg.norm(newstruct_sub.lattice.matrix[0]) < limitxylen,
                       np.linalg.norm(newstruct_sub.lattice.matrix[1]) < limitxylen]
            areadiff = np.linalg.norm(np.cross(newstruct_sub.lattice.matrix[0],newstruct_sub.lattice.matrix[1])) - slabsurface
            if lattlen[0] and lattlen[1] and abs(areadiff) < 0.001:
                count=0
                for iflisttag, ifi in zip(interfacelist_tags, interfacelist):
                    count+=1
                    #create asymmetric slabs
                    infos_halfi, half = create_onesided_analysestruct(ifi)
                    data_frame.loc[number] = [primitive_slab.species[0]] + \
                    [water_polymorph] +[polarity]+[water_struct[0][0]]+[count]+[len(newstruct_film.sites)/3]+[half]
                    number+=1
    except Exception as e:
        if not ifb.matches:
            #logging.exception('not matches found for '+ water_structs+' number '+str(wsindex))
            pass
        else:
            print('Watch out! other error!')

def wateronsurface(films,primitive_s,horizontaldist,distz,limitxylen):
    """
    apply matching function for all 160 water configurations.

    Parameters
    ----------
    films : dict
        contains all precalculated water films for the 8 polymorphs
        in total 160 structures
    primitive_s : 2D-tuple
        3x3 array transformtion matrix, third array [0,0,1]
        pymatgen structure
    horizontaldist : float
        the distance with which the different lateral
        arrangements are scaned in the unitcell.
    distz : float
        the distance to the surface where the water
        layer is placed.
    limitxylen : float
        limit on any cell length in Angstrom,
        to ignore directly all very assymmetric cells

    Returns
    -------
    type
        Description of returned object.

    """
    df = pd.DataFrame(columns=['metal','polymorph','polarity', 'water_lattice','number','H2O','struct'])
    films_polarity = get_structure_polarity(films)

    for water_structs in films.keys():
        water_struct_list = films[water_structs]
        water_struct_list = sorted(water_struct_list, key=lambda x: x[0][0])
        polarity = films_polarity[water_structs]
        for wsindex, water_struct in enumerate(water_struct_list):
            match_surf(primitive_s,water_struct,polarity,water_structs,horizontaldist,distz,limitxylen,df)
    return df
