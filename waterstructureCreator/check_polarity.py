import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Getting the polarity

def get_structure_polarity(selected_h2o):
    """Check the water structures for polarity

    Parameters
    ----------
    selected_h2o : dict
        dictionary of the structures

    Returns
    -------
    structure_polarity  : dict
        dictionary of structures and their polarity as boolean
    """

    structure_polarity = {}
    for water_structs in selected_h2o.keys():
        water_struct_list = selected_h2o[water_structs]

        water_struct_list = sorted(water_struct_list, key = lambda x : x[0][0])
        # sorting does what it should do!
        polt = []
        for wsindex, water_struct in enumerate(water_struct_list):
            filmstruct = water_struct[-1].copy()
            sg = SpacegroupAnalyzer(filmstruct, symprec=0.1)
            mytest = any([np.linalg.norm(a.rotation_matrix[2] - np.array([0,0,-1])) < 0.0001 for a in sg.get_space_group_operations()])
            polt.append(float(mytest))
        polt = not bool(int(np.round(np.mean(polt),0)))
        structure_polarity[water_structs] = polt
    return structure_polarity
