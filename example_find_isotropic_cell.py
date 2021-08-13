import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from pymatgen.analysis.waterstructureCreator.load_dictionaries import (
    import_bulk_structures)

from pymatgen.analysis.waterstructureCreator.construct_supercells import (
    get_independent_Supercell_Matrices)

from pymatgen.analysis.waterstructureCreator.create_slabs import (
    get_bulk_and_slab)

from pymatgen.analysis.waterstructureCreator.create_combined_slab import (
    get_isotropic_SS, export_lattices)

from pymatgen.analysis.waterstructureCreator.get_isotropic_supercell import (
    get_maximally_isotropic_cell)


# Calcualte all independent Supercell Matrices
SuperCellSizes = [1, 12]

allredm = get_independent_Supercell_Matrices(SuperCellSizes)


# Read bulk structures
bulk_structs = import_bulk_structures(filename="bulk_export.json")


# Create a slab with selected miller indices
oriented_primitive_bulk_o, primitive_slab = get_bulk_and_slab(bulk_structs['Pt'],
                                                              miller=[1,1,1],
                                                              layers=6,
                                                              vacuum=17.)

for i in np.arange(SuperCellSizes[0], SuperCellSizes[1]+1):
    # Try out all supercell matrices and find the most isotropic one
    best, allresults_sorted  = get_isotropic_SS(primitive_slab, allredm, i)
    
    # Plot the supercell lattices in xy
    fig = export_lattices(best, allresults_sorted, '111', i, return_fig=True)
    plt.show()