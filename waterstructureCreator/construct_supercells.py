import numpy as np
from itertools import product

from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import reduce_vectors

########### Construction of supercell matrices

def get_independent_Supercell_Matrices(max_supercell_sizes, limit = None):
    """Find all independent supercell matrices for a given range of cell sizes.

    Parameters
    ----------
    max_supercell_sizes : list object
        Min. and max. size of the created supercell matrizes.
    limit : int, optional
        Min. and max. integer value of a matrix element

    Returns
    -------
    ind_supercell_matrices : dict
        Dictionary containing the list of independent reduced matrices
        ("independent"), list of lists of matrices equal to the independent
        ones ("equal_reduced_matrices") and list of the ratio between the two length of the two
        dimensions of the matrix and the angle between them ("asym_angle") for
        each supercell sizeDescription of returned object.

    """
    
    target_supercell_areas, all_reduced_matrices = construct_all_possible_supercell_Matrices(
        max_supercell_sizes, limit=limit)
    ind_supercell_matrices = {}

    for n in sorted(all_reduced_matrices.keys()):
        if n in target_supercell_areas:
            val = all_reduced_matrices[n]
            ind_reduced_matrices, equal_reduced_matrices = get_unequal_Matrices(all_reduced_matrices[n])
            ind_lens_angles = []
            for m in ind_reduced_matrices:
                ind_lens_angles.append(
                    [np.linalg.norm(m[1])/(np.linalg.norm(m[0])),
                    round(np.arccos(np.dot(m[0], m[1])/(np.linalg.norm(m[0])*np.linalg.norm(m[1])))/np.pi*180),])
            ind_supercell_matrices[n] = {"equals": equal_reduced_matrices,
            "independent": ind_reduced_matrices, "asym_angle" : ind_lens_angles}
    return ind_supercell_matrices


def construct_all_possible_supercell_Matrices(max_supercell_sizes, limit = None):
    """Find all independent supercell matrices for a given range of cell sizes.

    Parameters
    ----------
    max_supercell_sizes : list object
        Min. and max. size of the created supercell matrizes
    limit : int, optional
        Min. and max. integer value of a matrix element

    Returns
    -------
    target_supercell_areas  : list object
        list of the generated cell sizes
    all_reduced_matrices : dict
        Dictionary of [0] all generated supercell matrices and [1] the reduced
        matrix.
    """
    target_supercell_areas = list(np.arange(max_supercell_sizes[0], max_supercell_sizes[1]+1))

    #This is a bit heuristic but we want to restrict the number of possibly assumed multiples
    if limit == None:
        limits = int(2*np.sqrt(max_supercell_sizes[1]))
    else:
        limits = int(limit)
    x = np.arange(-limits, limits+1, 1)

    all_reduced_matrices = {}
    count = 0
    for val in product(x,repeat=4):
        count +=1
        if count%10000 == 0:
            print(count, " supercell matrices created")
        mat = np.array(val).reshape(2,2)



        # we don't want negative determinant. It should stay right hand system
        if np.linalg.det(mat) > 0.3 and np.linalg.det(mat) < max_supercell_sizes[1]+1:

            if float(np.linalg.det(mat)) < 0:
                mat = [-mat[0], mat[1]]

            dd = int(round(float(np.linalg.det(mat))))
            if dd not in all_reduced_matrices:
                all_reduced_matrices[dd] = []
            # Note the reduced matrix might be not any more right handed
            rmat = np.array(reduce_vectors(mat[0], mat[1]))
            if np.linalg.det(rmat) < 0:
                rmat = np.dot(  rmat , np.array([[0,1],[1,0]]))
            #we want rmat[0,0] always positive as a rule
            if np.isclose(rmat[0][0], 0):
                if rmat[0][1] < -0.1:
                    rmat = np.dot(  rmat , np.array([[-1,0],[0,-1]]))
            if rmat[0][0] < -0.1:
                rmat = np.dot(  rmat , np.array([[-1,0],[0,-1]]))


            all_reduced_matrices[dd].append((mat, rmat))
    return target_supercell_areas, all_reduced_matrices


def get_unequal_Matrices(all_reduced_matrices):
    """Create a list of independent supercell matrices and list of all equal ones

    Parameters
    ----------
    all_reduced_matrices : dict
        Dictionary of [0] all generated supercell matrices and [1] the reduced matrix

    Returns
    -------
    ind_reduced_matrices  : list object
        list of independet reduced matrices
    equal_reduced_matrices : list object
        list of equal matrices
    """
    # these are the reduced matrices, should be right handed
    red = [rr[1] for rr in all_reduced_matrices]
    ind_reduced_matrices = []
    equal_reduced_matrices = []
    for e, r in enumerate(red):
        if ind_reduced_matrices == []:
            ind_reduced_matrices.append(r)
        else:
            found = False
            for e2, r2 in enumerate(ind_reduced_matrices):
                # are they the same?
                if not np.isclose(r,r2).all():
                    rot = np.dot( r, np.linalg.inv(r2) )
                    # are they just related via a simple change of axis = rotation
                    if np.linalg.norm(rot.astype(int) - rot) < 0.01:
                        equal_reduced_matrices.append((r, r2))
                        found = True
                        break

                else:
                    found = True
                    equal_reduced_matrices.append((r, r2))
                    break
            if not found:
                ind_reduced_matrices.append(r)
    return ind_reduced_matrices, equal_reduced_matrices