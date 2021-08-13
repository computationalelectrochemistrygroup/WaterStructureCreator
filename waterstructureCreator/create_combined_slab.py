import os
import numpy as np
import pylab as plt

from pymatgen.analysis.interface import align_x, get_ortho_axes
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
SupercellTransformation)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import (
    ZSLGenerator_mod, reduce_vectors)
from pymatgen.analysis.waterstructureCreator.interface_mod import (
    InterfaceBuilder_mod)
from pymatgen.analysis.waterstructureCreator.get_isotropic_supercell import (
    get_maximally_isotropic_cell)
from pymatgen.analysis.waterstructureCreator.plot_lattices import (
    plot_matrices)
from pymatgen.analysis.waterstructureCreator.create_slabs import (
    export_structure, create_onesided_analysestruct, get_symmtrafo)


########### Adsorb water


def get_export_adsorbed_water_SS(target_N, primitive_slab_small, \
                                 allpossibleSS, films, all_singlesided_structures, \
                                 material, folder, millername, lays, horizontaldist, distz,  limitxylen, films_polarity):
    """Place water films on the substrate and export the generated structures.

    Parameters
    ----------
    target_N : int
        Number of substrate atoms per layer
    primitive_slab_small : pymatgen structure
        Primitive slab structure of the sustrate
    allpossibleSS (allredm) : dict
        Dictionary containing the list of independent reduced matrices
        ("independent"), list of lists of matrices equal to the independent
        ones ("equals") and list of the ratio between the two length of the two
        dimensions of the matrix and the angle between them ("asym_angle") for
        each supercell sizeDescription of returned object.
    films (filedict) : dict
        Dictionary of the water film structures
    all_singlesided_structures : list
        list of miller indices
    material : str
        Name of the substrate e.g. Pt
    folder : str
        File path to the folder to export the generated structures
    millername : str
        Miller indices as str
    lays : int
        Number of layers
    horizontaldist : float
        The distance with which we scan the different lateral arrangements in the unitcell
    distz : float
        The distance to the surface where we want to place the waters
    limitxylen : float
        Limit on any cell length in Angstrom, in order to ignore directly all very assymmetric cells
    films_polarity  : dict
        dictionary of structures and their polarity

    Returns
    -------
    fname  : str
        path of the exported structure
    """

    best, allresults_sorted  = get_isotropic_SS(primitive_slab_small, allpossibleSS, target_N)
    # (stdevPCA, ired_KPOINTSnr, kpmesh, lattmat, SSmatrix])
    export_lattices(best, allresults_sorted, millername, target_N)
    return_lattice_info = [best, allresults_sorted, millername, target_N]
    SSmatrix = best[-1]


    all_singlesided_structures[material][millername][lays][target_N] = {}
    em = np.zeros((3,3))
    em[:2,:2] = np.array(SSmatrix)
    em[2][2] = 1
    sst = SupercellTransformation(em)
    primitive_slab = sst.apply_transformation(primitive_slab_small)

    # Somehow atoms will be outside cell, dont ask me

    primitive_slab = Structure(primitive_slab.lattice, primitive_slab.species, \
                               [ c.coords for c in primitive_slab.sites], coords_are_cartesian = True,
                              to_unit_cell = True)

    # Note the lattice orientation will be still as in the primitive_slab_small
    # That is nice as we then have always knowledge about the in plane lattice directions

    all_singlesided_structures[material][millername][lays][target_N]["Opt_Supercell_Properties"] = {"stdev_PCA": best[0], \
         "ired_kpoints": best[1], "suggested_kpoints_mesh": best[2],\
         "lattice_matrix": best[3], "supercell_trafo_prim": best[4]}
    slabsurface = np.linalg.norm(np.cross(primitive_slab.lattice.matrix[0] , primitive_slab.lattice.matrix[1]  ))
    for water_structs in films.keys():
        all_singlesided_structures[material][millername][lays][target_N][water_structs] = {}
        all_singlesided_structures[material][millername][lays][target_N][water_structs]["is_polar"] =  films_polarity[water_structs]
        water_struct_list = films[water_structs]

        water_struct_list = sorted(water_struct_list, key = lambda x : x[0][0] )

        for wsindex, water_struct in enumerate(water_struct_list):
            film_description = water_struct[0]

            filmstruct = water_struct[-1].copy()
            

            ifb = InterfaceBuilder_mod(primitive_slab, filmstruct)
            try:
                # Note: Standard Interface builder of PMG will try to find primitive. We don't want that.
                # We want only to find if it fits into the cell
                # Note we need to allow for max_area larger than *1 otherwise it seems to miss stuff.
                # We choose max_area_ratio_tol=0.10, max_length_tol=0.05 because we have constructed the
                # Water database according to these restrictions meaning, we only want ideally one of the
                # lattice constants in the water database to match (at least not neighbouring ones).
                # Indeed there is a hazard of missing something when exactly on the limit, but we have
                # enough interfaces anyways.
                # We edited standard builder so that it includes film and substrate bonds as we don't want
                # that the water film is broken up, and we want here only to match onto our predefined
                # slab so that not an infinte amount of additional cuts are made, e.g. for an alloy.
                # We rather control this externally by own slab creation.

                ifb.get_oriented_slabs(lowest=True, film_millers=[[0,0,1]], \
                                        substrate_millers=[[0,0,1]], film_layers=1,\
                                        substrate_layers=1,\
                                        zslgen = ZSLGenerator_mod(max_area_ratio_tol=0.10, \
                                                max_area=slabsurface*3, max_length_tol=0.05,\
                                                              max_angle_tol=0.05), \
                                        reorient = False, primitive = False, film_bonds = {('H', 'O'): 3, ('O', 'O'): 5}, \
                                        substrate_bonds = {(material, material): 4})

                # use only those matches that have the slabsurface area
                # = only with untransformed slab surface (matched surface area = slabsurface)

                if not abs((ifb.matches[0]['match_area']-slabsurface)/(slabsurface)) > 0.1:
                    # We create our own interfaces, shift them as we want and adjust vacuum.
                    # This includes in particular in plane shifts (dist) which seems necessary
                    # we assume the substrate primitive cell is smaller than film primitive cell
                    # which is why we shift relative to substrate cell
                    # This might be changed for bigger systems
                    # which is why we use the substrate primitive cell to shift systems
                    # dist is the xy shift distance, choose randomly 1.5

                    # We use an external polarity definition for the polarity, as pymatgen uses
                    # Laue groups which also takes in plane into account, we don't care about that.
                    polarity = films_polarity[water_structs]
                    ifb.combine_slabs_nico(offset = [distz, 0, 0], polar = polarity)
                    interfacelist, interfacelist_tags, newstruct_sub,\
                    newstruct_film, supercelldict  = \
                        create_all_interfaces(ifb, horizontaldist = horizontaldist, distz = distz, vacuum = 17, \
                                              polarity = polarity, primitive_SS_trafo= em )
                    #print("after_builder3")
                    lattlen = [np.linalg.norm(newstruct_sub.lattice.matrix[0]) > limitxylen, \
                    np.linalg.norm(newstruct_sub.lattice.matrix[1]) > limitxylen]

                    areadiff = np.linalg.norm(np.cross(newstruct_sub.lattice.matrix[0],\
                                                       newstruct_sub.lattice.matrix[1]))- \
                    np.linalg.norm(np.cross(primitive_slab.lattice.matrix[0],primitive_slab.lattice.matrix[1]))



                    print( lattlen[0] , lattlen[1] , abs(areadiff))

                    if not lattlen[0] and not lattlen[1] and not abs(areadiff) > 0.001:

                        # We have a substrate centered in the periodic cell
                        # We calculate the symmetry operations that transform substrate into itself
                        # This allows us to use small cells for relaxation etc. and use large cell for control calculation
                        # e.g. prerelaxations
                        # then apply symmetry operation create double slab and recalculate

                        trafo = get_symmtrafo(newstruct_sub)


                        infos_half, newstruct_sub_half = create_onesided_analysestruct(newstruct_sub)


                        newstruct_sub_half_name = export_structure(newstruct_sub_half, folder, material, "sub",\
              millername, "{}_{}_{}_slab_half".format(target_N, water_structs, wsindex))


                        newstruct_film_name = export_structure(newstruct_film, folder, material, "film",\
              millername, "{}_{}_{}_slab".format(target_N, water_structs, wsindex))



                        number_of_water_molec = len(newstruct_film.sites)/3

                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex] = \
                        {"description": film_description}
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["is_polar"] =\
                        films_polarity[water_structs]

                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["film"] = newstruct_film_name
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["subslab_half"] = newstruct_sub_half_name
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["trafo"] = trafo
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["infos_half"] = infos_half
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["supercelldict"] = supercelldict
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["combined_half"] = {}
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["number_of_water_molecules"] = number_of_water_molec
                        all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["number_of_interfaces"] = len(interfacelist)

                        localcount = 0 # to stay consistent

                        for iflisttag, ifi in zip(interfacelist_tags, interfacelist):

                            localcount +=1

                            #print("interface_count", localcount)
                            #atoms = AseAtomsAdaptor.get_atoms(ifi)
                            infos_halfi, half = create_onesided_analysestruct(ifi) #create_onesided_analysestruct
                            # this is only half the interface used fro prerelaxation!
                            # never relax the lowest substrate atoms as we can then create alway in
                            # the center a unrelaxed-bulk-like geometry

                            #view(atoms*(1,1,1))
###
                            # here we calculate the double sided slab
                            #doubled = apply_symmtrafo(trafo, half)
                            #atoms = AseAtomsAdaptor.get_atoms(doubled)
                            #view(atoms*(1,1,1))

                            #if millername == "(100)" and target_N == 10:
                            #    view(AseAtomsAdaptor.get_atoms(half))


                            half_name = export_structure(half, folder, material, "combined",\
              millername, "{}_{}_{}_{}_slab_half".format(target_N, water_structs, wsindex, localcount))
                            llll = {"tag" :iflisttag, "combined_half":half_name, "infos_half" : infos_halfi}
                            all_singlesided_structures[material][millername][lays][target_N][water_structs][wsindex]["combined_half"][localcount] = llll


                    else:
                        pass
            except Exception as e:
                pass
    return return_lattice_info





def export_lattices(best, allmatrices_2Dsorted, millername, target_N, return_fig=False):
    """Export lattices to pdf in the folder 'optimal_lattices'

    Parameters
    ----------
    best : list object
        List of the most isotropic supercell containing stdev_EV, the number of independet supercells,
        kpoint mesh, cell dimensions and supercell matrix
    allmatrices_2Dsorted : list
        List of lists for each possible supercell containing stdev_EV, the number of independet supercells,
        kpoint mesh, cell dimensions and supercell matrix
    millername : str
        Miller indices as str
    target_N : int
        Number of substrate atoms per layer
        
    Returns
    -------
    """
    os.makedirs('optimal_lattices', exist_ok=True)
    fig = plt.figure(figsize = (4.5, 3))

    ax = fig.add_subplot(111)

    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)

    cols = np.array([al[0] for al in allmatrices_2Dsorted])/np.max([al[0] for al in allmatrices_2Dsorted])

    plot_matrices(ax, [al[3] for al in allmatrices_2Dsorted], cols, best[3])

    ax.set_title("surface: {}, #:{}, independent: {}".format(millername, target_N, len(allmatrices_2Dsorted)))
    ax.set_xlabel(r"x ($\AA$)")
    ax.set_ylabel(r"y ($\AA$)")

    l = Lattice(best[3])

    ax.text(0.05,0.6, np.array(best[-1]),transform=ax.transAxes)
    ax.text(0.05,0.9, r'x,y,$\alpha$ (for Pt(fcc)):',transform=ax.transAxes)
    ax.text(0.05,0.8, '{}, {}, {}'.format(np.round(l.lengths[0],1), \
                                                   np.round(l.lengths[1],1), np.round(l.angles[-1],0)),\
           transform=ax.transAxes)
    plt.tight_layout()
    if return_fig:
        return(fig)
    else:
        plt.savefig('optimal_lattices/optimal_{}_{}.pdf'.format(millername, target_N))
        plt.close()


def construct_slab_analyzekp(prim_slab_small, SSmatrix, distance = 0.15, force_parity=False):
    """Construct supercell slab and get the suggested kpoint mesh.

    Parameters
    ----------
    prim_slab_small : pymatgen structure
        Pymatgen structure of the primitive slab
    SSmatrix : 2x2 array
        Supercell matrix
    distance : float
        Minimal spacing of kpoints
    force_parity : boolean
        ???

    Returns
    -------
    ired : int
         Number of irreducible k-points
    kpointsmesh : 3x1 array
        Array of the kpoint mesh in x, y, z
    primitive_slab : 3x3 array
        Cell dimensions
    """
    em = np.zeros((3,3))
    em[:2,:2] = np.array(SSmatrix)
    em[2][2] = 1
    sst = SupercellTransformation(em)
    primitive_slab = sst.apply_transformation(prim_slab_small)
    sa = SpacegroupAnalyzer(primitive_slab)
    rec_cell = primitive_slab.lattice.reciprocal_lattice.matrix
    # I first round to the fifth digit |b|/distance (to avoid that e.g.
    # 3.00000001 becomes 4)
    kpointsmesh = [
        max(int(np.ceil(round(np.linalg.norm(b) / distance, 5))), 1)
        if pbc else 1 for pbc, b in zip([True, True, False], rec_cell)]
    if force_parity:
        kpointsmesh = [k + (k % 2) if pbc else 1
                       for pbc, k in zip([True, True, False], kpointsmesh)]


    ired = len(sa.get_ir_reciprocal_mesh(mesh=kpointsmesh, is_shift=(0, 0, 0)))
    
    return ired, kpointsmesh, primitive_slab.lattice.matrix


def test_equal_lattices(allredmat):
    """??? Why do we check that again

    Parameters
    ----------
    allredmat : list object
        List of the cell dimensions of all indepent supercell matrices in the xy-plane

    Returns
    -------
    newred : list object
        List of the cell dimensions of all indepent supercell matrices in the xy-plane (reduced)
    newredindices  : ???
        ???
    equals  : ???
        ???
    """
    #These matrices should be right handed, as det(A,B) = det(A)det(B)
    red = [np.array(reduce_vectors(rr[0], rr[1])) for rr in allredmat]
    newred = []
    equals = []
    newredindices= []
    for e, r in enumerate(red):
        if newred == []:
            newred.append(r)
            newredindices.append(e)
        else:
            found = False
            for e2, r2 in enumerate(newred):
                if not np.isclose(r , r2).all():
                    rot = np.dot( r, np.linalg.inv(r2) )
                    if np.linalg.norm(rot.astype(int) - rot) < 0.01:
                        equals.append((r, r2))
                        found = True
                        break


                else:
                    found = True
                    equals.append((r, r2))
                    break
            if not found:
                newred.append(r)
                newredindices.append(e)
    return newred, newredindices, equals

def get_isotropic_SS(prim_slab_small, allSSmatrices, target_N):
    """Finds the most isotropic supercell for a given primitive slab structure and target number of surface atoms.

    Parameters
    ----------
    prim_slab_small : pymatgen structure
        Pymatgen structure of the primitive slab
    allSSmatrices : dict
        Dictionary with the independent supercell matrices for target_N
    target_N : int
        Number of substrate atoms per layer

        
    Returns
    -------
    best : list object
        List of the most isotropic supercell containing stdev_EV, the number of independet supercells,
        kpoint mesh, cell dimensions and supercell matrix
    allresults_sorted : list
        List of lists for each possible supercell containing stdev_EV, the number of independet supercells,
        kpoint mesh, cell dimensions and supercell matrix
    """
    lattice_prim = prim_slab_small.lattice.matrix
    matrix = lattice_prim[:2,:2]
    multis = allSSmatrices[target_N]["independent"]
    allmatrices = [ np.dot(mul, matrix) for mul in multis]

    newred, allSSmatrices_indices, equals = test_equal_lattices(allmatrices)

    mr = [multis[kk] for kk in allSSmatrices_indices]
    #The code knows no symmetry therefore more supercells than necessary
    # basically only 111 will be reduced due to lattice similarity
    # In the end we keep only one cell

    optindex_fromnewred, opt_2D, allmatrices_2Dsorted = get_maximally_isotropic_cell(newred, method = 'PCA')


    allresults_sorted = []
    selected_results = []
    # selection criteria, all lower than 0.15 asymmetry, then lowest ired
    for datas in allmatrices_2Dsorted:
        optorigindex = allSSmatrices_indices[datas[2]]
        #print(optorigindex)
        optmulti = multis[optorigindex]
        #we know now the multiand know allproperties
        #here we can do analysis of kpoints
        #for this we really want to make the supercell with vacuum and then multiply and then kpoint analyze
        ired, kpmesh, lattmat = construct_slab_analyzekp(prim_slab_small, \
                                                 optmulti, \
                                                 distance = 0.22, \
                                                 force_parity=False)
        allresults_sorted.append([datas[3], ired, kpmesh, lattmat, optmulti])
        if selected_results == []:
            selected_results.append([datas[3], ired, kpmesh, lattmat, optmulti])
        #this is the best with given asymmetry thresholds
        elif datas[3] < 0.15:
            selected_results.append([datas[3], ired, kpmesh, lattmat, optmulti])
        #this would select the one with highest symmetry according to kp_grid
        #But then it seems we can't match a small 110 surface
        #Therefore we go back to just the most symmetric surface, no kpoint selection
        #best = selected_results[np.argmin([ i[1] for i in selected_results])]
        best = selected_results[0]



    # selection criteria, all lower than 0.15 asymmetry, then lowest ired


    return best, allresults_sorted


def build_grid(vecs, horizontaldist=2., distz=2.2):
    """Export structure

    Parameters
    ----------
    vecs : pymatgen structure
        pymatgen structure of the bulk material
    horizontaldist : float
        The distance with which we scan the different lateral arrangements in the unitcell
    distz : float
        The distance to the surface where we want to place the waters

    Returns
    -------
    allvecs  : list
        ??? Grid for the path of the exported structure
    """
    #vecs defines the substrate unit cell aligned with the supercell
    allvecs = []
    nrx = int(np.ceil(np.linalg.norm(vecs[0])/horizontaldist))
    nry = int(np.ceil(np.linalg.norm(vecs[1])/horizontaldist))

    vxy0 = np.array([0,0,distz])
    for lx in np.linspace(0,1,nrx,endpoint=False):
        for ly in np.linspace(0,1,nry,endpoint=False):
            vxy = vxy0 + lx*vecs[0] + ly*vecs[1]
            #This is because pymatgen is weird in defining the offset z,x,y!
            allvecs.append([lx, ly, (vxy[2], vxy[0], vxy[1])])
    return allvecs

def get_rotation_from_strain_slabs(sub_slab, film_slab):
    """Align the substrate and film by rotation.

    Parameters
    ----------
    sub_slab : pymatgen structure
        pymatgen structure of the bulk material
    film_slab : str
        path of the folder to export structures

    Returns
    -------
    rotation  : str
        path of the exported structure
    """


    """
    copied from pymatgen strain_slabs, to get the rotation matrix
    Strain the film_slab to match the sub_slab,
    orient the structures to match each other,
    and return the new matching structures.
    Nico: See below!!! don't strain substrate!!!!

    Args:
        sub_slab (Slab): substrate supercell slab
        film_slab (Slab): film supercell slab

    Returns:
        sub_struct (Slab): substrate structure oriented
            to match the film supercell
        film_struct (Slab): film structure strained to match
            the substrate supercell lattice.
    """



    sub_struct = sub_slab.copy()
    latt_1 = sub_struct.lattice.matrix.copy()
    film_struct = align_x(film_slab, get_ortho_axes(sub_struct)).copy()
    latt_2 = film_struct.lattice.matrix.copy()

    # Rotate film so its diagonal matches with the sub's diagonal
    diag_vec = np.add(latt_1[0, :], latt_1[1, :])
    sub_norm_diag_vec = diag_vec / np.linalg.norm(diag_vec)
    sub_b = np.cross(sub_norm_diag_vec, [0, 0, 1])
    sub_matrix = np.vstack([sub_norm_diag_vec, sub_b, [0, 0, 1]])

    diag_vec = np.add(latt_2[0, :], latt_2[1, :])
    film_norm_diag_vec = diag_vec / np.linalg.norm(diag_vec)
    film_b = np.cross(film_norm_diag_vec, [0, 0, 1])
    film_matrix = np.vstack([film_norm_diag_vec, film_b, [0, 0, 1]])

    rotation = np.dot(np.linalg.inv(film_matrix), sub_matrix)

    #new_latt = Lattice(np.dot(film_struct.lattice.matrix, rotation))

    #xzy coords should rotate alike
    
    print('get_rotation_from_strain_slabs')
    print(rotation)


    return rotation

def adjust_vacuum(combinedstructure, nrsubstrate, vacuum=17):
    """Generate separate structures from the combined system and adjust them in vacuum

    Parameters
    ----------
    combinedstructure : pymatgen structure
        pymatgen structure of the combined slab and film
    nrsubstrate : str
        path of the folder to export structures
    vacuum : str
        path of the folder to export structures

    Returns
    -------
    newstruct_combined  : pymatgen structure
        pymatgen structure of the combined slab and film adjusted in vacuum
    newstruct_sub  : pymatgen structure
        pymatgen structure of the slab adjusted in vacuum 
    newstruct_film  : pymatgen structure
        pymatgen structure of the film adjusted in vacuum
    """

    #put substrate and film in the same cell
    # at same positions?!

    oldlattice = combinedstructure.lattice.matrix

    allcoords = np.array([kk.coords for kk in combinedstructure.sites ])

    oldstructure_substrate_mean = np.mean(allcoords[:nrsubstrate,2])

    oldstructure_mean = np.mean(allcoords[:,2])

    oldstructure_substrate_top = np.mean(allcoords[:nrsubstrate,2])
    # we should use the mean otherwise we get into trouble with thicker layers
    oldstructure_max = np.max(allcoords[:,2])
    oldstructure_min = np.min(allcoords[:,2])

    slth = oldstructure_max-oldstructure_min

    newlattice = oldlattice.copy()
    newthickness = vacuum + slth
    newlattice[2,2] = newthickness



    newstruct_combined = Structure(
        Lattice(newlattice), combinedstructure.species,
        allcoords + np.array([0, 0, float(newthickness)/2 - oldstructure_substrate_mean ]),
        validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)

    newstruct_sub = Structure(
        Lattice(newlattice), combinedstructure.species[:nrsubstrate],
        allcoords[:nrsubstrate] + np.array([0, 0, float(newthickness)/2 - oldstructure_substrate_mean]),
        validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)

    newstruct_film = Structure(
        Lattice(newlattice), combinedstructure.species[nrsubstrate:],
        allcoords[nrsubstrate:] + np.array([0, 0, float(newthickness)/2 - oldstructure_substrate_mean]),
        validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)

    return newstruct_combined, newstruct_sub, newstruct_film


def create_all_interfaces(ifb, horizontaldist=2., distz=2.2, vacuum=17,
                          polarity=None, primitive_SS_trafo=None):
    """Place the water on the slab

    Parameters
    ----------
    ifb : class ???
        Modified version of the Interface builder initialiyed with the slab and the film structure
    horizontaldist : float
        The distance with which we scan the different lateral arrangements in the unitcell
    distz : float
        The distance to the surface where we want to place the waters
    vacuum : float, optional
        Thickness of vacuum
    polarity : boolean
        Boolean for polar stuctures 
    primitive_SS_trafo : ???
        ???

    Returns
    -------
    interfacelist  : list
        List of pymatgen objects of the constructed interfaces
    interfacelist_tags  : list
        List of strings with tags for the constructed interfacs
    newstruct_sub  : pymatgen structure
        Pymatgen structure of the slab
    newstruct_film  : pymatgen structure
        Pymatgen structure of the film
    supercell_dict  : dict
        ???
    """
    print('create_all_interfaces')

    # We enforce external polarity because this is a better measure than
    # internally (cf symprec) and Laue Problem for the given systems

    # We use our own definition of the interface combine builder
    # Official version stretches both substrate and adsorbed layer for no good reason and with no proper
    # eg. analysis of elastic moduli/volume fractions etc. which would make sense

    ifb.combine_slabs_nico(offset=[distz, 0, 0], polar=polarity) #??? Renaming

    substrate = ifb.strained_substrate
    nrsubstrate = len(substrate.species)

    #subvecs = ifb.matches[0]['sub_vecs'].copy()
    #this could be a problem, as we expected originally to have the primitive

    # Here we recreate the primitive vectors of substrate, so we can shift along them
    subvecs = ifb.substrate_structures[0].lattice.matrix[:2,:]
    if primitive_SS_trafo is None:
        subvecs = ifb.matches[0]['film_vecs'].copy()
        #Then we use the water supercell to create shifts!
        #here we look for rotation matrix
        rotation = get_rotation_from_strain_slabs(ifb.oriented_substrate, ifb.oriented_film)
        #new_latt = Lattice(np.dot(film_struct.lattice.matrix, rotation))
        subvecs = np.dot(subvecs, rotation)

    else:
        try:
            inv = np.linalg.inv(primitive_SS_trafo)
            subvecs = np.dot(subvecs, inv)
        except Exception as e:
            raise e

    supercell = ifb.matches[0]

    supercell_dict = {}

    for k, v in supercell.items():
        try:
            newv = v.tolist()
            supercell_dict[k] = newv
        except:
            supercell_dict[k] = v
            pass
    #??? Why do we reduce the vectors all the time 
    all_subvecs = build_grid(reduce_vectors(subvecs[0], subvecs[1]),
                             horizontaldist=horizontaldist, distz=distz)

    interfacelist = []
    interfacelist_tags = []
    if polarity:
        interfacelist = [[], []]
        interfacelist_tags = [[], []]
    for s in all_subvecs:

        ifb.combine_slabs_nico(offset=s[-1], polar=polarity)
        if polarity:

            for po, ifi in enumerate(ifb.interfaces):
                newstruct_combined, newstruct_sub, newstruct_film = (
                    adjust_vacuum(ifi, nrsubstrate, vacuum=vacuum))
                interfacelist[po].append(newstruct_combined)
                interfacelist_tags[po].extend(
                    [l + '_{}_{}'.format(np.round(s[0], 2), np.round(s[1], 2))
                    for l in [ifb.interface_labels[po]]])

        else:
            for ifi in ifb.interfaces:
                newstruct_combined, newstruct_sub, newstruct_film = (
                    adjust_vacuum(ifi, nrsubstrate, vacuum=vacuum))
                interfacelist.append(newstruct_combined)
            interfacelist_tags.extend(
                [l + '_{}_{}'.format(np.round(s[0], 2), np.round(s[1], 2))
                for l in  ifb.interface_labels])

    if polarity:
        interfacelist = interfacelist[0] + interfacelist[1]
        interfacelist_tags = interfacelist_tags[0] + interfacelist_tags[1]
    print(interfacelist, interfacelist_tags, newstruct_sub,
            newstruct_film, supercell_dict)

    return (interfacelist, interfacelist_tags, newstruct_sub,
            newstruct_film, supercell_dict)