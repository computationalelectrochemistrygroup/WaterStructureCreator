import copy
import numpy as np
from itertools import combinations, product

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
SupercellTransformation)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import (
    ZSLGenerator_mod)


########### Create substrate slabs

def get_export_subslabs_clean(bulk, miller, millername, layers, material, folder, vacuum = 17):
    """Create clean primitive slab from bulk

    Parameters
    ----------
    bulk : pymatgen structure
        pymatgen structure of the bulk material
    miller : list
        list of miller indices
    millername : str
        miller indices as str
    layers : int
        number of atomic layers
    material : str
        name of the substrate
    folder : str
        path of the folder to export structures
    vacuum : float, optional
        Thickness of vacuum

    Returns
    -------
    cleanslab_dict  : dict
        dictionary of the bulk, full and half slab strucrue + some additional information
    primitive_slab  : pymatgen structure
        pymatgen structure of the slab
    """

    oriented_primitive_bulk_o, primitive_slab = get_bulk_and_slab(bulk,
                                                                  miller=miller,
                                                                  layers=layers,
                                                                  vacuum=vacuum)

    #this is important!
    primitive_slab = adjust_vacuum_clean(primitive_slab, vacuum)

    trafo = get_symmtrafo(primitive_slab)

    infos, half = create_onesided_analysestruct(primitive_slab) #infos, newstruct

    millersubbulk = export_structure(oriented_primitive_bulk_o,
                                     folder, material, "sub",
                                     millername, "bulk")

    millersubslab = export_structure(primitive_slab, folder, material, "sub",
                                     millername, "slab")

    millersubslab_half = export_structure(half, folder, material, "sub",
                                          millername, "slab_half")


    cleanslab_dict = {"millersubbulk": millersubbulk,
                      "millersubslab": millersubslab,
                      "millersubslab_half": millersubslab_half,
                      "trafo": trafo,
                      "infos_half": infos,
                     }

    return cleanslab_dict, primitive_slab


def get_bulk_and_slab(bulk, miller=[1,1,1], layers=4, vacuum=16):
    """Create a slab and conventional bulk cell from a bulk cell input

    Parameters
    ----------
    bulk : pymatgen structure
        pymatgen structure of the bulk material
    miller : list
        list of miller indices
    layers : int
        number of atomic layers
    vacuum : float, optional
        thickness of vacuum

    Returns
    -------
    oriented_primitive_bulk_o  : pymatgen structure
        pymatgen structure of the bulk
    primitive_slab  : pymatgen structure
        pymatgen structure of the slab
    """
    #vaccum is now also in unit planes!!!! we adjust vacuum anyways in the end
    # to do. set absolute thickness and then calculate how many layers these are, making it
    # an even number in total, so no atom is exactlty  at 0.5 so we have always one central
    # layer that is unrelaxed when doubling the cell!!!
    # Achtung: reorient lattice has Problems: orthogonal cell is the wrong!!!
    # so do it by hand via newstructure lattice

    sl = SlabGenerator(bulk, miller, layers, vacuum, lll_reduce=True,
                       center_slab=True, in_unit_planes=True, primitive=True,
                       max_normal_search=None, reorient_lattice=False)

    slab = sl.get_slab()


    primitive_slab = slab.get_orthogonal_c_slab()

    inplaneshift = primitive_slab.frac_coords[np.argmax(primitive_slab.frac_coords[:,2])]
    inplaneshift[2] = 0

    primitive_slab = Structure(
        Lattice.from_lengths_and_angles(
            primitive_slab.lattice.lengths, primitive_slab.lattice.angles),
        primitive_slab.species, primitive_slab.frac_coords-inplaneshift,
        validate_proximity=True, to_unit_cell=True,
        coords_are_cartesian=False,)


    slab_bulkref = slab.oriented_unit_cell

    #The bulkref is not primitive and not oriented like slab!!!

    zgen = ZSLGenerator_mod()
    atoms = AseAtomsAdaptor.get_atoms(slab_bulkref)
    res = list(zgen(slab_bulkref.lattice.matrix[:2,:],
                    slab.lattice.matrix[:2,:], lowest=True))

    #Attention: ZSLgen uses reduced_vectors (Zur) which randomly interchanges a and b vectors.
    #This is totally shit to get to the same supercell. As we cannot in this way get the real transformation


    tests = [np.array(i) for i in list(combinations(list(product([1, 0, -1] , repeat = 2)), 2))
             if np.isclose(np.abs(np.linalg.det(np.array(i))), 1.)]
    for t in tests:
        tt = np.dot(t, np.dot(res[0]['substrate_transformation'], slab.lattice.matrix[:2,:]))

        if np.isclose(slab_bulkref.lattice.matrix[:2,:]-tt, 0).all():
            break
            inv = np.linalg.inv(np.dot(t, res[0]['substrate_transformation']))
            break


    backtrafomatrix = np.linalg.inv(
        np.array([t[0].tolist() + [0], t[1].tolist() + [0], [0,0,1]])).astype(int)


    sst = SupercellTransformation(backtrafomatrix)
    newbulkcell = sst.apply_transformation(slab_bulkref)
    t = res[0]['substrate_transformation']
    bstrafo = np.array([t[0].tolist() + [0], t[1].tolist() + [0], [0,0,1]])
    prim_bulk_cell = np.dot(  np.linalg.inv(bstrafo), newbulkcell.lattice.matrix)


    # Here we find the in-plane primitive lattice vectors for the bulk cell
    # it seems the lattice is still in directions xyz as the bulk.
    # this is nice because then we can get the exact rotation matrix w.r.t. the bulk conventional cell
    # one could implement the strain contributions here
    # Now we could take over the lattice directly from the slab structure and put e.g. also all slab atóms in the bulk cell
    #they are still not aligned in xyz, which we want to do now!!!

    tests = Structure(Lattice(prim_bulk_cell), [list(newbulkcell.species)[0]],
                      [newbulkcell.cart_coords[0]], validate_proximity=True,
                      to_unit_cell=True, coords_are_cartesian=True)



    species = newbulkcell.species
    coords = newbulkcell.cart_coords
    s = tests.copy()

    # we add the other atoms
    for i, sp in enumerate(species):
        try:
            s.insert(i, sp, coords[i],\
                     coords_are_cartesian=True,\
                     validate_proximity=True)
        except:
            pass

    oriented_primitive_bulk = s.get_sorted_structure()

    #put into cell
    oriented_primitive_bulk = Structure(oriented_primitive_bulk.lattice,
                                        oriented_primitive_bulk.species,
                                        oriented_primitive_bulk.cart_coords,
                                        validate_proximity=True,to_unit_cell=True,
                                        coords_are_cartesian=True)


    def test(matrix1, matrix2):
        vecs = (np.isclose(np.linalg.norm(matrix1[0]), np.linalg.norm(matrix2[0]))
                and np.isclose(np.linalg.norm(matrix1[2]), np.linalg.norm(matrix2[2])))
        r = np.cross(matrix1[0], matrix1[1])
        right = (np.dot(r, matrix1[2]) > 0)
        return vecs, right


    combinationslist = [[[1,0],[0,1]], [[-1,0],[0,1]], [[-1,0],[0,-1]], [[1,0],[0,-1]],\
                    [[0,1],[1,0]], [[0,-1],[1,0]], [[0,-1],[-1,0]], [[0,1],[-1,0]], ]

    for c in combinationslist:
        for c3 in [1,-1]:
            m = np.zeros((3,3))
            m[:2,:2] = np.array(c)
            m[2,2] = c3
            newm = np.dot(m, oriented_primitive_bulk.lattice.matrix)
            vecs, right = test(newm, primitive_slab.lattice.matrix)
            if vecs and right:
                break

    sst = SupercellTransformation(m.astype(int))
    oriented_primitive_bulk = sst.apply_transformation(oriented_primitive_bulk)

    #this is the primitive bulk, with surface spanned by 0 and 1 component but not oriented!

    #slab is already orthogonalized an orthognonalized slab

    primitive_slab_L = primitive_slab.lattice.matrix
    primitive_slab_LTM2 = np.cross(primitive_slab_L[0], primitive_slab_L[1])
    primitive_slab_LTM2 /= np.linalg.norm(primitive_slab_LTM2)
    primitive_slab_LT = [primitive_slab_L[0], primitive_slab_L[1], primitive_slab_LTM2]
    # this is prim slab lattice matrix with 1 length in zdir
    # z-component does not matter
    # this is a fake lattice to find rotation matrix in 3D

    #oriented prim bulk is oriented as slab abnd not as the the orthogonalized prim slab lattice
    oriented_primitive_bulk_L = oriented_primitive_bulk.lattice.matrix
    oriented_primitive_bulk_LTM2 = np.cross(oriented_primitive_bulk_L[0],
                                            oriented_primitive_bulk_L[1])
    oriented_primitive_bulk_LTM2 /= np.linalg.norm(oriented_primitive_bulk_LTM2)
    oriented_primitive_bulk_LT = [oriented_primitive_bulk_L[0],
                                  oriented_primitive_bulk_L[1], oriented_primitive_bulk_LTM2]
    # this is a fake lattice to find rotation matrix in 3D

    #it should be tested if this is really a rot (LH and RH lattice is enforced by cross)
    #Note there could be still lattice vector 1 be lattice vector 2



    rot = np.dot(np.linalg.inv(oriented_primitive_bulk_LT), primitive_slab_LT)

    print("THIS VALUE SHOULD BE 1 ALWAYS", np.linalg.det(rot))

    oriented_primitive_bulk_lattice = np.dot( oriented_primitive_bulk_L, rot )



    oriented_primitive_bulk_o = Structure(Lattice(oriented_primitive_bulk_lattice),
                                          oriented_primitive_bulk.species,
                                          oriented_primitive_bulk.frac_coords,
                                          validate_proximity=True,
                                          to_unit_cell=True,
                                          coords_are_cartesian=False)


    return oriented_primitive_bulk_o, primitive_slab


def adjust_vacuum_clean(combinedstructure, vacuum=17):
    """Align slab in vacuum

    Parameters
    ----------
    combinedstructure : pymatgen structure
        pymatgen structure of the bulk material
    vacuum : float, optional
        thickness of vacuum

    Returns
    -------
    newstruct_combined  : pymatgen structure
        pymatgen structure of the slab
    """

    #put substrate and film in the same cell
    # at same positions?!

    oldlattice = combinedstructure.lattice.matrix

    allcoords = np.array([kk.coords for kk in combinedstructure.sites])

    oldstructure_substrate_mean = np.mean(allcoords[:,2])

    oldstructure_max = np.max(allcoords[:,2])
    oldstructure_min = np.min(allcoords[:,2])

    slth = oldstructure_max-oldstructure_min

    newlattice = oldlattice.copy()
    newthickness = vacuum + slth
    newlattice[2,2] = newthickness

    newstruct_combined = Structure(
        Lattice(newlattice), combinedstructure.species,
        allcoords + np.array([0,0, float(newthickness)/2-oldstructure_substrate_mean ]),
        validate_proximity=True, to_unit_cell=True, coords_are_cartesian=True)

    return newstruct_combined


def get_symmtrafo(newstruct_sub):
    """???

    Parameters
    ----------
    newstruct_sub : pymatgen structure
        pymatgen structure of the bulk material

    Returns
    -------
    trafo  : ???
        ???
    """
    sg = SpacegroupAnalyzer(newstruct_sub)
    trr = sg.get_symmetry_dataset()
    trafo = []
    for index, op in enumerate(trr['rotations']):
        if np.linalg.norm(np.array([0,0,-1]) - op[2]) < 0.0000001 and np.linalg.det(op) > 0 :
            #print('transformation found' ,op, index, trr['translations'][index])
            trafo ={'rot_frac': op.tolist(), 'trans_frac': trr['translations'][index].tolist() }
            break

    # Now we have the trafo (to be used on fractional coordinates)

    if trafo == []:
        for index, op in enumerate(trr['rotations']):
            if np.linalg.norm(np.array([0,0,-1]) - op[2]) < 0.0000001:
                #print('transformation found' ,op, index, trr['translations'][index])
                trafo ={'rot_frac': op.tolist(), 'trans_frac': trr['translations'][index].tolist() }
                break

    return trafo


def create_onesided_analysestruct(struct):
    """Create half slab

    Parameters
    ----------
    struct : pymatgen structure
        pymatgen structure of the bulk material

    Returns
    -------
    infos  : dict
        dictionary of slab structure and their file path
    newstruct  : pymatgen structure
        pymatgen structure of the slab
    """
    newlatt = copy.copy(struct.lattice.matrix)

    temppos = [pos[2] for i, pos in enumerate(struct.cart_coords)
               if abs(pos[2])/abs(struct.lattice.matrix[2][2]) > 0.4999]

    th = max(temppos) - min(temppos)

    target = th + 12 # prerelax with 12 AA vacuum

    newlatt[2][2] = target
    shift = np.array([0, 0, target/2 - np.mean(temppos)])

    allposi = [(i, np.array(pos)+ shift ) for i, pos in enumerate(struct.cart_coords)
               if abs(pos[2])/abs(struct.lattice.matrix[2][2]) > 0.4999]
    allposi = sorted(allposi, key=lambda points: points[1][2])

    allsp = [struct.sites[i[0]].specie.symbol for i in  allposi]

    allpos = [i[1] for i in allposi]
    lower = min(np.array(allpos)[:,2])

    fixed = ["fixed" if i[1][2] < lower+0.01 else "free" for i in allposi]


    newstruct = Structure(
        Lattice(newlatt), allsp, allpos, validate_proximity=True,
        to_unit_cell=True, coords_are_cartesian=True)

    infos = {"double_lattice" : struct.lattice.matrix,
             "single_lattice": newstruct.lattice.matrix,
             "addshift_tosingle_toget_double": -shift,
             "fixed" : fixed}

    return infos, newstruct


def export_structure(struct, folder, material, stype, miller, sname):
    """Export structure

    Parameters
    ----------
    struct : pymatgen structure
        pymatgen structure of the bulk material
    folder : str
        path of the folder to export structures
    material : str
        name of the substrate
    stype : str
        additional str for personalization
    miller : list
        list of miller indices
    sname : str
        additional str for personalization

    Returns
    -------
    fname  : str
        path of the exported structure
    """
    filedict  = {}
    p = Poscar(struct)
    fname = folder + "{}_{}_{}_{}.vasp".format(material, stype, miller, sname)
    p.write_file(fname)
    return fname