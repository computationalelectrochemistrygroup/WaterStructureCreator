import time
import numpy as np
from itertools import product, combinations

#make voronoi optimum
from sklearn.decomposition import PCA

from pymatgen.core.lattice import Lattice
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import reduce_vectors


def get_maximally_isotropic_cell(matrices, method='PCA'):
    """
    applies the analyse method function to identify the isotroproic supercell among

    Parameters
    ----------
    matrices : list
        list of 2x2 irreducible matrices to define the surfaces.
        The dimension of the list is given by the number of matrices
        found for a fixed number of surface atoms.
    method : str
        method to search the isotropic supercell. There are two: 'v' and 'PCA'.

    Returns
    -------
    tuple of dimension 3
        index of the optimal matrix.
        list of 5 elements with information of the opmtimization method, list[1] matrix 2x2.
        list of all matrices sorted by stdev_EV

    """
    opt_2D, allmatrices2 = analyse_voronoi(matrices, method=method)
    allmatrices_2Dsorted = sorted(allmatrices2, key=lambda x: x[3])

    return opt_2D[2], opt_2D, allmatrices_2Dsorted
## TODO: method str to Enum


def analyse_voronoi(matrices, method='v'):
    """
    find the matrix with the lowest stdev_EV (isotropic supercell)

    Parameters
    ----------
    matrices : list
        list of 2x2 irreducible matrices to define the surfaces.
        The dimension of the list is given by the number of matrices
        found for a fixed number of surface atoms.
    method : str
        method to search the isotropic supercell. There are two: 'v' and 'PCA'.

    Returns
    -------
    tuple of dimension 2
        list (5 elements), the results for the isotropic supercell
        list (same dimension as matrices), each element is a list (5-D)

    """
    opt_all = []
    allresults = []
    for index, km in enumerate(matrices):
        if method == 'v':
            verify, stdev_EV, coveigval = analyse_lattice_points_kmesh_voronoi(
                km)
        else:
            verify, stdev_EV, coveigval = analyse_lattice_points_kmesh_voronoi_PCA(
                km)
            #print('PCA', verify, stdev_EV, coveigval)
        if verify:
            allresults.append([verify, km, index, stdev_EV, coveigval])
        if verify:
            if opt_all == []:
                opt_all = [verify, km, index, stdev_EV, coveigval]
            elif stdev_EV < opt_all[3]:
                opt_all = [verify, km, index, stdev_EV, coveigval]

    return opt_all, allresults
### TODO: 2 ifs -> 1 if, results (list of lists -> panda dataframe)


def analyse_lattice_points_kmesh_voronoi(recarr_maybereduced, fast=True, prescreen_line_dens=True):
    """Short summary.

    Parameters
    ----------
    recarr_maybereduced : array
        array 2x2, reduced matrix for surface
    fast : bool
        applies a threshold for the generated kpoints
    prescreen_line_dens : bool
        ?

    Returns
    -------
    tuple of dimension 3
        bool, threshold condition
        float, stdev_EV of the eingenvalues from covariance matrix1
        list, list of eigenvalues

    """

    recarr = reduce_vectors(recarr_maybereduced[0], recarr_maybereduced[1])

    if np.linalg.det(recarr) < 0:
        recarr = np.array([-recarr[0], recarr[1]])

    bx = recarr[0]
    by = recarr[1]

    stepx = np.linalg.norm(bx)
    stepy = np.linalg.norm(by)

    lena = np.array([stepx, stepy])

    maxlen = max(lena)
    minlen = min(lena)

    recvol = np.linalg.det(recarr)

    if prescreen_line_dens:
        if minlen < 0.03 * recvol**(1./2):
            #very assymmetric cells
            #print "LINE DENS LINE DENS LINE DENSLINE DENSLINE DENS LINE DENS LINE DENS LINE DENSLINE DENSLINE DENSv"
            return False, 0, 0, 0
        else:

            # Construction of adjacent kpoints around [0,0,0]
            # This is rather heuristic and in principle problematic for highly mono/triclinic primitive cells
            # of the kpoint mesh.
            # However, in all relevant cases, i.e. the targeted isotropic systems, the Voronoi
            # cell will be consistently constructed.
            # As a remark. pymatgen.core.lattice.Lattice.get_wigner_seitz_cell
            # does it even worse!!! they only construct all neighbouring cells.
            # This corresponds to make_a_reasonable_supercell = np.array([0, 1, -1]).

            ipoints = []

            surrounding = [0., 1., -1., 2, -2, 3, -4, -3, 4]
            #for i, j, k in product(scnew[0], scnew[1], scnew[2]):
            for i, j in product(surrounding, surrounding):
                ipoints.append(i * bx + j * by)

            # Recution of points for Voronoi decomposition for speedup.
            # This is rather heuristic again and but not problematic.
            # The reduction of points within a sphere of radius radmax is non-problematic for all interesting
            # cases i.e. isotropic systems. E.g. all neighbouring points in cubic system with lattice length of
            # stepx are within a sphere of = sqrt(3)*(stepx*stepy*stepz)**(1./3) = 1.44*stepx
            # here we accept points within a sphere of 3.5 times this optimum size.
            # In case of non-isotropic systems this might lead to a point cloud,
            # where the Vornoi cell construction fails.
            # These exceptions are captured but this does not impact the result, as we are however not interested in those
            # kpoint meshes. There is no need for the construction to work out.
            # We save significant amount of time by limiting the size of the kpoint cloud for
            # Voronoi construction, as before for constructing large enough point clouds.
            if fast:
                radmax = 4.*recvol**(1./2)
                points = np.array(
                    [ip for ip in ipoints if np.linalg.norm(ip) < radmax])

            try:
                #oldmethod: does not work for very symmetric cells
                # as one can fit multiple ellipses through
                #vor = Voronoi(points)
                #pointset = []
                #vorvertices = vor.vertices
                #for indi in vor.regions[vor.point_region[0]]:
                #    pointset.append(vorvertices[indi])

                #pointsa = np.array(pointset)
                ##print np.shape(pointsa)
                #
                # newmethod: creates add. points on the convex hull
                pointset = get_voronoi_edges_edgepoints(points)

                pointsa = np.array(pointset)
                pmean = pointsa - np.mean(pointsa, axis=0)
                covarr = np.cov(pmean, rowvar=False)
                coveigval, coveigv = np.linalg.eig(covarr)
                stdev_EV = np.std(coveigval/np.mean(coveigval))
                # This checks if we have a 3d system of points or just coplanar
                # is probably not necessary as the voronoi construction will
                # already fail for 2d point cloud in 3d
                if all([abs(i) > 10**-5 for i in coveigval]):

                    return True,  stdev_EV, coveigval.tolist()
                else:
                    return False,  stdev_EV, coveigval.tolist()
            except:
                return False, 0, 0
    else:
        return False, 0, 0
### TODO: WHY ALWAYS REDUCING VECTORS?


def analyse_lattice_points_kmesh_voronoi_PCA(recarr_maybereduced, fast=True, prescreen_line_dens=True):
    """eigenvalues calculated from the pca method (single_values_)

    Parameters
    ----------
    recarr_maybereduced : array
        array of 2x2.
    fast : bool
        applies a threshold for the generated kpoints
    prescreen_line_dens : bool
        ?.

    Returns
    -------
    tuple of dimension 3
        bool, threshold condition
        float, stdev_EV of the eingenvalues
        list, list of eigenvalues

    """
    #? Why always applying reduced_vectors?
    recarr = reduce_vectors(recarr_maybereduced[0], recarr_maybereduced[1])

    if np.linalg.det(recarr) < 0:
        recarr = np.array([-recarr[0], recarr[1]])

    bx = recarr[0]
    by = recarr[1]

    stepx = np.linalg.norm(bx)
    stepy = np.linalg.norm(by)

    lena = np.array([stepx, stepy])

    maxlen = max(lena)
    minlen = min(lena)

    recvol = np.linalg.det(recarr)

    if prescreen_line_dens:
        if minlen < 0.03 * recvol**(1./2):
            #very assymmetric cells
            #print "LINE DENS LINE DENS LINE DENSLINE DENSLINE DENS LINE DENS LINE DENS LINE DENSLINE DENSLINE DENSv"
            return False, 0, 0, 0
        else:
            # Construction of adjacent kpoints around [0,0,0]
            # This is rather heuristic and in principle problematic for highly mono/triclinic primitive cells
            # of the kpoint mesh.
            # However, in all relevant cases, i.e. the targeted isotropic systems, the Voronoi
            # cell will be consistently constructed.
            # As a remark. pymatgen.core.lattice.Lattice.get_wigner_seitz_cell
            # does it even worse!!! they only construct all neighbouring cells.
            # This corresponds to make_a_reasonable_supercell = np.array([0, 1, -1]).

            ipoints = []

            surrounding = [0., 1., -1., 2, -2, 3, -4, -3, 4]
            #for i, j, k in product(scnew[0], scnew[1], scnew[2]):
            for i, j in product(surrounding, surrounding):
                ipoints.append(i * bx + j * by)

            # Recution of points for Voronoi decomposition for speedup.
            # This is rather heuristic again and but not problematic.
            # The reduction of points within a sphere of radius radmax is non-problematic for all interesting
            # cases i.e. isotropic systems. E.g. all neighbouring points in cubic system with lattice length of
            # stepx are within a sphere of = sqrt(3)*(stepx*stepy*stepz)**(1./3) = 1.44*stepx
            # here we accept points within a sphere of 3.5 times this optimum size.
            # In case of non-isotropic systems this might lead to a point cloud,
            # where the Vornoi cell construction fails.
            # These exceptions are captured but this does not impact the result, as we are however not interested in those
            # kpoint meshes. There is no need for the construction to work out.
            # We save significant amount of time by limiting the size of the kpoint cloud for
            # Voronoi construction, as before for constructing large enough point clouds.
            if fast:
                radmax = 4.*recvol**(1./2)
                points = np.array(
                    [ip for ip in ipoints if np.linalg.norm(ip) < radmax])

            try:
                #oldmethod: does not work for very symmetric cells
                # as one can fit multiple ellipses through
                #vor = Voronoi(points)
                #pointset = []
                #vorvertices = vor.vertices
                #for indi in vor.regions[vor.point_region[0]]:
                #    pointset.append(vorvertices[indi])

                #pointsa = np.array(pointset)
                ##print np.shape(pointsa)
                #
                # newmethod: creates add. points on the convex hull
                pointset = get_voronoi_edges_edgepoints(points)
                pointsa = np.array(pointset)

                pmean = pointsa - np.mean(pointsa, axis=0)
                pc = PCA(2)
                pc.fit(pmean)
                eigval = pc.singular_values_

                stdev_EV = np.std(eigval/np.mean(eigval))
                # This checks if we have a 3d system of points or just coplanar
                # is probably not necessary as the voronoi construction will
                # already fail for 2d point cloud in 3d
                if all([abs(i) > 10**-5 for i in eigval]):

                    return True,  stdev_EV, eigval.tolist()
                else:
                    return False,  stdev_EV, eigval.tolist()
            except Exception as e:
                #print('eception', e)
                return False, 0, 0
    else:
        return False, 0, 0


def get_voronoi_edges_edgepoints(points):
    """Short summary.

    Parameters
    ----------
    points : array
        previously created from the surface vectors

    Returns
    -------
    list
        voronoi vertices that and midpoints

    """

    vor = Voronoi(points)
    pointset = []
    vorvertices = vor.vertices
    for indi in vor.regions[vor.point_region[0]]:
        pointset.append(vorvertices[indi])
    pointsa = np.array(pointset)
    twodpoints = get_voronoi_convex_hull(pointsa)
    addpoints = create_midpoints(twodpoints, dist=.5)

    opoints = pointsa.tolist() + addpoints
    return opoints


def get_voronoi_convex_hull(vorverticespoints):
    """Short summary.

    Parameters
    ----------
    vorverticespoints : array
        voronoi vertices

    Returns
    -------
    list, each element an array of 2D
        voronoi vertix points from the perimeter

    """
    verticespoints = np.array(vorverticespoints)
    simplices = ConvexHull(verticespoints).simplices
    poly2d = [[verticespoints[simplices[ix][iy]] for iy in range(len(simplices[0]))]
              for ix in range(len(simplices))]
    return poly2d


def create_midpoints(startendvecs, dist=.5):
    """Short summary.

    Parameters
    ----------
    startendvecs : list
        a subset of voronoi vertices
    dist : float
        .

    Returns
    -------
    list
        each element is a 2D list


    """
    #dist + normed for longest distance
    n = int(1./dist)
    distreal = 1./n
    allpoints_onsides = []
    for a in startendvecs:
        newpoints = [[a[0] + alpha * (a[1]-a[0]), a[0] + (alpha+distreal) * (a[1]-a[0])]
                     for alpha in np.linspace(0, 1, n, endpoint=False)]
        newpoints_on_sides = np.array(newpoints)[:, 0]
        nl = newpoints_on_sides.tolist()
        nl.pop(0)
        allpoints_onsides.extend(nl)
    return allpoints_onsides
