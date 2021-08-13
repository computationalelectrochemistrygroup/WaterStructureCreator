# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module provides classes to identify optimal substrates for film growth
"""

from itertools import product
import numpy as np

from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.surface import (SlabGenerator,
                                   get_symmetrically_distinct_miller_indices)
from pymatgen.analysis.substrate_analyzer import ZSLGenerator, SubstrateAnalyzer, fast_norm, vec_area, gen_sl_transform_matricies 

__author__ = "Shyam Dwaraknath"
__copyright__ = "Copyright 2016, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Shyam Dwaraknath"
__email__ = "shyamd@lbl.gov"
__status__ = "Production"
__date__ = "Feb, 2016"


class ZSLGenerator_mod(ZSLGenerator):
    """
    This class generate matching interface super lattices based on the methodology
    of lattice vector matching for heterostructural interfaces proposed by
    Zur and McGill:
    Journal of Applied Physics 55 (1984), 378 ; doi: 10.1063/1.333084
    The process of generating all possible matching super lattices is:
    1.) Reduce the surface lattice vectors and calculate area for the surfaces
    2.) Generate all super lattice transformations within a maximum allowed area
        limit that give nearly equal area super-lattices for the two
        surfaces - generate_sl_transformation_sets
    3.) For each superlattice set:
        1.) Reduce super lattice vectors
        2.) Check length and angle between film and substrate super lattice
            vectors to determine if the super lattices are the nearly same
            and therefore coincident - get_equiv_transformations
    """


    def generate_sl_transformation_sets(self, film_area, substrate_area):
        """
        Generates transformation sets for film/substrate pair given the
        area of the unit cell area for the film and substrate. The
        transformation sets map the film and substrate unit cells to super
        lattices with a maximum area
        Args:
            film_area(int): the unit cell area for the film
            substrate_area(int): the unit cell area for the substrate
        Returns:
            transformation_sets: a set of transformation_sets defined as:
                1.) the transformation matricies for the film to create a
                super lattice of area i*film area
                2.) the tranformation matricies for the substrate to create
                a super lattice of area j*film area
        Nico: There is a problem with the abs.... it should look as implemented by me
        """

        #Nico changed to my version
        #transformation_indicies = [(i, j)
        #                           for i in range(1, int(self.max_area / film_area))
        #                           for j in range(1, int(self.max_area / substrate_area))
        #                           if np.absolute(film_area / substrate_area - float(j) / i) < self.max_area_ratio_tol]

        transformation_indicies = [(i, j)
                                   for i in range(1, int(self.max_area / film_area))
                                   for j in range(1, int(self.max_area / substrate_area))
                                   if np.absolute(film_area / substrate_area - float(j) / i) < self.max_area_ratio_tol*float(j) / i]


        # Sort sets by the square of the matching area and yield in order
        # from smallest to largest
        for x in sorted(transformation_indicies, key=lambda x: x[0] * x[1]):
            yield (gen_sl_transform_matricies(x[0]),
                   gen_sl_transform_matricies(x[1]))

    def get_equiv_transformations(self, transformation_sets, film_vectors,
                                  substrate_vectors):
        """
        Applies the transformation_sets to the film and substrate vectors
        to generate super-lattices and checks if they matches.
        Returns all matching vectors sets.
        Args:
            transformation_sets(array): an array of transformation sets:
                each transformation set is an array with the (i,j)
                indicating the area multipes of the film and subtrate it
                corresponds to, an array with all possible transformations
                for the film area multiple i and another array for the
                substrate area multiple j.
            film_vectors(array): film vectors to generate super lattices
            substrate_vectors(array): substrate vectors to generate super
                lattices
        """

        for (film_transformations, substrate_transformations) in \
                transformation_sets:
            # Apply transformations and reduce using Zur reduce methodology
            films = [reduce_vectors(*np.dot(f, film_vectors)) for f in film_transformations]

            substrates = [reduce_vectors(*np.dot(s, substrate_vectors)) for s in substrate_transformations]

            # Check if equivalant super lattices
            for (f_trans, s_trans), (f, s) in zip(product(film_transformations, substrate_transformations),
                                                  product(films, substrates)):
                if self.is_same_vectors(f, s):
                    yield [f, s, f_trans, s_trans]

    def __call__(self, film_vectors, substrate_vectors, lowest=False):
        """
        Runs the ZSL algorithm to generate all possible matching
        :return:
        """

        film_area = vec_area(*film_vectors)
        substrate_area = vec_area(*substrate_vectors)

        # Generate all super lattice comnbinations for a given set of miller
        # indicies
        transformation_sets = self.generate_sl_transformation_sets(film_area, substrate_area)

        # Check each super-lattice pair to see if they match
        for match in self.get_equiv_transformations(transformation_sets,
                                                    film_vectors,
                                                    substrate_vectors):
            # Yield the match area, the miller indicies,
            yield self.match_as_dict(match[0], match[1], film_vectors, substrate_vectors, vec_area(*match[0]),
                                     match[2], match[3])

            # Just want lowest match per direction
            if (lowest):
                break


class SubstrateAnalyzer_mod(SubstrateAnalyzer):
    """
    This class applies a set of search criteria to identify suitable
    substrates for film growth. It first uses a topoplogical search by Zur
    and McGill to identify matching super-lattices on various faces of the
    two materials. Additional criteria can then be used to identify the most
    suitable substrate. Currently, the only additional criteria is the
    elastic strain energy of the super-lattices
    """
    

    def __init__(self, zslgen=ZSLGenerator_mod(), film_max_miller=1, substrate_max_miller=1):
        """
            Initializes the substrate analyzer
            Args:
                zslgen(ZSLGenerator): Defaults to a ZSLGenerator with standard
                    tolerances, but can be fed one with custom tolerances
                film_max_miller(int): maximum miller index to generate for film
                    surfaces
                substrate_max_miller(int): maximum miller index to generate for
                    substrate surfaces
        """
        self.zsl = zslgen
        self.film_max_miller = film_max_miller
        self.substrate_max_miller = substrate_max_miller


    #Nico: allow primitive and do not reorient Lattice !!! Otherwise substrate different than before!
    def generate_surface_vectors(self, film_millers, substrate_millers, primitive = False, reorient_lattice=True):
        """
        Generates the film/substrate slab combinations for a set of given
        miller indicies

        Args:
            film_millers(array): all miller indices to generate slabs for
                film
            substrate_millers(array): all miller indicies to generate slabs
                for substrate
        Nico: changed with primitive and reorient option
        """
        vector_sets = []

        for f in film_millers:
            film_slab = SlabGenerator(self.film, f, 20, 15,
                                      primitive=primitive, reorient_lattice = reorient_lattice).get_slab()
            film_vectors = reduce_vectors(film_slab.lattice.matrix[0],
                                          film_slab.lattice.matrix[1])

            for s in substrate_millers:
                substrate_slab = SlabGenerator(self.substrate, s, 20, 15,
                                               primitive=primitive, reorient_lattice = reorient_lattice).get_slab()
                substrate_vectors = reduce_vectors(
                    substrate_slab.lattice.matrix[0],
                    substrate_slab.lattice.matrix[1])

                vector_sets.append((film_vectors, substrate_vectors, f, s))

        return vector_sets

    def calculate(self, film, substrate, elasticity_tensor=None,
                  film_millers=None, substrate_millers=None,
                  ground_state_energy=0, lowest=False, primitive = False):
        """
        Finds all topological matches for the substrate and calculates elastic
        strain energy and total energy for the film if elasticity tensor and
        ground state energy are provided:

        Args:
            film(Structure): conventional standard structure for the film
            substrate(Structure): conventional standard structure for the
                substrate
            elasticity_tensor(ElasticTensor): elasticity tensor for the film
                in the IEEE orientation
            film_millers(array): film facets to consider in search as defined by
                miller indicies
            substrate_millers(array): substrate facets to consider in search as
                defined by miller indicies
            ground_state_energy(float): ground state energy for the film
            lowest(bool): only consider lowest matching area for each surface

        Nico: added primitive option
        """
        self.film = film
        self.substrate = substrate

        # Generate miller indicies if none specified for film
        if film_millers is None:
            film_millers = sorted(get_symmetrically_distinct_miller_indices(
                self.film, self.film_max_miller))

        # Generate miller indicies if none specified for substrate
        if substrate_millers is None:
            substrate_millers = sorted(
                get_symmetrically_distinct_miller_indices(self.substrate,
                                                          self.substrate_max_miller))

        # Check each miller index combination
        #Nico: added primitive option and reorient option. Why would you
        surface_vector_sets = self.generate_surface_vectors(film_millers, substrate_millers, primitive=primitive, reorient_lattice = False)
        for [film_vectors, substrate_vectors, film_miller, substrate_miller] in surface_vector_sets:
            for match in self.zsl(film_vectors, substrate_vectors, lowest):
                match['film_miller'] = film_miller
                match['sub_miller'] = substrate_miller
                if elasticity_tensor is not None:
                    energy, strain = self.calculate_3D_elastic_energy(
                        film, match, elasticity_tensor, include_strain=True)
                    match["elastic_energy"] = energy
                    match["strain"] = strain
                if ground_state_energy != 0:
                    match['total_energy'] = match.get('elastic_energy', 0) + ground_state_energy

                yield match


def reduce_vectors_pmg(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)

    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))

    return [a, b]


def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    Nico: that is all nice but then we create also left/handed systems
    I think this should be avoided
    why not rather allow for negative a or rather longer a axis?
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)

    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))
    #Nico adjusted det check
    # Careful this function is called with 2d and 3D vectors
    # For our purpose zaxis will always be along 001
    # This might fail in general usecase
    if len(a) == 3:
        bigmatrix = np.zeros((3,3))
        bigmatrix[:2,:] = np.array([a,b])
        bigmatrix[2,2] = 1.
        if np.linalg.det(bigmatrix) < 0:
            return [b, a]
        else:
            return [a, b]
    else:
        if np.linalg.det([a,b]) < 0:
            return [b, a]
        else:
            return [a, b]
