# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module provides classes to store, generate, and manipulate material interfaces.
"""

from pymatgen.core.surface import SlabGenerator
from pymatgen import Lattice, Structure
from pymatgen.core.surface import Slab
from itertools import product
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matplotlib import pyplot as plt
from pymatgen.core.operations import SymmOp
from matplotlib.lines import Line2D
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.sites import PeriodicSite
#from pymatgen.analysis.substrate_analyzer import (SubstrateAnalyzer, reduce_vectors)
import warnings

#from substrate_analyzer_mod import (SubstrateAnalyzer_mod, reduce_vectors)
from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import (SubstrateAnalyzer_mod, reduce_vectors)
from pymatgen.analysis.interface import InterfaceBuilder, Interface, align_x, get_ortho_axes, merge_slabs

__author__ = "Eric Sivonxay, Shyam Dwaraknath, and Kyle Bystrom"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Kyle Bystrom"
__email__ = "kylebystrom@gmail.com"
__date__ = "5/29/2019"
__status__ = "Prototype"


class InterfaceBuilder_mod(InterfaceBuilder):
    """
    This class constructs the epitaxially matched interfaces between two crystalline slabs
    """


    def generate_interfaces(self, film_millers=None, substrate_millers=None, film_layers=3, substrate_layers=3,
                            **kwargs):
        """
        Generate a list of Interface (Structure) objects and store them to self.interfaces.

        Args:
            film_millers (list of [int]): list of film surfaces
            substrate_millers (list of [int]): list of substrate surfaces
            film_layers (int): number of layers of film to include in Interface structures.
            substrate_layers (int): number of layers of substrate to include in Interface structures.
        """
        self.get_oriented_slabs(lowest=True, film_millers=film_millers,
                                substrate_millers=substrate_millers, film_layers=film_layers,
                                substrate_layers=substrate_layers)

        self.combine_slabs(**kwargs)
        return


    #Nico: changed input and stuff
    def get_oriented_slabs(self, film_layers=3, substrate_layers=3, match_index=0, do_sym_mod = False, reorient = True,  **kwargs):
        """
        Get a list of oriented slabs for constructing interfaces and put them
        in self.film_structures, self.substrate_structures, self.modified_film_structures,
        and self.modified_substrate_structures.
        Currently only uses first match (lowest SA) in the list of matches
        
        Nico: I added reorient parameter so the lattice is not automatically turned.
        If that happens it is basically impossible to get back primitive cell

        Args:
            film_layers (int): number of layers of film to include in Interface structures.
            substrate_layers (int): number of layers of substrate to include in Interface structures.
            match_index (int): ZSL match from which to construct slabs.
        """
        self.match_index = match_index
        self.substrate_layers = substrate_layers
        self.film_layers = film_layers

        if 'zslgen' in kwargs.keys():
            sa = SubstrateAnalyzer_mod(zslgen=kwargs.get('zslgen'))
            del kwargs['zslgen']
        else:
            sa = SubstrateAnalyzer_mod()


        #Nico:
        if "film_bonds" in kwargs.keys():
            bonds = kwargs.get("film_bonds")
            del kwargs["film_bonds"]
        else:
            bonds = None
        if "substrate_bonds" in kwargs.keys():
            subsbonds = kwargs.get("substrate_bonds")
            del kwargs["substrate_bonds"]
        else:
            subsbonds = None


        #Nico: add primitive functionality

        primitive_key = False

        if "primitive" in kwargs.keys():
            primitive_key = kwargs.get("primitive")
            del kwargs["primitive"]
            



        # Generate all possible interface matches
        self.matches = list(sa.calculate(self.original_film_structure, self.original_substrate_structure, **kwargs))
        match = self.matches[match_index]

        # Generate substrate slab and align x axis to (100) and slab normal to (001)
        # Get no-vacuum structure for strained bulk calculation
        self.sub_sg = SlabGenerator(self.original_substrate_structure, match['sub_miller'], substrate_layers, 0,
                                    in_unit_planes=True,
                                    reorient_lattice=False,
                                    primitive=primitive_key)
        no_vac_sub_slab = self.sub_sg.get_slab()
        no_vac_sub_slab = get_shear_reduced_slab(no_vac_sub_slab)
        if reorient:
            self.oriented_substrate = align_x(no_vac_sub_slab)
        else:
            self.oriented_substrate = no_vac_sub_slab
        self.oriented_substrate.sort()

        # Get slab with vacuum
        self.sub_sg = SlabGenerator(self.original_substrate_structure, match['sub_miller'], substrate_layers, 1,
                                    in_unit_planes=True,
                                    reorient_lattice=False,
                                    primitive=primitive_key)
        #Nico: added substrate bonds
        sub_slabs = self.sub_sg.get_slabs(bonds =subsbonds)
        for i, sub_slab in enumerate(sub_slabs):
            sub_slab = get_shear_reduced_slab(sub_slab)
            if reorient:
                sub_slab = align_x(sub_slab)
            
            sub_slab.sort()
            sub_slabs[i] = sub_slab

        self.substrate_structures = sub_slabs

        # Generate film slab and align x axis to (100) and slab normal to (001)
        # Get no-vacuum structure for strained bulk calculation
        self.film_sg = SlabGenerator(self.original_film_structure, match['film_miller'], film_layers, 0,
                                     in_unit_planes=True,
                                     reorient_lattice=False,
                                     primitive=primitive_key)
        
        no_vac_film_slab = self.film_sg.get_slab()
        no_vac_film_slab = get_shear_reduced_slab(no_vac_film_slab)
        if reorient:
            self.oriented_film = align_x(no_vac_film_slab)
        else:
            self.oriented_film = no_vac_film_slab
        self.oriented_film.sort()

        # Get slab with vacuum
        # Nico: We add bonds so no movement of film slabs (e.g. h2o adlayers)

        self.film_sg = SlabGenerator(self.original_film_structure, match['film_miller'], film_layers, 1,
                                     in_unit_planes=True,
                                     reorient_lattice=False,
                                     primitive=primitive_key)
        # Nico: here I added bonds
        film_slabs = self.film_sg.get_slabs(bonds = bonds)
        for i, film_slab in enumerate(film_slabs):
            film_slab = get_shear_reduced_slab(film_slab)
            if reorient:
                film_slab = align_x(film_slab)
            film_slab.sort()
            film_slabs[i] = film_slab

        self.film_structures = film_slabs

        # Apply transformation to produce matched area and a & b vectors
        self.apply_transformations(match)

        #Nico: I take out all the sym stuff, not interesting for us, leads to problems
        # Get non-stoichioimetric substrate slabs
        if do_sym_mod:
            sym_sub_slabs = []
            for sub_slab in self.modified_substrate_structures:
                sym_sub_slab = self.sub_sg.nonstoichiometric_symmetrized_slab(sub_slab)
                for slab in sym_sub_slab:
                    if not slab == sub_slab:
                        sym_sub_slabs.append(slab)

            self.sym_modified_substrate_structures = sym_sub_slabs

            # Get non-stoichioimetric film slabs
            sym_film_slabs = []
            for film_slab in self.modified_film_structures:
                sym_film_slab = self.film_sg.nonstoichiometric_symmetrized_slab(film_slab)
                for slab in sym_film_slab:
                    if not slab == film_slab:
                        sym_film_slabs.append(slab)

            self.sym_modified_film_structures = sym_film_slabs

        # Strained film structures (No Vacuum)
        self.strained_substrate, self.strained_film = strain_slabs(self.oriented_substrate, self.oriented_film)

        return

    def apply_transformation(self, structure, matrix):
        """
        Make a supercell of structure using matrix

        Args:
            structure (Slab): Slab to make supercell of
            matrix (3x3 np.ndarray): supercell matrix

        Returns:
            (Slab) The supercell of structure
        """
        modified_substrate_structure = structure.copy()
        # Apply scaling
        modified_substrate_structure.make_supercell(matrix)

        # Reduce vectors
        new_lattice = modified_substrate_structure.lattice.matrix.copy()
        new_lattice[:2, :] = reduce_vectors(*modified_substrate_structure.lattice.matrix[:2, :])
        modified_substrate_structure = Slab(lattice=Lattice(new_lattice), species=modified_substrate_structure.species,
                                            coords=modified_substrate_structure.cart_coords,
                                            miller_index=modified_substrate_structure.miller_index,
                                            oriented_unit_cell=modified_substrate_structure.oriented_unit_cell,
                                            shift=modified_substrate_structure.shift,
                                            scale_factor=modified_substrate_structure.scale_factor,
                                            coords_are_cartesian=True, energy=modified_substrate_structure.energy,
                                            reorient_lattice=modified_substrate_structure.reorient_lattice,
                                            to_unit_cell=True)

        return modified_substrate_structure
    

    def apply_transformations(self, match):
        """
        Using ZSL match, transform all of the film_structures by the ZSL
        supercell transformation.

        Args:
            match (dict): ZSL match returned by ZSLGenerator.__call__
        """
        film_transformation = match["film_transformation"]
        sub_transformation = match["substrate_transformation"]

        modified_substrate_structures = [struct.copy() for struct in self.substrate_structures]
        modified_film_structures = [struct.copy() for struct in self.film_structures]

        # Match angles in lattices with 𝛾=θ° and 𝛾=(180-θ)°
        if np.isclose(180 - modified_film_structures[0].lattice.gamma, modified_substrate_structures[0].lattice.gamma,
                      atol=3):
            reflection = SymmOp.from_rotation_and_translation(((-1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 1))
            for modified_film_structure in modified_film_structures:
                modified_film_structure.apply_operation(reflection, fractional=True)
            self.oriented_film.apply_operation(reflection, fractional=True)

        sub_scaling = np.diag(np.diag(sub_transformation))

        # Turn into 3x3 Arrays
        sub_scaling = np.diag(np.append(np.diag(sub_scaling), 1))
        temp_matrix = np.diag([1, 1, 1])
        temp_matrix[:2, :2] = sub_transformation

        for modified_substrate_structure in modified_substrate_structures:
            modified_substrate_structure = self.apply_transformation(modified_substrate_structure, temp_matrix)
            self.modified_substrate_structures.append(modified_substrate_structure)

        self.oriented_substrate = self.apply_transformation(self.oriented_substrate, temp_matrix)

        film_scaling = np.diag(np.diag(film_transformation))

        # Turn into 3x3 Arrays
        film_scaling = np.diag(np.append(np.diag(film_scaling), 1))
        temp_matrix = np.diag([1, 1, 1])
        temp_matrix[:2, :2] = film_transformation

        for modified_film_structure in modified_film_structures:
            modified_film_structure = self.apply_transformation(modified_film_structure, temp_matrix)
            self.modified_film_structures.append(modified_film_structure)

        self.oriented_film = self.apply_transformation(self.oriented_film, temp_matrix)

        return


    #Nico: adapted
    def combine_slabs_nico(self, offset = None, polar = None):
        """
        Combine the slabs generated by get_oriented_slabs into interfaces
        """

        all_substrate_variants = []
        sub_labels = []
        for i, slab in enumerate([self.modified_substrate_structures[0]]):
            all_substrate_variants.append(slab)
            sub_labels.append(str(i))



        all_film_variants = []
        film_labels = []
        for i, slab in enumerate(self.modified_film_structures):
            all_film_variants.append(slab)
            film_labels.append(str(i))
            sg = SpacegroupAnalyzer(slab, symprec=1e-3)
            # Nico: for water systems that is too strict and 
            # and the in plane symmetry is irrelevant. therefore change to externally determined
            # polarity
            #if not sg.is_laue():
            if polar == None:
                mirrortest = any([np.linalg.norm(a.rotation_matrix[2] - np.array([0,0,-1])) < 0.0001 for a in sg.get_space_group_operations()])
                if not any(mirrortest):
                    mirrored_slab = slab.copy()
                    reflection_z = SymmOp.from_rotation_and_translation(((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))
                    mirrored_slab.apply_operation(reflection_z, fractional=True)
                    translation = [0, 0, -min(mirrored_slab.frac_coords[:, 2])]
                    mirrored_slab.translate_sites(range(mirrored_slab.num_sites), translation)
                    all_film_variants.append(mirrored_slab)
                    film_labels.append('%dm'%i)
            elif polar:
                mirrored_slab = slab.copy()
                reflection_z = SymmOp.from_rotation_and_translation(((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))
                mirrored_slab.apply_operation(reflection_z, fractional=True)
                translation = [0, 0, -min(mirrored_slab.frac_coords[:, 2])]
                mirrored_slab.translate_sites(range(mirrored_slab.num_sites), translation)
                all_film_variants.append(mirrored_slab)
                film_labels.append('%dm'%i)



        # substrate first index, film second index
        self.interfaces = []
        self.interface_labels = []
        # self.interfaces = [[None for j in range(len(all_film_variants))] for i in range(len(all_substrate_variants))]
        for i, substrate in enumerate(all_substrate_variants):
            for j, film in enumerate(all_film_variants):
                self.interfaces.append(self.make_interface(substrate, film, offset=offset))
                self.interface_labels.append('%s/%s' % (film_labels[j], sub_labels[i]))







    #Nico: Offset
    def combine_slabs(self, offset=None):
        """
        Combine the slabs generated by get_oriented_slabs into interfaces
        """

        all_substrate_variants = []
        sub_labels = []
        for i, slab in enumerate(self.modified_substrate_structures):
            all_substrate_variants.append(slab)
            sub_labels.append(str(i))
            sg = SpacegroupAnalyzer(slab, symprec=1e-3)
            if not sg.is_laue():
                mirrored_slab = slab.copy()
                reflection_z = SymmOp.from_rotation_and_translation(((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))
                mirrored_slab.apply_operation(reflection_z, fractional=True)
                translation = [0, 0, -min(mirrored_slab.frac_coords[:, 2])]
                mirrored_slab.translate_sites(range(mirrored_slab.num_sites), translation)
                all_substrate_variants.append(mirrored_slab)
                sub_labels.append('%dm' % i)

        all_film_variants = []
        film_labels = []
        for i, slab in enumerate(self.modified_film_structures):
            all_film_variants.append(slab)
            film_labels.append(str(i))
            sg = SpacegroupAnalyzer(slab, symprec=1e-3)
            if not sg.is_laue():
                mirrored_slab = slab.copy()
                reflection_z = SymmOp.from_rotation_and_translation(((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 0))
                mirrored_slab.apply_operation(reflection_z, fractional=True)
                translation = [0, 0, -min(mirrored_slab.frac_coords[:, 2])]
                mirrored_slab.translate_sites(range(mirrored_slab.num_sites), translation)
                all_film_variants.append(mirrored_slab)
                film_labels.append('%dm' % i)

        # substrate first index, film second index
        self.interfaces = []
        self.interface_labels = []
        # self.interfaces = [[None for j in range(len(all_film_variants))] for i in range(len(all_substrate_variants))]
        for i, substrate in enumerate(all_substrate_variants):
            for j, film in enumerate(all_film_variants):
                #Nico offset
                self.interfaces.append(self.make_interface(substrate, film,offset = offset))
                self.interface_labels.append('%s/%s' % (film_labels[j], sub_labels[i]))

    def make_interface(self, slab_substrate, slab_film, offset=None):
        """
        Strain a film to fit a substrate and generate an interface.

        Args:
            slab_substrate (Slab): substrate structure supercell
            slab_film (Slab): film structure supercell
            offset ([int]): separation vector of film and substrate
        """

        # Check if lattices are equal. If not, strain them to match
        # NOTE: CHANGED THIS TO MAKE COPY OF SUBSTRATE/FILM, self.modified_film_structures NO LONGER STRAINED
        unstrained_slab_substrate = slab_substrate.copy()
        slab_substrate = slab_substrate.copy()
        unstrained_slab_film = slab_film.copy()
        slab_film = slab_film.copy()
        latt_1 = slab_substrate.lattice.matrix.copy()
        latt_1[2, :] = [0, 0, 1]
        latt_2 = slab_film.lattice.matrix.copy()
        latt_2[2, :] = [0, 0, 1]
        if not Lattice(latt_1) == Lattice(latt_2):
            # Calculate lattice strained to match:
            matched_slab_substrate, matched_slab_film = strain_slabs(slab_substrate, slab_film)
        else:
            matched_slab_substrate = slab_substrate
            matched_slab_film = slab_film

        # Ensure substrate has positive c-direction:
        if matched_slab_substrate.lattice.matrix[2, 2] < 0:
            latt = matched_slab_substrate.lattice.matrix.copy()
            latt[2, 2] *= -1
            new_struct = matched_slab_substrate.copy()
            new_struct.lattice = Lattice(latt)
            matched_slab_substrate = new_struct

        # Ensure film has positive c-direction:
        if matched_slab_film.lattice.matrix[2, 2] < 0:
            latt = matched_slab_film.lattice.matrix.copy()
            latt[2, 2] *= -1
            new_struct = matched_slab_film.copy()
            new_struct.lattice = Lattice(latt)
            matched_slab_film = new_struct

        if offset is None:
            offset = (2.5, 0.0, 0.0)

        _structure = merge_slabs(matched_slab_substrate, matched_slab_film, *offset)
        
        orthogonal_structure = _structure.get_orthogonal_c_slab()
        #Nico: WTF why would you sort. BS
        #orthogonal_structure.sort()

        if not orthogonal_structure.is_valid(tol=1):
            warnings.warn("Check generated structure, it may contain atoms too closely placed")

        # offset_vector = (offset[1], offset[2], offset[0])
        interface = Interface(orthogonal_structure.lattice.copy(), orthogonal_structure.species,
                              orthogonal_structure.frac_coords,
                              slab_substrate.miller_index, slab_film.miller_index,
                              self.original_substrate_structure, self.original_film_structure,
                              unstrained_slab_substrate, unstrained_slab_film,
                              slab_substrate, slab_film, init_inplane_shift=offset[1:],
                              site_properties=orthogonal_structure.site_properties)

        return interface


def strain_slabs(sub_slab, film_slab):
    """
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
    new_latt = Lattice(np.dot(film_struct.lattice.matrix, rotation))
    film_struct.lattice = new_latt

    # Average the two lattices (Should get equal strain?)
    # Nico: No!! should not get equal strain!

    mean_a = np.mean([film_struct.lattice.matrix[0, :], sub_struct.lattice.matrix[0, :]], axis=0)
    mean_b = np.mean([film_struct.lattice.matrix[1, :], sub_struct.lattice.matrix[1, :]], axis=0)
    #Nico: I changed this so it works out
    #Nico: Ideally one could adjust the zaxis of the filmstruct so that the energy/stress is minimized !!! if you want to optimize
    
    #new_latt = np.vstack([mean_a, mean_b, sub_struct.lattice.matrix[2, :]])
    new_latt = sub_struct.lattice.matrix
    sub_struct.lattice = Lattice(new_latt)
    #Nico
    #new_latt = np.vstack([mean_a, mean_b, film_struct.lattice.matrix[2, :]])
    #film_struct.lattice = Lattice(new_latt)
    new_latt = np.vstack([sub_struct.lattice.matrix[0, :], sub_struct.lattice.matrix[1, :], film_struct.lattice.matrix[2, :]])
    film_struct.lattice = Lattice(new_latt)



    return sub_struct, film_struct


def get_shear_reduced_slab(slab):
    """
    Reduce the vectors of the slab plane according to the algorithm in
    substrate_analyzer, then make a new Slab with a Lattice with those
    reduced vectors.

    Args:
        slab (Slab): Slab to reduce

    Returns:
        Slab object of identical structure to the input slab
        but rduced in-plane lattice vectors
    """
    reduced_vectors = reduce_vectors(
        slab.lattice.matrix[0],
        slab.lattice.matrix[1])
    new_lattice = Lattice([reduced_vectors[0], reduced_vectors[1], slab.lattice.matrix[2]])
    return Slab(lattice=new_lattice, species=slab.species,
                coords=slab.cart_coords,
                miller_index=slab.miller_index,
                oriented_unit_cell=slab.oriented_unit_cell,
                shift=slab.shift,
                scale_factor=slab.scale_factor,
                coords_are_cartesian=True, energy=slab.energy,
                reorient_lattice=slab.reorient_lattice,
                to_unit_cell=True)
