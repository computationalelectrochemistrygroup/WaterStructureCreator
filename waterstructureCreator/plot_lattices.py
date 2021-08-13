import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
from matplotlib.cm import jet
from pymatgen.analysis.waterstructureCreator.substrate_analyzer_mod import reduce_vectors

def plot_lattice_small_2d(ax, reccell, color="b", lw=0.8, ls='-'):
    """Short summary.

    Parameters
    ----------
    ax : object
        for plotting
    reccell : array
        array 2x2
    color : str
    lw : float
    ls : str

    Returns
    -------
    type
        Description of returned object.

    """
    r = [0, 1]
    for s, e in combinations(np.array(list(product(r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            snew = np.dot(np.array(s), reccell)
            enew = np.dot(np.array(e), reccell)
            ax.plot(*zip(snew, enew), color=color, lw=lw, ls=ls)

def plot_matrices(ax, matrices, cols, optmatrix):
    """Short summary.

    Parameters
    ----------
    ax : object
        for plotting
    matrices : list
        list of 2x2 possible matrices for a surface with a given number of metal atoms.
    cols : array
        give the color of the cmap
    optmatrix : array
        2x2, optimal reduced matrix for a surface with a given number of metal atoms.

    Returns
    -------
    type
        plot of all possible supercells for a surface with a given number of metal atoms.

    """
    for ini , matrix in enumerate(matrices):
        mm = np.array(matrix)[:2,:2]
        plot_lattice_small_2d( ax, reduce_vectors(mm[0], mm[1]), color = jet(cols[ini]))


    om = np.array(optmatrix)[:2,:2]
    plot_lattice_small_2d( ax, reduce_vectors(om[0], om[1]  ), color = 'k', lw = 3)
    ax.set_aspect('equal', 'datalim')
    return
