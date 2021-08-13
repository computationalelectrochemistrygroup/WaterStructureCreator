import json
from pymatgen.io.vasp import Poscar

def import_bulk_structures(filename):
    print('public version')
    """Read bulk structures from file and return a dictionary of it.

    Parameters
    ----------
    filename : str
        Filename of the file containing the bulk structures

    Returns
    -------
    filedict : dict
        Dictionary of the structures

    """
    filedict = {}
    with open(filename, "r") as o:
        data = json.load(o)
    for i, v  in data.items():
        struct = Poscar.from_file("./"+v[1]).structure
        filedict[i] = struct
    return filedict


def import_slab_structures(filename):
    """Read 2D water structures from file and return a dictionary of it.

    Parameters
    ----------
    filename : str
        Filename of the file containing the bulk structures

    Returns
    -------
    filedict : dict
        Dictionary of the structures

    """
    filedict = {}
    with open(filename, "r") as o:
        data = json.load(o)
    for i, items in data.items():
        filedict[i] = []
        for v in items:
            struct = Poscar.from_file("./"+v[-1]).structure
            filedict[i].append((v, struct))

    return filedict