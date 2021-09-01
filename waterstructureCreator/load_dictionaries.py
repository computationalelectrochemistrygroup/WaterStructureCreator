import json
from pymatgen.io.vasp import Poscar
import pandas as pd

def import_bulk_structures(filename):
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
        #struct = Poscar.from_file("./"+v[1]).structure
        struct = Poscar.from_file(('/').join(filename.split('/')[:-1]) + '/' + v[1]).structure
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
            #struct = Poscar.from_file("./"+v[-1]).structure
            struct = Poscar.from_file(('/').join(filename.split('/')[:-1]) + '/' + v[-1]).structure

            filedict[i].append((v, struct))

    return filedict


def import_bulk_structures_pd(filename):
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
    df = pd.read_csv(filename)
    df['struct'] = ''
    
    for index, row in df.iterrows():
        df.at[index, 'struct'] = Poscar.from_file(('/').join(filename.split('/')[:-1]) + '/' + row['Filename'])
    return df


def import_slab_structures_pd(filename):
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
    df = pd.read_csv(filename)
    df['Cell Length'] = [json.loads(a) for a in df['Cell Length'].tolist()]
    df['struct'] = ''
    
    for index, row in df.iterrows():
        df.at[index, 'struct'] = Poscar.from_file(('/').join(filename.split('/')[:-1]) + '/' + row['Filename'])
    return df
