import numpy as np
from pathlib import Path

import ase.io

from easymap import Harmonic, Mapping


# specifying absolute path ensures tests may be executed from any directory
here = Path(__file__).parent


def uio66_ff():
    """Return force field hessian and optimized structure"""
    path_system = here / 'uio66'
    atoms     = ase.io.read(str(path_system / 'conventional.cif'))
    hessian   = np.load(path_system / 'ff' / 'hessian.npy')
    positions = np.load(path_system / 'ff' / 'positions.npy')
    cell      = np.load(path_system / 'ff' / 'cell.npy')

    atom_types =  24 * [0] # Zr
    atom_types += 16 * [1] # O_OX
    atom_types += 16 * [2] # O_OH
    atom_types += 96 * [3] # O_CA
    atom_types += 48 * [4] # C_CA
    atom_types += 48 * [5] # C_PC
    atom_types += 96 * [6] # C_PH
    atom_types += 16 * [7] # H_OH
    atom_types += 96 * [8] # H_PH
    atom_types = np.array(atom_types)
    equivalencies = np.zeros((456, 456), dtype=np.int32)
    equivalencies[atom_types, np.arange(456)] = 1

    atoms.set_positions(positions)
    atoms.set_cell(cell)
    return atoms, hessian, equivalencies


def get_harmonic(name):
    if name == 'uio66_ff':
        atoms, hessian, equivalencies = uio66_ff()
        #mapping = Mapping(atoms.get_masses(), equivalencies)
        return Harmonic(atoms, hessian), equivalencies
    else:
        raise ValueError('')
