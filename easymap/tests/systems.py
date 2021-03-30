import numpy as np
from pathlib import Path

import ase.io

from easymap import Harmonic


# specifying absolute path ensures tests may be executed from any directory
here = Path(__file__).parent


def uio66_ff():
    """Return force field hessian and optimized structure"""
    path_system = here / 'uio66'
    atoms     = ase.io.read(str(path_system / 'conventional.cif'))
    hessian   = np.load(path_system / 'ff' / 'hessian.npy')
    positions = np.load(path_system / 'ff' / 'positions.npy')
    cell      = np.load(path_system / 'ff' / 'cell.npy')

    atoms.set_positions(positions)
    atoms.set_cell(cell)
    return atoms, hessian


def get_harmonic(name):
    if name == 'uio66_ff':
        atoms, hessian = uio66_ff()
        return Harmonic(atoms, hessian)
