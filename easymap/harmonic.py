import numpy as np


class Harmonic:
    """Represents a second-order atomic structure."""

    def __init__(self, atoms, hessian):
        """Constructor

        Parameters
        ----------

        atoms : ase.Atoms
            atoms instance of the atomic structure that contains the optimized
            geometry and atomic element information.

        hessian : 2darray
            second-order derivatives of the PES of the structure with respect
            to the atomic coordinates. Should be a real, symmetric matrix.

        """
        self.atoms    = atoms
        self.periodic = np.prod(atoms.pbc)
        self.ndof     = 3 * len(atoms)

        assert hessian.shape == (self.ndof, self.ndof)
        assert np.allclose(hessian, hessian.T)
        self.hessian = hessian
        self.frequencies = None
        self.modes = None

    def compute_eigenmodes(self):
        """Computes and stores the eigenmodes of the system

        The eigenmodes are computed as the eigenvectors of the mass-weighted
        hessian, whose eigenvalues are the corresponding eigenfrequencies.

        """
        masses = np.repeat(self.atoms.get_masses(), 3)
        mass_matrix = 1 / np.sqrt(np.outer(masses, masses))
        hessian_mw = mass_matrix * self.hessian
        self.frequencies, self.modes = np.linalg.eigh(hessian_mw)
        return self.frequencies, self.modes
