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

        self.similarities = None

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

    def add_similarities(self, similarities):
        """Adds similarity relations to atoms in the system

        Two or more atoms may be defined as similar in order to enforce
        subsequent dimensionality reduction procedures to treat them in
        exactly the same manner. More specific, if atom a and atom b are
        similar, then they must be clustered in exactly the same manner (i.e.
        they must belong to the same cluster type).

        Parameters
        ----------

        similarities : 2darray of shape (natom, natom)
            integer array (with values 0 or 1) that defines how atoms are
            partitioned into similarity classes. Atom j is contained in
            similarity class i iff similarities[i, j] == 1. If the identity
            matrix is given, then all atoms belong to their own similarity
            class (i.e. no two atoms are similar).

        """
        pass
