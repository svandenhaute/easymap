import numpy as np

from easymap.utils import apply_mic, determine_rcut


class Mapping:
    """Class to represent a center-of-mass mapping of atoms into clusters"""

    def __init__(self, masses):
        """Constructor

        Parameters
        ----------

        masses : 1darray of shape (natom,) [ASE mass unit]
            contains masses of individual atoms. This is necessary to compute
            the positions of clusters based on the atomic positions
            (and box vectors).

        """
        self.masses    = masses.copy()
        self.natoms    = len(masses)
        self.clusters  = None
        self.nclusters = None
        self.deltas    = None
        self.transform = None
        self.update_clusters(np.eye(len(masses), dtype=np.int32))

    def apply(self, positions, rvecs=None):
        """Computes cluster centers of mass

        For periodic systems, clusters are not allowed to contain relative
        vectors which are longer than half of the smallest diagonal box vector
        component (in its reduced representation). A ValueError is raised if
        this is the case.

        Parameters
        ----------

        positions : 2darray of shape (natom, 3) [angstrom]
            atomic positions.

        rvecs : 2darray of shape (3, 3) [angstrom] or None
            box vectors of the configuration, if applicable.

        """
        dvecs = self.deltas @ positions
        if rvecs is not None: # apply mic to dvecs
            mic(dvecs, rvecs)
            rcut = determine_rcut(rvecs)
            assert np.all(np.linalg.norm(dvecs, axis=1) < rcut)
        positions_com = np.zeros((self.nclusters, 3))
        for i, group in enumerate(self):
            positions_com[i, :] = positions[group[0], :].copy()
        positions_com += self.transform @ dvecs
        return positions_com

    def update_clusters(self, clusters):
        """Updates the clusters configuration of the mapping

        Parameters
        ----------

        clusters : 2darray of shape (natoms, natoms)
            integer array (i.e. with values 0 or 1) that defines how atoms are
            partitioned into clusters or beads. Atom j is contained in
            bead i iff clusters[i, j] == 1. The dtype is expected to be
            np.int32.

        """
        assert np.all((clusters == 1) + (clusters == 0))
        assert clusters.shape == (self.natoms,) * 2
        assert np.allclose(np.ones(self.natoms), np.sum(clusters, axis=0))
        self.clusters  = clusters.copy()
        self.nclusters = np.sum(np.sum(clusters, axis=1) > 0)
        self.deltas    = np.zeros((self.natoms, self.natoms), dtype=np.int32)
        self.transform = np.zeros((self.nclusters, self.natoms))
        count = 0
        for i, group in enumerate(self):
            mass = np.sum(self.masses[group])
            for j, atom in enumerate(group): # add delta for each atom
                self.deltas[count, group[0]] -= 1
                self.deltas[count, atom] += 1
                self.transform[i, count] = self.masses[atom] / mass
                count += 1
        assert count == self.natoms

    def __iter__(self):
        for i in range(self.nclusters):
            yield np.where(self.clusters[i, :] == 1)[0]
