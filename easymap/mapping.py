import numpy as np

from easymap.utils import apply_mic, determine_rcut


class Mapping:
    """Class to represent a center-of-mass mapping of atoms into clusters

    The clustering of the atoms is encoded in the clusters attribute.

    Two or more atoms inside a mapping may be defined as equivalent in order
    to enforce subsequent dimensionality reduction procedures to treat
    them in exactly the same manner.
    More specific, if atom a and atom b are equivalent, then they must be
    clustered in exactly the same manner (i.e. they must belong to the
    same cluster type). Atom equivalency may arise as a result of e.g.:

        1. symmetry considerations (e.g. symmetry-constrained hessians)
        2. atom-typing (as done in force fields)

    """

    def __init__(self, masses, equivalency=None):
        """Constructor

        Parameters
        ----------

        masses : 1darray of shape (natoms,) [ASE mass unit]
            contains masses of individual atoms. This is necessary to compute
            the positions of clusters based on the atomic positions
            (and box vectors).

        equivalency : 2darray of shape (natoms, natoms)
            integer array (with values 0 or 1) that defines how atoms are
            partitioned into equivalence classes. Atom j is contained in
            equivalency class i iff equivalency[i, j] == 1. If the identity
            matrix is given, then all atoms belong to their own equivalency
            class (i.e. no two atoms are equivalent).
            dtype is expected to be np.int32

        """
        self.masses    = masses.copy()
        self.natoms    = len(masses)
        self.clusters  = None
        self.nclusters = None
        self.deltas    = None
        self.transform = None
        self.update_clusters(np.eye(len(masses), dtype=np.int32))

        #self.equivalency = equivalency
        self.atom_types    = None
        self.cluster_types = None
        self.infer_atom_types(equivalency)
        self.update_identities() # update atom_types and cluster_types

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

    def update_identities(self, validate=True):
        """Derives cluster identities based on atom equivalencies

        Parameters
        ----------

        validate : bool
            if True, this verifies that the cluster assignment satisfies
            the specified atom equivalencies (equivalent atoms must belong to
            equivalent groups).

        """
        self.cluster_types = {}
        for i, group in enumerate(self):
            types = self.atom_types[np.array(group)]
            key = tuple(np.sort(types))
            self.cluster_types[i] = key

        if validate:
            atom_type_to_cluster_type = {}
            for i in range(self.natoms):
                index = np.where(self.clusters[:, i] == 1)[0][0]
                atom_type = self.atom_types[i]
                if atom_type in atom_type_to_cluster_type.keys():
                    assert self.cluster_types[index] == atom_type_to_cluster_type[atom_type]
                else:
                    atom_type_to_cluster_type[atom_type] = self.cluster_types[index]

    def infer_atom_types(self, equivalency):
        """Defines atom types based on an equivalency matrix

        Parameters
        ----------

        equivalency : 2darray of shape (natoms, natoms)
            integer array (with values 0 or 1) that defines how atoms are
            partitioned into equivalence classes. Atom j is contained in

        """
        if equivalency is not None:
            assert np.all((equivalency == 1) + (equivalency == 0))
            assert equivalency.shape == (self.natoms,) * 2
            assert np.allclose(np.ones(self.natoms), np.sum(equivalency, axis=0))

            self.atom_types = np.zeros(self.natoms, dtype=np.int32)
            count = 0
            for i in range(self.natoms):
                indices = np.where(equivalency[i, :] > 0)[0]
                for index in indices:
                    self.atom_types[index] = count
                count += 1
        else:
            self.atom_types = np.arange(self.natoms, dtype=np.int32)

    def __iter__(self):
        for i in range(self.nclusters):
            yield np.where(self.clusters[i, :] == 1)[0]
