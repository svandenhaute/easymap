import numpy as np

from easymap.utils import apply_mic, determine_rcut, expand_xyz, get_entropy


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

        self.atom_types    = None
        self.cluster_types = None
        self.infer_atom_types(equivalency)
        self.update_identities() # update atom_types and cluster_types

    def apply(self, positions, rvecs=None):
        """Computes cluster centers of mass

        For periodic systems, clusters are not allowed to contain relative
        vectors which are longer than half of the smallest diagonal box vector
        component in its reduced representation (i.e. a_x, b_y, or c_z).
        A ValueError is raised if this is the case.

        Parameters
        ----------

        positions : 2darray of shape (natom, 3) [angstrom]
            atomic positions.

        rvecs : 2darray of shape (3, 3) [angstrom] or None
            box vectors of the configuration, if applicable.

        """
        dvecs = self.deltas @ positions
        if rvecs is not None: # apply mic to dvecs
            apply_mic(dvecs, rvecs)
            rcut = determine_rcut(rvecs)
            if not np.all(np.linalg.norm(dvecs, axis=1) < rcut):
                raise ValueError('Maximum allowed size of distance vectors'
                        ' was exceeded (rcut = {}).'.format(rcut))
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

        A boolean is returned which specifies whether the current clustering
        preserves the equivalence of atoms.

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

        equiv = True # whether equivalence is satisfied
        atom_type_to_cluster_type = {}
        for i in range(self.natoms):
            index = np.where(self.clusters[:, i] == 1)[0][0]
            atom_type = self.atom_types[i]
            if atom_type in atom_type_to_cluster_type.keys():
                equiv = equiv and (self.cluster_types[index] == atom_type_to_cluster_type[atom_type])
            else:
                atom_type_to_cluster_type[atom_type] = self.cluster_types[index]
        return equiv

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

    @staticmethod
    def merge(clusters, groups):
        """Merges groups of clusters

        Parameters
        ----------

        clusters : 2darray of shape (natoms, natoms)
            integer array (i.e. with values 0 or 1) that defines how atoms are
            partitioned into clusters or beads.

        groups : list of tuples
            list of tuples. Each tuple contains the cluster indices that should
            be merged into a single cluster.

        """
        merger = np.eye(clusters.shape[1], dtype=np.int32)
        for group in groups:
            for index in group[1:]:
                merger[group[0], index] = 1 # add to first cluster
                merger[index, index]    = 0 # remove existing
        new_clusters = merger @ clusters
        # sort clusters such that zero rows are at the bottom of the array
        indices = np.argsort(np.sum(new_clusters, axis=1))
        return new_clusters[indices[::-1]]

    def __iter__(self):
        for i in range(self.nclusters):
            yield np.where(self.clusters[i, :] == 1)[0]

    def copy(self):
        """Copies current instance"""
        mapping = Mapping(self.masses)
        mapping.update_clusters(self.clusters)
        mapping.atom_types = self.atom_types.copy()
        mapping.update_identities(validate=True)
        return mapping

    def write(self, filename=None):
        """Saves all information of a mapping object to a .npz archive"""
        # prepare masses, equivalencies, and current clusters arrays
        masses = self.masses
        equivalencies = np.zeros((self.natoms, self.natoms), dtype=np.int32)
        for i in range(self.natoms):
            equivalencies[self.atom_types[i], i] = 1
        clusters = self.clusters
        if filename is not None:
            np.savez(
                    filename,
                    masses=masses,
                    equivalencies=equivalencies,
                    clusters=clusters,
                    )
        return masses, equivalencies, clusters

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        mapping = cls(data['masses'], data['equivalencies'])
        mapping.update_clusters(data['clusters'])
        mapping.update_identities()
        return mapping

    def check_mic(self, positions, rvecs, translate_atoms=True):
        """Verifies whether clusters are sufficiently small as to apply the mic

        For each cluster, the largest relative vector between member atoms is
        determined and compared with the allowed cutoff of the unit cell.

        Parameters
        ----------

        positions : 2darray of shape (natom, 3) [angstrom]
            atomic positions.

        rvecs : 2darray of shape (3, 3) [angstrom]
            box vectors of the configuration

        translate_atoms : bool
            determines whether to translate atoms such that individual
            clusters do not require the box vectors for the computation of their
            center of mass.

        """
        rcut = determine_rcut(rvecs)
        for group in self:
            n = len(group)
            if n > 1:
                r = group[0] # take random central atom as reference
                deltas = np.zeros((n, self.natoms))
                for i in range(n):
                    deltas[i, r] -= 1
                    deltas[i, group[i]] += 1 # becomes zero for index == r
                dvecs = deltas @ positions
                apply_mic(dvecs, rvecs)
                group_pos = np.zeros((n, 3)) # construct pos with dvecs
                group_pos = positions[r, :].reshape((1, 3)) + dvecs
                if translate_atoms:
                    positions[group, :] = group_pos[:]
                # iterate over all relative vectors
                for k in range(n):
                    for l in range(n):
                        d = np.linalg.norm(group_pos[k, :] - group_pos[l, :])
                        if d > rcut:
                            return False
        return True

    def sort_by_cluster_type(self):
        """Sorts cluster definitions by type"""
        sorting = sorted(range(self.nclusters), key=lambda x: self.cluster_types[x])


def score(mapping, hessian_mw=None, harmonic=None, temperature=300):
    """Scores a list of reductions based on a mapping and a harmonic"""
    mapping_matrix = np.zeros((mapping.nclusters, mapping.natoms))
    for i, group in enumerate(mapping):
        mapping_matrix[i, group] = mapping.masses[group]
    mapping_matrix /= np.sum(mapping_matrix, axis=1, keepdims=True)
    mapping_matrix = np.sqrt(mapping_matrix)

    if harmonic is not None: # compute mass-weighted hessian
        masses = np.repeat(harmonic.atoms.get_masses(), 3)
        mass_matrix = 1 / np.sqrt(np.outer(masses, masses))
        hessian_mw = mass_matrix * harmonic.hessian
    else: # mass-weighted hessian already computed
        assert hessian_mw is not None

    # apply SVD to generate orthonormal basis in null space of M
    _, sigmas, VT = np.linalg.svd(mapping_matrix)
    assert np.allclose(sigmas, np.ones(len(sigmas)))
    basis = np.transpose(expand_xyz(VT[mapping.nclusters:, :]))

    # obtain hessian in null space of mapping and compute eigenvalues
    hessian_null = basis.T @ hessian_mw @ basis
    omegas = np.linalg.eigvals(hessian_null)
    frequencies = np.sqrt(omegas[omegas > 0]) / (2 * np.pi)
    entropy, _ = get_entropy(
            frequencies,
            temperature,
            use_quantum=True,
            remove_zero=True,
            )
    smap = np.sum(entropy)
    return smap
