import numpy as np


class SpatialEnvironment:
    """Represents the spatial environment of a given particle in a structure

    Spatial environment objects characterize the surroundings of a particle
    based on the distances and types of neighboring particles. They are used
    to suggest ``Reduction`` instances which satisfy equivalences.

    """

    def __init__(self, index, size, indices, distances, cluster_types):
        """Constructor

        Parameters
        ----------

        index : int
            index of particle for which environment is constructed

        size : float
            cutoff distance of neighbors that are included in the environment.

        indices : 1darray, np.int32
            indices of all surrounding particles

        distances : 1darray, np.float64
            distances of all surrounding particles

        cluster_types : list of tuples
            list of cluster types for all surrounding atoms

        """
        assert len(indices) == len(distances)
        assert len(indices) == len(cluster_types)

        self.index = index
        self.size  = size
        self.indices = indices
        self.distances = distances
        self.cluster_types = cluster_types

    def is_similar_to(self, env, cutoff=None, tol=1e-1):
        """Evaluates equivalence of environments up to certain cutoff

        Environments are considered equivalent if they contain the same
        cluster types and if the corresponding distances are near each other.
        Equivalence is typically not checked for the whole environment but
        only for a certain cutoff. This is because clusters near the edge
        of the size of the environments may or may not be included and therefore
        would cause dissimilar behavior, even though the difference in distance
        may be within the tolerance.

        Parameters
        ----------

        env : ``SpatialEnvironment`` instance
            spatial environment for which to check equivalence

        """
        if cutoff is None: # default cutoff significantly below size of env 
            cutoff = self.size - (3 * tol)
        if (self.size < cutoff) or (env.size < cutoff):
            raise ValueError('Equality with cutoff {} cannot be checked for'
                    'environments with sizes {}, {}'.format(
                        cutoff,
                        self.size,
                        env.size,
                        ))

        # to avoid accidental misses near the boundary, the cutoff is enlarged
        # until no cluster has a distance in the interval cutoff +/- tol.
        _all = np.concatenate((
            self.distances,
            env.distances,
            ))
        _all.sort()
        while (np.min(np.abs(_all - cutoff)) < tol):
            cutoff += tol

        # embed type information in the distance by adding an integer number of
        # times the largest size of both environments, such that clusters
        # of different types are completely separated in space and a pure
        # distance comparison suffices
        _all_types = list(set(self.cluster_types + env.cluster_types))
        step = max(self.size, env.size)

        embedding0 = self.distances.copy()
        cutoff0 = cutoff * np.ones(embedding0.shape)
        for i in range(len(self.cluster_types)):
            multiplier = _all_types.index(self.cluster_types[i])
            embedding0[i] += multiplier * step
            cutoff0[i]    += multiplier * step
        mask0 = embedding0 < cutoff0

        embedding1 = env.distances.copy()
        cutoff1 = cutoff * np.ones(embedding1.shape)
        for i in range(len(env.cluster_types)):
            multiplier = _all_types.index(env.cluster_types[i])
            embedding1[i] += multiplier * step
            cutoff1[i]    += multiplier * step
        mask1 = embedding1 < cutoff1

        assert len(mask0) == len(mask1)
        return np.allclose(embedding0[mask0], embedding1[mask1], atol=tol, rtol=0)

    @staticmethod
    def check_compatibility(*args, cutoff, tol=1e-1):
        pass
