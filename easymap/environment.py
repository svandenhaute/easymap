import logging
import numpy as np
import networkx as nx

from easymap.utils import get_distances, get_nlist, apply_mic


logger = logging.getLogger(__name__) # logging per module


class ClusterEnvironment:
    """Represents the spatial environment of a given particle in a structure

    Spatial environment objects characterize the surroundings of a particle
    based on the distances and types of neighboring particles. Their main use
    is to be able to define similarity classes between environments. Internally,
    they implement a networkX graph object with particles as nodes and
    distances as edge weights.

    """

    def __init__(self, positions, cluster_types, indices):
        """Constructor

        Parameters
        ----------

        positions : 2darray [angstrom]
            contains cartesian coordinates of all particles in the environment.
            The first particle is considered to be the central particle.
            These positions should satisfy the minimum image convention in case
            of a periodic system.

        indices : 1darray
            array of cluster indices (referring to the clusters in a ``Mapping``
            instance).

        """
        nparticles = positions.shape[0]
        assert nparticles == len(cluster_types)
        assert nparticles == len(indices)

        self.positions     = positions
        self.cluster_types = cluster_types
        self.indices       = indices
        self.graph   = nx.Graph()
        for i in range(nparticles):
            self.graph.add_node(i, cluster_type=cluster_types[i])

        for i in range(nparticles):
            for j in range(nparticles):
                self.graph.add_edge(
                        i,
                        j,
                        weight=np.linalg.norm(positions[i] - positions[j]),
                        )

    def equals(self, env, tol):
        """Checks whether a given environment is 'equal' to self

        Equality between environments is defined based on whether the
        corresponding graphs are isomorphic (up to a certain tolerance in
        the edge weights).

        Parameters
        ----------

        env : ``ClusterEnvironment`` instance
        """
        match = nx.is_isomorphic(
                self.graph,
                env.graph,
                edge_match=lambda x, y: abs(x['weight'] - y['weight']) < tol,
                node_match=lambda x, y: x['cluster_type'] == y['cluster_type'],
                )
        return match

    def match_template(self, template, tol):
        """Matches a template environment and returns all matches

        This is used to generate equivalent groups given equivalent particles.
        Templates are matched by finding isomorphic subgraphs.

        Parameters
        ----------

        template : ``ClusterEnvironment`` instance
            template for which to find a match

        """
        gm = nx.algorithms.isomorphism.GraphMatcher(
                self.graph,
                template.graph,
                edge_match=lambda x, y: abs(x['weight'] - y['weight']) < tol,
                node_match=lambda x, y: x['cluster_type'] == y['cluster_type'],
                )
        groups = []
        for iso in gm.subgraph_isomorphisms_iter():
            # check whether central particle is included
            indices = np.array(list(iso.keys()), dtype=np.int32)
            if 0 not in indices:
                continue
            # permutations of equivalent atoms may sometimes show up as
            # different isomorphisms; this means we first have to check whether
            # the current set of indices is already present in groups
            cluster_indices = tuple(np.sort(self.indices[indices]))
            if cluster_indices not in groups:
                groups.append(cluster_indices)
        return groups

    def extract(self, clusters):
        """Extracts clusters from current env to create a new environment

        Parameters
        ----------

        clusters : 1darray of ints
            Clusters which should be extracted to create a new
            environment. These integers refer to the positions, cluster_types,
            and indices arrays.

        """
        return self.__class__(
                self.positions[clusters],
                [self.cluster_types[i] for i in clusters],
                self.indices[clusters],
                )

    def generate_templates(self, max_num_equiv_clusters=6):
        """Generates templates based on current environment

        Templates are essentially subgraphs of the current environment that
        ultimately determine which groups of clusters are considered as
        candidates for future reductions. For each cluster type that is present
        in the neighborhood, all possible combinations of groups are considered
        up until the maximum number of equivalent clusters is reached.

        """
        # generate table of cluster types present in the environment;
        # but ignore the central particle (i.e. start at 1)
        table = {}
        for i, cluster_type in enumerate(self.cluster_types):
            if i == 0:
                continue
            if cluster_type not in table.keys():
                table[cluster_type] = []
            table[cluster_type].append(i)

        templates = []
        for cluster_type, neighbors in table.items():
            # sort neighbors by distance and only consider first N neighbors
            # whereby N == max_num_equiv_clusters parameter
            pos = self.positions[np.array(neighbors)]
            distances = np.linalg.norm(
                    pos - self.positions[0].reshape(1, -1),
                    axis=1,
                    )
            neighs = np.argsort(distances)[:max_num_equiv_clusters]
            for i in range(1, 2 ** len(neighs)):
                clusters = [0] # always include first cluster
                for j in range(len(neighs)):
                    # include cluster j if i // (2 ** j) == 1
                    if (i // (2 ** j) % 2):
                        clusters.append(neighbors[neighs[j]])
                assert len(clusters) > 1
                templates.append(self.extract(np.array(clusters)))
        return templates


def generate_environments(mapping, harmonic, cutoff, tol=1e-1):
    """Generates ``ClusterEnvironment`` instances based on cluster positions

    An ASE neighborlist object is used to identify clusters within a certain
    cutoff radius. Care is taken to ensure that no cluster resides close to
    the cutoff radius, as this may cause similar environments to contain
    a different number of atoms. To achieve this, a 'radius' is determined that
    is smaller than or equal to the cutoff but for which it is guaranteed that
    clusters are far enough from the boundary and do not cause any issues.
    The radius is determined per cluster_type.

    Parameters
    ----------

    mapping : easymap.Mapping
        mapping for which to generate a list of candidates

    harmonic : easymap.Harmonic
        contains the structure based on which the neighbor list is generated

    cutoff: float [angstrom]
        cutoff of the neighborlist used to construct the environment

    tol : float [angstrom]
        distance threshold above which atoms are considered distinguishable


    """
    cell = harmonic.atoms.get_cell()
    if np.allclose(cell.lengths(), np.zeros(3)):
        rvecs = None
    else:
        rvecs = cell
    positions = mapping.apply(harmonic.atoms.get_positions(), rvecs)
    nlist = get_nlist(positions, rvecs, cutoff=cutoff)

    # build lookup table of cluster_types
    table = {}
    for i, cluster_type in mapping.cluster_types.items():
        if cluster_type not in table.keys():
            table[cluster_type] = []
        table[cluster_type].append(i)

    environments = {} # environments per cluster_type
    radii = {} # determined environment radii per cluster_type
    for cluster_type, cluster_indices in table.items():
        environments[cluster_type] = []
        indices_distances = []
        for i in cluster_indices: # parse nlist for every cluster
            indices, _ = nlist.get_neighbors(i)
            if len(indices) > 0:
                _dist_indices = np.array([(i, j) for j in indices])
                distances = get_distances(_dist_indices, positions, rvecs)
                indices_distances.append(
                        (i, indices, distances),
                        )
            else: # add empty entry
                indices_distances.append((i, np.array([]), np.array([])))

        # search for proper radius value based on all distances
        _all_distances = [d for (_, __, d) in indices_distances]
        _all_distances = np.concatenate(_all_distances)
        radius = cutoff - 2 * tol # start safely below cutoff
        if len(_all_distances) > 0:
            while np.min(np.abs(_all_distances - radius)) < 2 * tol:
                radius -= 0.1 * tol
        radii[cluster_type] = radius
        for i, indices, distances in indices_distances:
            within_radius = indices[np.where(distances < radius)[0]].astype(np.int32)
            within_radius_all = np.concatenate((
                    np.array([i]),
                    within_radius,
                    ))
            types = [mapping.cluster_types[j] for j in within_radius_all]
            pos = positions[within_radius_all]
            if rvecs is not None: # apply minimum image convention
                deltas = pos - positions[i, :].reshape(1, -1)
                apply_mic(deltas, rvecs)
                assert np.allclose(np.zeros(3), deltas[0])
                pos = deltas + positions[i, :].reshape(1, -1)
            env = ClusterEnvironment(
                    pos,
                    types,
                    within_radius_all,
                    )
            environments[cluster_type].append(env)
    return environments, radii
