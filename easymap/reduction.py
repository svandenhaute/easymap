import logging
import numpy as np

from easymap.mapping import Mapping
from easymap.utils import get_nlist, get_distances
from easymap.environment import ClusterEnvironment, generate_environments


logger = logging.getLogger(__name__) # logging per module


class EquivalenceViolationException(Exception):
    pass


class Reduction:
    """Class to represent groups of clusters that will be combined

    The groups attribute determines which clusters are combined.

    """

    def __init__(self, groups):
        """Constructor

        Parameters
        ----------

        groups : list of tuple of ints or None
            list of tuples. Each tuple contains two or more integers that
            specify indices of clusters that will be combined.

        """
        _past_indices = []
        count = 0
        for group in groups:
            count += len(group)
            for index in group:
                if index not in _past_indices:
                    _past_indices.append(index)
        if not (len(_past_indices) == count):
            raise ValueError('groups were not disjunct: {}'.format(groups))
        self.groups = list(groups)

    def __contains__(self, query):
        """Checks whether cluster indices in query are contained in a group

        Parameters
        ----------

        query : tuple of ints
            tuple of cluster indices

        """
        for group in self.groups:
            if query[0] in group: # check if *first* index is in group
                complete = True
                for index in query: # check if *all* indices are in group
                    if index not in group:
                        complete = False
                return complete
            else: # if not, continue to next group
                continue
        return False

    def apply(self, mapping):
        """Applies reduction to mapping and merges clusters

        Parameters
        ----------

        mapping : easymap.Mapping instance
            mapping with clusters that will be merged according to self.groups

        """
        clusters = mapping.merge(mapping.clusters, self.groups)
        mapping.update_clusters(clusters)

    def check_equivalence(self, mapping):
        """Checks whether this reduction preserves equivalencies

        Parameters
        ----------

        mapping : easymap.Mapping instance
            mapping which defines the cluster types

        """
        dummy = mapping.copy()
        self.apply(dummy)
        return dummy.update_identities()

    def __eq__(self, reduction):
        """Checks whether two reductions are equal"""
        sorted_groups = [tuple(sorted(group)) for group in self.groups]
        if len(reduction.groups) != len(self.groups):
            return False
        for group in reduction.groups:
            if not tuple(sorted(group)) in sorted_groups:
                return False
        return True


def generate_reductions(mapping, harmonic, cutoff, tol,
        max_num_equiv_clusters=6):
    """Generates ``Reduction`` instances based on a mapping and a harmonic

    For each cluster type, group templates are matched to cluster
    environments in order to generate reductions which preserve equivalency
    between clusters as present in the current mapping.

    Parameters
    ----------

    """
    environments, radii = generate_environments(
            mapping,
            harmonic,
            cutoff,
            tol,
            )
    reductions = []
    for cluster_type, envs in environments.items():
        reference = envs[0]
        templates = reference.generate_templates(
                max_num_equiv_clusters=max_num_equiv_clusters,
                )
        n = len(templates)
        for env in envs:
            # verify that envs of the same cluster_type are 'equal'
            if not reference.equals(env, tol):
                raise EquivalenceViolationException('')
            # verify that each env generates the same number of templates
            n_ = len(env.generate_templates(
                    max_num_equiv_clusters=max_num_equiv_clusters,
                    ))
            if not (n_ == n):
                raise EquivalenceViolationException('')

        i = 0
        while i < len(templates):
            if len(reference.match_template(templates[i], tol)) > 1:
                templates.pop(i)
            else:
                i += 1
        for template in templates: # match with every env
            groups = []
            for env in envs:
                matches = env.match_template(template, tol)
                if not (len(matches) == 1):
                    raise EquivalenceViolationException('')
                groups.append(matches[0])
            try:
                reductions.append(Reduction(groups))
            except ValueError: # groups were not disjunct; discard template
                continue
    return reductions
