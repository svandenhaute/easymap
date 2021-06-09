

class Reduction:
    """Class to represent groups of clusters that will be combined

    The groups attribute determines which clusters are combined.

    """

    def __init__(self, groups=None):
        """Constructor

        Parameters
        ----------

        groups : list of tuple of ints or None
            list of tuples. Each tuple contains two or more integers that
            specify indices of clusters that will be combined. No index may
            appear more than once.

        """
        self.groups = groups

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
                    if index not in query:
                        complete = False
                return complete
            else: # if not, continue to next group
                continue
        return False

    def add_group(self, group, merge=False):
        """Adds group to reduction

        If the current group overlaps with any of the existing groups, the
        behavior depends on the merge keyword: if True, then both groups are
        merged into a single group. Otherwise, a ValueError is raised.

        Parameters
        ----------

        group : tuple of ints
            specifies indices of clusters that will be combined

        merge : bool
            determines merging behavior in case of overlap

        """
        pass

    def apply(self, mapping):
        pass


def generate(mapping, harmonic, epsilon=1e-1, max_size=3):
    """Generates reduction based on current cluster configuration

    1.  a neighbor list is generated based on the optimized positions.
    2.  candidate reductions are created based on proximity of clusters.

    Parameters
    ----------

    mapping : easymap.Mapping
        mapping for which to generate a list of candidates

    harmonic : easymap.Harmonic
        contains the structure based on which the neighbor list is generated

    epsilon : float
        distance threshold above which atoms are considered distinguishable

    max_size : int
        maximum size of individual groups in a candidate.

    """
    rvecs = harmonic.atoms.get_cell()
    positions = mapping.apply(harmonic.atoms.get_positions(), rvecs)
    nlist = get_nlist(positions, rvecs, cutoff=cutoff)

    #candidates = [] # stores list of candidate objects
    #size       = 2  # start with pairs
    #while size < max_size:
        # prepare dictionary that maps atom_types to tuples of indices

        #for i in range(natoms):
        #    neighbors, _ = nlist.get_neighbors(i)
        #    for j in neighbors:
        #        # check if pair already present in previous candidates:
        #        present = False
        #        count = 0
        #        while (not present) and (count < len(candidates)):
        #            present = ((i, j) in candidates[count])
        #            count += 1
        #        if not present:
        #            # generate candidate for pair

        #        else: # ignore
        #            pass
    return candidates
