import logging
import numpy as np

from easymap.utils import get_entropy, get_nlist
from easymap.mapping import score
from easymap.reduction import generate_reductions

logger = logging.getLogger(__name__)


class Algorithm:
    """Implements a greedy approach to the optimization of the mapping entropy

    """

    def __init__(self, harmonic, mapping, temperature=300, min_neighbors=3,
            max_num_equiv_clusters=6):
        """Constructor

        Parameters
        ----------

        """
        self.harmonic    = harmonic
        self.mapping     = mapping
        self.temperature = temperature

        # compute and store total entropy of the system
        frequencies, _ = harmonic.compute_eigenmodes()
        entropies, _   = get_entropy(frequencies, temperature)
        self.entropy   = np.sum(entropies)

        # determine initial cutoff of the neighbor lists such that every atom
        # contains a minimum number of neighbors; as determined by the
        # min_neighbors parameter.
        #self.adjust_cutoff(min_neighbors)
        self.max_num_equiv_clusters = max_num_equiv_clusters

    def adjust_cutoff(self, min_neighbors, max_iter=20):
        """Adjusts cutoff such that each particle has enough neighbors"""
        count = 0
        while count < max_iter:
            # check whether current cutoff contains enough neighbors for all
            cell = self.harmonic.atoms.get_cell()
            if np.allclose(cell.lengths(), np.zeros(3)):
                rvecs = None
            else:
                rvecs = cell
            positions = self.mapping.apply(
                    self.harmonic.atoms.get_positions(),
                    rvecs,
                    )
            nlist = get_nlist(positions, rvecs, cutoff=self.cutoff)
            for i in range(self.mapping.natoms):
                neighs, _ = nlist.get_neighbors(i)
                if len(neighs) < min_neighbors:
                    violation = len(neighs)
                    break # exit for loop because violation found
                if i == (self.mapping.natoms - 1): # no violations found
                    return None

            # increase cutoff with 5 percent and restart
            self.cutoff *= 1.05
            count += 1

        if count == max_iter:
            logger.critical('did not find cutoff for which each atom possesses'
                    ' at least {} neighbors (found atom with only {})'.format(
                        min_neighbors,
                        violation,
                        ))

    def run(self, threshold, cutoff, tol):
        """Executes the algorithm

        """
        n = self.mapping.nclusters
        count = 0
        while (n > threshold):
            logger.debug('')
            logger.debug('=' * 80)
            logger.debug('ITERATION {}'.format(count))
            logger.debug('=' * 80)
            logger.debug('')
            logger.debug('GENERATING REDUCTIONS')
            reductions = generate_reductions(
                    self.mapping,
                    self.harmonic,
                    cutoff,
                    tol,
                    self.max_num_equiv_clusters,
                    )
            if len(reductions) > 0:
                ndof = np.array([(len(r.groups) * (len(r.groups[0]) - 1)) for r in reductions])
                nequiv = np.array([len(r.groups) for r in reductions])
                logger.debug('SCORING REDUCTIONS')
                smap = np.zeros(len(reductions))

                # compute mass-weighted Hessian. Do not remove translational/rotational
                # modes.
                masses = np.repeat(self.harmonic.atoms.get_masses(), 3)
                mass_matrix = 1 / np.sqrt(np.outer(masses, masses))
                hessian_mw = mass_matrix * self.harmonic.hessian
                for i, reduction in enumerate(reductions):
                    mapping = self.mapping.copy()
                    reduction.apply(mapping)
                    smap[i] = score(
                            mapping,
                            hessian_mw=hessian_mw,
                            temperature=self.temperature,
                            )
                penalty = (smap) / (nequiv) # lowest smap increase
                winner = np.argmin(penalty)
                self.log_scores(penalty, ndof, reductions, winner)
                reductions[winner].apply(self.mapping)
                self.mapping.update_identities()
                assert self.mapping.nclusters == (n - ndof[winner])
                n = self.mapping.nclusters
                count += 1
            else:
                logger.debug('no reductions found; algorithm is terminating')
                break

    def log_scores(self, penalty, ndof, reductions, winner):
        """Logs the mapping entropy scores of reductions"""
        order = np.argsort(penalty)
        nspaces = 10
        header  = 'cluster types'
        header += nspaces * ' '
        header += 'number of equivalent clusters'
        header += nspaces * ' '
        header += 'NDOF reduction'
        header += nspaces * ' '
        header += 'penalty (Smap / nequiv)'
        logger.info(header)
        for index in order:
            r = reductions[index]
            cluster_types = [self.mapping.cluster_types[i] for i in r.groups[0]]
            nequiv = len(r.groups)
            ndof = nequiv * (len(r.groups[0]) - 1)

            logger.info('')
            if index == winner:
                line = '(*){:^9}'.format(str(cluster_types[0]))
            else:
                line = '{:^12}'.format(str(cluster_types[0]))
            line += nspaces * ' '
            line += '{:^30}'.format(nequiv)
            line += nspaces * ' '
            line += '{:^15}'.format(ndof)
            line += nspaces * ' '
            line += '{:^10.6f}'.format(penalty[index])
            logger.info(line)
            for j in range(1, len(cluster_types)):
                line = '{:^12}'.format(str(cluster_types[j]))
                logger.info(line)
