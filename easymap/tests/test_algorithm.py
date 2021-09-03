import pytest

from easymap.algorithm import Algorithm
from easymap.mapping import Mapping

from systems import get_harmonic


@pytest.mark.skip
def test_algorithm_uio():
    harmonic, equivalencies = get_harmonic('uio66_ff')
    mapping = Mapping(harmonic.atoms.get_masses(), equivalencies)

    min_neighbors = 3
    max_num_equiv_clusters = 6
    temperature = 300
    algorithm = Algorithm(
            harmonic,
            mapping,
            temperature,
            min_neighbors,
            max_num_equiv_clusters,
            )

    cutoff = 4
    tol = 1e-1
    threshold = 450 # performs only one reduction step
    algorithm.run(threshold, cutoff, tol)
