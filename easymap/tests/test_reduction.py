import pytest
import numpy as np

from ase.io import write

from easymap.reduction import Reduction, generate_reductions
from easymap.mapping import Mapping
from easymap.environment import ClusterEnvironment

from systems import get_harmonic


def test_init_contains():
    groups = [
            (0, 1, 2),
            (3, 4),
            (10, 12),
            ]
    reduction = Reduction(groups)
    assert (0, 1) in reduction
    assert not ((11,) in reduction)
    assert not ((0, 3) in reduction)

    groups = [
            (0, 1, 2),
            (3, 4, 0),
            (10, 12),
            ]
    with pytest.raises(ValueError):
        reduction = Reduction(groups)
    groups = [
            (0, 0, 2),
            (3, 4, 34),
            (10, 12),
            ]
    with pytest.raises(ValueError):
        reduction = Reduction(groups)


def test_equality():
    groups = [
            (0, 1, 2),
            (3, 4),
            (10, 12),
            ]
    reduction = Reduction(groups)

    groups = [
            (2, 0, 1),
            (3, 4),
            (10, 12),
            ]
    reduction_ = Reduction(groups)
    assert reduction == reduction_

    groups = [
            (2, 0, 1),
            (3,),
            (10, 12),
            ]
    reduction_ = Reduction(groups)
    assert reduction != reduction_


def test_apply():
    natoms = 5
    masses = np.ones(natoms)
    equivalency = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        ], dtype=np.int32)
    clusters = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        ], dtype=np.int32)
    mapping = Mapping(masses, equivalency)
    mapping.update_clusters(clusters)
    assert mapping.update_identities()

    groups = [
            (0, 1),
            ]
    reduction = Reduction(groups)
    assert reduction.check_equivalence(mapping)
    groups = [
            (0, 2),
            ]
    reduction = Reduction(groups)
    assert not reduction.check_equivalence(mapping)


def test_uio():
    harmonic, equivalencies = get_harmonic('uio66_ff')
    mapping = Mapping(harmonic.atoms.get_masses(), equivalencies)
    #write('test.xyz', harmonic.atoms)

    # contains reference values on the number of reductions as a function of
    # the maximum number of equivalent clusters to be considered
    cutoff = 4
    tol = 1e-1
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=1,
            )
    assert len(reductions) == 10
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=2,
            )
    assert len(reductions) == 18
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=3,
            )
    assert len(reductions) == 21
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=4,
            )
    assert len(reductions) == 24
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=5,
            )
    assert len(reductions) == 24
    reductions = generate_reductions(
            mapping,
            harmonic,
            cutoff=cutoff,
            tol=tol,
            max_num_equiv_clusters=6,
            )
    assert len(reductions) == 27
