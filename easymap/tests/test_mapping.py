import pytest
import numpy as np

from easymap import Mapping


def test_project_toy():
    natoms = 5
    masses = np.ones(natoms)
    mapping = Mapping(masses)
    assert np.all(mapping.clusters == np.eye(natoms, dtype=np.int32))
    assert mapping.nclusters == natoms
    assert np.all(mapping.deltas == np.zeros((natoms, natoms), dtype=np.int32))
    assert np.all(mapping.transform == np.eye(natoms))

    clusters = np.array([
        [1, 1, 0, 0, 0], # first group
        [0, 0, 1, 1, 1], # second group
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        ], dtype=np.int32)
    deltas = np.array([
        [0, 0, 0, 0, 0],
        [-1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, -1, 0, 1],
        ], dtype=np.int32)
    transform = np.array([
        [1/2, 1/2, 0, 0, 0],
        [0, 0, 1/3, 1/3, 1/3],
        ])
    mapping.update_clusters(clusters)
    assert np.all(mapping.clusters == clusters)
    assert np.all(mapping.deltas == deltas)
    assert np.allclose(mapping.transform, transform)

    # try updating corrupt clusters
    clusters = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], # atom 1 appears in cluster 0 and cluster 4
        ], dtype=np.int32)
    with pytest.raises(AssertionError):
        mapping.update_clusters(clusters)


def test_manual_computation_nonperiodic():
    natoms = 47
    masses = np.random.uniform(3, 10, size=(natoms,))
    mapping = Mapping(masses)
    group_sizes = [19, 10, 1, 5, 12]
    clusters = np.zeros((natoms, natoms))
    for i, size in enumerate(group_sizes):
        start = sum(group_sizes[:i])
        end = start + size
        clusters[i, start:end] = 1
    mapping.update_clusters(clusters)

    positions = np.random.uniform(-5, 5, size=(natoms, 3))
    positions_com  = mapping.apply(positions)

    # manual computation
    positions_com_manual = np.zeros(positions_com.shape)
    for i, group in enumerate(mapping):
        mass = np.sum(masses[np.array(group)])
        for j in group:
            positions_com_manual[i, :] += positions[j, :] * masses[j] / mass
    assert np.allclose(positions_com, positions_com_manual)

    positions_com_ = mapping.apply(positions + 2.3)
    assert np.allclose(positions_com, positions_com_ - 2.3)
