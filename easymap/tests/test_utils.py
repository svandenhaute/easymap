import numpy as np

from easymap.utils import apply_mic, compute_distance_matrix, get_nlist, \
        expand_xyz


def test_mic():
    length = 20
    rvecs = length * np.eye(3)
    rvecs[1, 0] = 2 * length + 0.2 # create strongly triclinic box
    vectors = np.random.uniform(-6, 6, size=(10000, 3))
    deltas = vectors.copy()
    apply_mic(deltas, rvecs) # vectors already satisfy the mic
    assert np.allclose(vectors, deltas)
    deltas += np.random.randint(-3, 3, size=(1, 3)).dot(rvecs)
    apply_mic(deltas, rvecs) # vectors already satisfy the mic
    assert np.allclose(vectors, deltas)


def test_distance_matrix():
    natoms = 10
    positions = np.random.uniform(-4, 4, size=(natoms, 3))
    distances = np.zeros((natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            distances[i, j] = np.linalg.norm(positions[i, :] - positions[j, :])
    assert np.allclose(distances, compute_distance_matrix(positions))

    distances = np.zeros((natoms, natoms))
    rvecs = 6 * np.eye(3) + np.random.uniform(-1, 1, size=(3, 3))
    for i in range(natoms):
        for j in range(natoms):
            delta = positions[i, :] - positions[j, :]
            apply_mic(delta.reshape(1, 3), rvecs)
            distances[i, j] = np.linalg.norm(delta)
    assert np.allclose(distances, compute_distance_matrix(positions, rvecs))


def test_get_nlist():
    natoms = 100
    positions = np.random.uniform(-4, 4, size=(natoms, 3))
    rvecs = None
    cutoff = 2
    nlist = get_nlist(positions, rvecs, cutoff)
    base = 2
    for i in range(natoms):
        indices, offsets = nlist.get_neighbors(i)
        for index in indices:
            assert np.linalg.norm(positions[i, :] - positions[index, :]) < cutoff


def test_expand_xyz():
    a = np.array([[1, 4, 0], [0, 0, 7], [0, 0, 0]])
    b = np.array([
        [1, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
    assert np.allclose(b, expand_xyz(a))
