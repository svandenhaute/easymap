import numpy as np

from easymap.utils import apply_mic


def test_mic():
    length = 20
    rvecs = length * np.eye(3)
    rvecs[1, 0] = 40.1 # create strongly triclinic box
    vectors = np.random.uniform(-6, 6, size=(10000, 3))
    deltas = vectors.copy()
    apply_mic(deltas, rvecs) # vectors already satisfy the mic
    assert np.allclose(vectors, deltas)
    deltas += np.random.randint(-3, 3, size=(1, 3)).dot(rvecs)
    apply_mic(deltas, rvecs) # vectors already satisfy the mic
    assert np.allclose(vectors, deltas)
