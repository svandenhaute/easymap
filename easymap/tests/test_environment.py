import numpy as np

from easymap.environment import SpatialEnvironment


def test_init():
    tol = 1e-1
    index = 1290
    size = 5.0
    indices = np.array([1, 3, 25, 432])
    distances = np.array([1.0, 1.4, 0.9, 3.9])
    cluster_types = [
            (0, 1, 1),
            (2,),
            (3,),
            (4, 5),
            ]
    env  = SpatialEnvironment(
            index,
            size,
            indices.copy(),
            distances.copy(),
            cluster_types.copy(),
            )
    distances += np.random.uniform(-tol, tol, size=distances.shape)
    env_ = SpatialEnvironment(
            index,
            size,
            indices.copy(),
            distances.copy(),
            cluster_types.copy(),
            )
    assert env.is_similar_to(env_, tol=tol)
    distances[0] += 2 * tol
    env__ = SpatialEnvironment(
            index,
            size,
            indices.copy(),
            distances.copy(),
            cluster_types.copy(),
            )
    assert not env.is_similar_to(env__, cutoff=4.0, tol=tol)

    # create two distances which are similar
    distances = np.array([2.5, 1.4, 0.901, 0.902])
    env = SpatialEnvironment(index, size, indices, distances, cluster_types)
    distances += np.random.uniform(-tol, tol, size=distances.shape)
    env_ = SpatialEnvironment(index, size, indices, distances, cluster_types)
    assert env.is_similar_to(env_, cutoff=4.0, tol=tol)

    # change one of the cluster types
    cluster_types = [
            (0, 1, 1, 1),
            (2,),
            (3,),
            (4, 5),
            ]
    env_ = SpatialEnvironment(index, size, indices, distances, cluster_types)
    assert not env.is_similar_to(env_, cutoff=4.0, tol=tol)

    # place differing cluster type outside cutoff
    assert env.is_similar_to(env_, cutoff=2.0, tol=tol)

    # check boundary effects
    distances = np.array([1.0, 1.4, 0.9, 3.905])
    env  = SpatialEnvironment(
            index,
            size,
            indices.copy(),
            distances.copy(),
            cluster_types.copy(),
            )
    distances = np.array([1.0, 1.4, 0.9, 3.89])
    env_ = SpatialEnvironment(
            index,
            size,
            indices.copy(),
            distances.copy(),
            cluster_types.copy(),
            )
    assert env.is_similar_to(env_, cutoff=3.904)
    assert env_.is_similar_to(env, cutoff=3.91)
