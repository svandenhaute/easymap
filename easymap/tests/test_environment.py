import numpy as np
import pytest
import ase

from easymap.environment import ClusterEnvironment, generate_environments
from easymap.mapping import Mapping
from easymap.harmonic import Harmonic

from systems import get_harmonic


def test_eq():
    tol = 1e-2
    nparticles = 100
    positions = np.random.uniform(0, 4, size=(nparticles, 3))
    cluster_types = [
            (0, 0,),
            (0, 1, 1),
            (2,),
            (3,),
            (4, 5, 6, 7, 7, 8,),
            ]
    _ = np.random.randint(0, len(cluster_types), size=(nparticles,))
    indices = np.random.randint(0, 1000, size=(nparticles,))
    foo = ClusterEnvironment(
            positions,
            [cluster_types[i] for i in _], # random cluster types from list
            indices,
            )
    bar = ClusterEnvironment(
            positions + np.random.uniform(0, 0.45 * tol, size=(nparticles, 3)),
            [cluster_types[i] for i in _], # random cluster types from list
            indices,
            )
    assert foo.equals(bar, tol=tol)

    bar = ClusterEnvironment( # perturbation too large
            positions + np.random.uniform(0, 1.0 * tol, size=(nparticles, 3)),
            [cluster_types[i] for i in _], # random cluster types from list
            indices,
            )
    assert not foo.equals(bar, tol=tol)

    cluster_types = [
            (0, 0,),
            (0, 1, 1),
            (12389,), # change one
            (3,),
            (4, 5, 6, 7, 7, 8,),
            ]
    bar = ClusterEnvironment(
            positions + np.random.uniform(0, 0.45 * tol, size=(nparticles, 3)),
            [cluster_types[i] for i in _], # random cluster types from list
            indices,
            )
    assert not foo.equals(bar, tol=tol)


def test_template():
    tol = 1e-2
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1.4, 0],
        [2, 1, 0],
        ])
    cluster_types = [
            (0,),
            (1,),
            (2,),
            (0, 1, 2,),
            ]
    indices = np.random.randint(0, 100, size=(4,))
    env = ClusterEnvironment(
            positions,
            cluster_types,
            indices,
            )
    template = ClusterEnvironment(
            np.array([[1, 0, 0], [1, 1.4003, 0]]),
            [(0,), (2,)],
            np.zeros(2),
            )
    assert len(env.match_template(template, tol)) > 0
    template = ClusterEnvironment(
            np.array([[1, 0, 0], [1, 1.5003, 0]]),
            [(0,), (2,)],
            np.zeros(2),
            )
    assert len(env.match_template(template, tol)) == 0
    template = ClusterEnvironment(
            np.array([[1, 0, 0], [1, 1.4003, 0]]),
            [(0,), (0, 1,)],
            np.zeros(2),
            )
    assert len(env.match_template(template, tol)) == 0


def test_generate_toy():
    natoms = 3
    mapping = Mapping(np.ones(natoms), np.eye(natoms))
    atoms = ase.Atoms(
            numbers=np.ones(natoms),
            positions=np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0]]),
            pbc=False,
            )
    harmonic = Harmonic(atoms, np.ones((3 * natoms, 3 * natoms)))

    cutoff = 2.001
    tol = 1e-2
    envs, radii = generate_environments(mapping, harmonic, cutoff, tol)
    assert tuple(envs.keys()) == ((0,), (1,), (2,))
    assert radii[(1,)] == cutoff - 2 * tol # radius at default value
    assert radii[(2,)] == radii[(0,)]


def test_template_toy():
    natoms = 6
    equivalencies = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]])
    mapping = Mapping(np.ones(natoms), equivalencies)
    atoms = ase.Atoms(
            numbers=np.ones(natoms),
            positions=np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                    [4, 0, 0],
                    [5, 0, 0]]),
            pbc=False,
            )
    harmonic = Harmonic(atoms, np.ones((3 * natoms, 3 * natoms)))

    cutoff = 6.5
    tol = 1e-2
    envs, radii = generate_environments(mapping, harmonic, cutoff, tol)
    assert tuple(envs.keys()) == ((0,), (1,)) # only two cluster types
    assert len(envs[(0,)]) == 3
    assert len(envs[(1,)]) == 3

    env = envs[(0,)][0]
    templates = env.generate_templates()
    assert len(templates) == 7 + 3

    templates = env.generate_templates(max_num_equiv_clusters=2)
    assert len(templates) == 3 + 3

    templates = env.generate_templates(max_num_equiv_clusters=1)
    assert len(templates) == 1 + 1

    # decrease cutoff
    cutoff = 4.1
    tol = 1e-2
    envs, radii = generate_environments(mapping, harmonic, cutoff, tol)
    assert tuple(envs.keys()) == ((0,), (1,)) # only two cluster types
    assert len(envs[(0,)]) == 3
    assert len(envs[(1,)]) == 3

    env = envs[(0,)][0] # env for 0, 0, 0 position
    templates = env.generate_templates()
    assert len(templates) == 3 + 3

    env = envs[(0,)][2] # env for 2, 0, 0 position
    templates = env.generate_templates()
    assert len(templates) == 3 + 7


def test_generate_uio():
    harmonic, equivalencies = get_harmonic('uio66_ff')
    mapping = Mapping(harmonic.atoms.get_masses(), equivalencies)

    cutoff = 3.0
    tol = 1e-1
    environments, radii = generate_environments(mapping, harmonic, cutoff, tol)
    # check whether each cluster receives precisely one environment 
    _all = []
    for _, envs in environments.items():
        for env in envs:
            _all.append(env.indices[0]) # get index of central atom
    assert np.allclose(np.sort(np.array(_all)), np.arange(mapping.nclusters))

    # verify that envs of the same cluster_type are 'equal'
    # verify that each env generates the same number of templates
    for cluster_type, envs in environments.items():
        reference = envs[0]
        templates = reference.generate_templates()
        n = len(templates)
        for env in envs:
            assert reference.equals(env, tol)
            assert len(env.generate_templates()) == n

        # discard those which cannot be matched uniquely 
        i = 0
        while i < len(templates):
            if len(reference.match_template(templates[i], tol)) > 1:
                templates.pop(i)
            else:
                i += 1
        for env in envs:
            for template in templates:
                assert len(env.match_template(template, tol)) == 1
