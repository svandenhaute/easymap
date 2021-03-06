import numpy as np

from ase import Atoms
import ase.units
import ase.geometry
import ase.neighborlist


def get_entropy(frequencies, temperature, use_quantum=True, remove_zero=True):
    """Computes the entropy of a series of harmonic oscillators

    Frequencies below 1e-10 ase units are removed by default because these can
    only be related to translational invariance (in case of periodic systems) or
    translational and rotational invariance (in case of nonperiodic systems).
    The remaining frequencies are also returned.

    Parameters
    ----------

    frequencies : 1darray [ASE unit of time]
        eigenmode frequencies

    temperature : float [kelvin]
        temperature at which to compute the entropy

    remove_zero : bool
        whether or not to discard near-zero frequencies when computing the
        entropy

    use_quantum : bool
        determines whether to use the quantum mechanical or classical treatment;
        for classical oscillators, the entropy becomes negative at large
        frequencies.

    """
    if remove_zero:
        mask = frequencies > 1e-5
        frequencies = frequencies[mask]
    entropy = np.zeros(frequencies.shape)
    frequencies_si = frequencies * ase.units.s # in s
    h = ase.units._hplanck # in J s
    k = ase.units._k # in J / K
    beta = 1 / (k * temperature)
    thetas = beta * h * frequencies_si # dimensionless quantity
    if use_quantum:
        q_quantum = np.exp(- thetas / 2) / (1 - np.exp(- thetas))
        f_quantum = - np.log(q_quantum) / beta
        s_quantum = -k * (np.log(1 - np.exp(- thetas)) - thetas / (np.exp(thetas) - 1))
        s_quantum /= 1000
        s_quantum *= ase.units._Nav # to kJ/mol
        entropy[:] = s_quantum
    else:
        q_classical = 1 / (thetas)
        f_classical = - np.log(q_classical) / beta
        s_classical = k * (1 + np.log(1 / thetas))
        s_classical /= 1000
        s_classical *= ase.units._Nav
        entropy[:] = s_classical
    return entropy, frequencies


def is_lower_triangular(rvecs):
    """Returns whether rvecs are in lower triangular form

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (rvecs[0, 0] > 0 and # positive volumes
            rvecs[1, 1] > 0 and
            rvecs[2, 2] > 0 and
            rvecs[0, 1] == 0 and # lower triangular
            rvecs[0, 2] == 0 and
            rvecs[1, 2] == 0)


def is_reduced(rvecs):
    """Returns whether rvecs are in reduced form

    OpenMM puts requirements on the components of the box vectors.
    Essentially, rvecs has to be a lower triangular positive definite matrix
    where additionally (a_x > 2*|b_x|), (a_x > 2*|c_x|), and (b_y > 2*|c_y|).

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    return (rvecs[0, 0] > abs(2 * rvecs[1, 0]) and # b mostly along y axis
            rvecs[0, 0] > abs(2 * rvecs[2, 0]) and # c mostly along z axis
            rvecs[1, 1] > abs(2 * rvecs[2, 1]) and # c mostly along z axis
            is_lower_triangular(rvecs))


def transform_lower_triangular(pos, rvecs, reorder=False):
    """Transforms coordinate axes such that cell matrix is lower diagonal

    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.

    Parameters
    ----------

    pos : array_like
        (natoms, 3) array containing atomic positions

    rvecs : array_like
        (3, 3) array with box vectors as rows

    reorder : bool
        whether box vectors are reordered from largest to smallest

    """
    if reorder: # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(rvecs, axis=1)
        ordering = np.argsort(norms)[::-1] # largest first
        a = rvecs[ordering[0], :].copy()
        b = rvecs[ordering[1], :].copy()
        c = rvecs[ordering[2], :].copy()
        rvecs[0, :] = a[:]
        rvecs[1, :] = b[:]
        rvecs[2, :] = c[:]
    q, r = np.linalg.qr(rvecs.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r)) # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors # full (improper) rotation
    pos[:]   = pos @ rotation
    rvecs[:] = rvecs @ rotation
    assert np.allclose(rvecs, np.linalg.cholesky(rvecs @ rvecs.T))
    rvecs[0, 1] = 0
    rvecs[0, 2] = 0
    rvecs[1, 2] = 0


def reduce_box_vectors(rvecs):
    """Uses linear combinations of box vectors to obtain the reduced form

    The reduced form of a cell matrix is lower triangular, with additional
    constraints that enforce vector b to lie mostly along the y-axis and vector
    c to lie mostly along the z axis.

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows. These should already by in
        lower triangular form.

    """
    # simple reduction algorithm only works on lower triangular cell matrices
    assert is_lower_triangular(rvecs)
    # replace c and b with shortest possible vectors to ensure 
    # b_y > |2 c_y|
    # b_x > |2 c_x|
    # a_x > |2 b_x|
    rvecs[2, :] = rvecs[2, :] - rvecs[1, :] * np.round(rvecs[2, 1] / rvecs[1, 1])
    rvecs[2, :] = rvecs[2, :] - rvecs[0, :] * np.round(rvecs[2, 0] / rvecs[0, 0])
    rvecs[1, :] = rvecs[1, :] - rvecs[0, :] * np.round(rvecs[1, 0] / rvecs[0, 0])


def determine_rcut(rvecs):
    """Determines the maximum allowed cutoff radius of rvecs

    The maximum cutoff radius should be determined based on the reduced cell

    Parameters
    ----------

    rvecs : array_like
        (3, 3) array with box vectors as rows

    """
    rvecs_ = rvecs.copy()
    # reorder necessary for some systems (e.g. cau13); WHY?
    transform_lower_triangular(np.zeros((1, 3)), rvecs_, reorder=True)
    reduce_box_vectors(rvecs_)
    assert is_reduced(rvecs_)
    return min([
            rvecs_[0, 0],
            rvecs_[1, 1],
            rvecs_[2, 2],
            ]) / 2


def apply_mic(deltas, rvecs):
    """Applies minimum image convention to relative vectors in place

    The computation of the minimum image vector here is correct even if its
    length exceeds half the shortest diagonal box vector components in the
    reduced representation.

    Parameters
    ----------

    deltas : 2darray of shape (ndeltas, 3)
        array of relative vectors

    rvecs : 2darray of shape (3, 3)
        box vectors; these need not be in lower triangular form.

    """
    cell = ase.geometry.Cell(rvecs)
    mic_vectors, _ = ase.geometry.geometry.find_mic(deltas, rvecs)
    deltas[:] = mic_vectors


def compute_distance_matrix(positions, rvecs=None):
    """Computes distance matrix of a set of particles"""
    natoms = positions.shape[0]
    pos = np.copy(positions).reshape(natoms, 3, 1)
    deltas = pos - pos.T
    if rvecs is not None:
        reshaped = np.swapaxes(deltas, 1, 2).reshape((-1, 3))
        apply_mic(reshaped, rvecs)
        distances = np.linalg.norm(reshaped, axis=1).reshape((natoms, natoms))
    else:
        distances = np.linalg.norm(deltas, axis=1)
    return distances


def get_nlist(positions, rvecs, cutoff):
    """Constructs ASE neighbor list"""
    # create dummy Atoms instance
    natoms = positions.shape[0]
    atoms = Atoms(
            positions=positions,
            )
    if rvecs is not None:
        cell = ase.geometry.Cell(rvecs)
        atoms.set_cell(cell)
        atoms.set_pbc(True)
    else:
        cell = None
    nlist = ase.neighborlist.NeighborList(
            cutoff / 2 * np.ones(natoms),
            sorted=False,
            bothways=True,
            skin=0.0,
            self_interaction=False,
            primitive=ase.neighborlist.NewPrimitiveNeighborList,
            )
    nlist.update(atoms)
    return nlist


def get_distances(indices, positions, rvecs=None):
    """Compute distances between particles"""
    atoms = Atoms(
            positions=positions,
            )
    if rvecs is not None:
        cell = ase.geometry.Cell(rvecs)
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        mic = True
    else:
        cell = None
        mic = False

    ndist = indices.shape[0]
    distances = np.zeros(ndist)
    for i in range(ndist):
        distances[i] = atoms.get_distance(indices[i, 0], indices[i, 1], mic=mic)
    return distances


def expand_xyz(m):
    """Expands an m-by-n matrix into a 3m-by-3n matrix

    The original matrix is repeated on the diagonals of a 3-by-3 block matrix.
    Next, its rows and columns are interleaved.

    """
    N = m.shape[0]
    n = m.shape[1]
    block = np.kron(np.eye(3), m)
    indices_n = np.arange(3 * n).reshape(3, n).T.flatten()
    indices_N = np.arange(3 * N).reshape(3, N).T.flatten()
    permute_rows = block[indices_N, :]
    total = permute_rows[:, indices_n]
    return total
