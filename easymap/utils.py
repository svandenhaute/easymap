import numpy as np
import ase.units


def get_entropy(frequencies, temperature, use_quantum=True, remove_zero=True):
    """Computes the entropy of a series of harmonic oscillators

    Frequencies below 1e-10 ase units are removed by default because these can
    only be related to translational invariance (in case of periodic systems) or
    translational and rotational invariance ( in case of nonperiodic systems).
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
        mask = np.where(frequencies > 1e-5)[0]
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
