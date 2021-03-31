import numpy as np
import ase.units

from easymap.utils import get_entropy

from systems import get_harmonic


def test_init():
    name = 'uio66_ff'
    harmonic = get_harmonic(name)
    assert harmonic.periodic


def test_frequencies():
    name = 'uio66_ff'
    harmonic = get_harmonic(name)
    values, _ = harmonic.compute_eigenmodes()
    frequencies = np.sqrt(values) / (2 * np.pi)
    entropy, _ = get_entropy(frequencies, 300)
    assert np.isclose(np.sum(entropy), 5.328, atol=1e-3) # reference value

    # check whether largest frequencies are below 4000 and above 3800 cm^-1
    frequencies_invcm = frequencies * ase.units.s / (ase.units._c * 100)
    assert np.all(frequencies_invcm[3:] < 4e3)
    assert np.any(frequencies_invcm[3:] > 3.8e3)
