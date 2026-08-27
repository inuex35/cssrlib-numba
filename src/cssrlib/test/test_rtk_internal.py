import numpy as np
from types import SimpleNamespace

from cssrlib.engine.rtk import rtkpos


def test_build_frequency_diff_primary_only():
    helper = SimpleNamespace(nav=SimpleNamespace(nf=1))
    rover = np.array([[10.0, 5.0], [0.0, 7.0]])
    base = np.array([[4.0, 1.0], [3.0, 2.0]])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([[6.0], [0.0]]))


def test_build_frequency_diff_keeps_each_band_in_its_own_column():
    """Column f is band f, for every row.

    This used to fill column 1 with "the first column at or above 1 where
    both receivers observed something", which meant a satellite missing L2
    but carrying L5 put its L5 difference in the L2 column, beside another
    satellite's actual L2 difference. Consumers read the column as a
    frequency -- update_ambiguities takes the wavelength from
    obs.sig[sys][uTYP.L][f] and forms cp - pr/lam -- so that seeded an L5
    ambiguity with L2's wavelength.
    """
    helper = SimpleNamespace(nav=SimpleNamespace(nf=2))
    rover = np.array([
        [10.0, 0.0, 30.0, 40.0],   # no L2, but L5 is there
        [20.0, 5.0, 0.0, 50.0],    # L2 on the rover, not on the base
        [30.0, 0.0, 0.0, 0.0],     # L1 only
    ])
    base = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 7.0, 8.0],
        [3.0, 9.0, 8.0, 7.0],
    ])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([
        [9.0, 0.0],
        [18.0, 0.0],
        [27.0, 0.0],
    ]))


def test_build_frequency_diff_reaches_beyond_the_second_band():
    """Under nf >= 3 the high bands used to be permanently zero."""
    helper = SimpleNamespace(nav=SimpleNamespace(nf=4))
    rover = np.array([[10.0, 20.0, 30.0, 40.0]])
    base = np.array([[1.0, 2.0, 3.0, 4.0]])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([[9.0, 18.0, 27.0, 36.0]]))


def test_build_frequency_diff_handles_shorter_base_columns():
    helper = SimpleNamespace(nav=SimpleNamespace(nf=2))
    rover = np.array([[10.0, 0.0, 20.0], [11.0, 12.0, 13.0]])
    base = np.array([[1.0, 2.0], [3.0, 4.0]])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([
        [9.0, 0.0],
        [8.0, 8.0],
    ]))
