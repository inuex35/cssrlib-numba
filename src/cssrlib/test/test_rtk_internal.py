import numpy as np
from types import SimpleNamespace

from cssrlib.rtk import rtkpos


def test_build_frequency_diff_primary_only():
    helper = SimpleNamespace(nav=SimpleNamespace(nf=1))
    rover = np.array([[10.0, 5.0], [0.0, 7.0]])
    base = np.array([[4.0, 1.0], [3.0, 2.0]])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([[6.0], [0.0]]))


def test_build_frequency_diff_uses_first_valid_secondary_pair():
    helper = SimpleNamespace(nav=SimpleNamespace(nf=2))
    rover = np.array([
        [10.0, 0.0, 30.0, 40.0],
        [20.0, 5.0, 0.0, 50.0],
        [30.0, 0.0, 0.0, 0.0],
    ])
    base = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 7.0, 8.0],
        [3.0, 9.0, 8.0, 7.0],
    ])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([
        [9.0, 27.0],
        [18.0, 42.0],
        [27.0, 0.0],
    ]))


def test_build_frequency_diff_handles_shorter_base_columns():
    helper = SimpleNamespace(nav=SimpleNamespace(nf=2))
    rover = np.array([[10.0, 0.0, 20.0], [11.0, 12.0, 13.0]])
    base = np.array([[1.0, 2.0], [3.0, 4.0]])

    diff = rtkpos._build_frequency_diff(helper, rover, base)

    np.testing.assert_allclose(diff, np.array([
        [9.0, 0.0],
        [8.0, 8.0],
    ]))
