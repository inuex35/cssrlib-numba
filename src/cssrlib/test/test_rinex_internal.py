from io import StringIO

import numpy as np

from cssrlib.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.rinex import rnxdec


def _obs_field(value, lli=" ", snr=" "):
    if value is None:
        return " " * 16
    return f"{value:14.3f}{lli}{snr}"


def test_decode_obs_keeps_selected_signals_and_shapes():
    dec = rnxdec()
    dec.setSignals([
        rSigRnx("GC1C"),
        rSigRnx("GL1C"),
        rSigRnx("GD1C"),
        rSigRnx("GS1C"),
        rSigRnx("RC1C"),
        rSigRnx("RL1C"),
        rSigRnx("RD1C"),
        rSigRnx("RS1C"),
    ])

    header = (
        "     3.03           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES\n"
        "R    4 C1C L1C D1C S1C                                      SYS / # / OBS TYPES\n"
        "END OF HEADER\n"
    )
    epoch = "> 2021 03 19 00 00  0.0000000  0  2\n"
    gps = "G01" + "".join([
        _obs_field(12345.0),
        _obs_field(23456.0, "1"),
        _obs_field(12.5),
        _obs_field(45.0),
    ]) + "\n"
    glo = "R02" + "".join([
        _obs_field(22345.0),
        _obs_field(33456.0),
        _obs_field(22.5),
        _obs_field(35.0),
    ]) + "\n"

    dec.fobs = StringIO(header)
    assert dec._decode_obsh() == 0
    dec.fobs = StringIO(epoch + gps + glo)
    obs = dec.decode_obs()

    np.testing.assert_array_equal(
        obs.sat,
        np.array([1, uGNSS.GLOMIN + 2], dtype=np.int32),
    )
    assert obs.P.shape == (2, 1)
    assert obs.L.shape == (2, 1)
    assert obs.D.shape == (2, 1)
    assert obs.S.shape == (2, 1)
    assert obs.lli.shape == (2, 1)
    np.testing.assert_allclose(obs.P[:, 0], np.array([12345.0, 22345.0]))
    np.testing.assert_allclose(obs.L[:, 0], np.array([23456.0, 33456.0]))
    np.testing.assert_allclose(obs.D[:, 0], np.array([12.5, 22.5]))
    np.testing.assert_allclose(obs.S[:, 0], np.array([45.0, 35.0]))
    np.testing.assert_array_equal(obs.lli[:, 0], np.array([1, 0], dtype=np.int32))


def test_decode_obs_skips_unselected_constellation():
    dec = rnxdec()
    dec.setSignals([rSigRnx("GC1C")])

    header = (
        "     3.03           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G    1 C1C                                                  SYS / # / OBS TYPES\n"
        "R    1 C1C                                                  SYS / # / OBS TYPES\n"
        "END OF HEADER\n"
    )
    epoch = "> 2021 03 19 00 00  0.0000000  0  2\n"
    gps = "G01" + _obs_field(12345.0) + "\n"
    glo = "R02" + _obs_field(22345.0) + "\n"

    dec.fobs = StringIO(header)
    assert dec._decode_obsh() == 0
    dec.fobs = StringIO(epoch + gps + glo)
    obs = dec.decode_obs()

    np.testing.assert_array_equal(obs.sat, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(obs.P[:, 0], np.array([12345.0]))
