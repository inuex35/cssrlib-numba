"""Antenna phase-centre corrections.

This path had no test at all, which is how a dead branch survived in
antModelRx: it selected between nav.rcv_ant and nav.rcv_ant_b on an
``rtype`` flag, but nothing ever set rcv_ant_b and nothing ever passed
rtype != 1, so reaching it would have dereferenced None.
"""

import os

import numpy as np
import pytest

from cssrlib.gnss import ecef2pos, epoch2time, rSigRnx
from cssrlib.models.antenna import antModelRx, atxdec, searchpcv

ATX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                   "test.atx")

RECEIVER_ECEF = np.array([-3962108.7, 3381309.5, 3668678.6])
T = [2021, 3, 19, 0, 0, 0]


@pytest.fixture(scope="module")
def pcv():
    dec = atxdec()
    dec.readpcv(ATX)
    return dec


def test_antex_file_decodes(pcv):
    assert len(pcv.pcvr) > 0, "no receiver antennas decoded"
    assert len(pcv.pcvs) > 0, "no satellite antennas decoded"


def test_searchpcv_finds_a_receiver_antenna(pcv):
    name = sorted({p.type for p in pcv.pcvr})[0]
    ant = searchpcv(pcv.pcvr, name, epoch2time(T))
    assert ant is not None and ant.type == name


def test_receiver_correction_is_a_plausible_phase_centre_offset(pcv):
    """A few centimetres, one value per requested signal, all finite."""
    name = sorted({p.type for p in pcv.pcvr})[0]
    ant = searchpcv(pcv.pcvr, name, epoch2time(T))

    pos = ecef2pos(RECEIVER_ECEF)
    e = np.array([0.3, 0.2, 0.93])
    e /= np.linalg.norm(e)
    sigs = [rSigRnx("GC1C"), rSigRnx("GC2W")]

    dant = antModelRx(ant, pos, e, sigs)

    assert dant.shape == (len(sigs),)
    assert np.all(np.isfinite(dant))
    assert np.all(np.abs(dant) < 0.2), f"{dant} m is too large for a PCO/PCV"
    assert not np.allclose(dant, 0.0), "correction is identically zero"


def test_correction_varies_with_elevation(pcv):
    """The phase-centre variation is a function of zenith angle."""
    name = sorted({p.type for p in pcv.pcvr})[0]
    ant = searchpcv(pcv.pcvr, name, epoch2time(T))
    pos = ecef2pos(RECEIVER_ECEF)
    sigs = [rSigRnx("GC1C")]

    zenith = antModelRx(ant, pos, np.array([0.0, 0.0, 1.0]), sigs)
    low = np.array([0.8, 0.0, 0.6])
    low /= np.linalg.norm(low)
    slant = antModelRx(ant, pos, low, sigs)

    assert not np.isclose(zenith[0], slant[0]), (
        "the correction does not depend on elevation")


def test_signature_takes_an_antenna(pcv):
    """Regression: it used to take a whole Nav to read one field off it."""
    import inspect

    params = list(inspect.signature(antModelRx).parameters)
    assert params == ["ant", "pos", "e", "sigs"], params
