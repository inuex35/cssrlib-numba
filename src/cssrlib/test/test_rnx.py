"""
 test of RINEX decoder
"""

import gzip
import os
import shutil

import pytest

from cssrlib.rinex import rnxdec
from cssrlib.gnss import uTYP, rSigRnx
from cssrlib.gnss import sat2prn

DATA = os.path.join(os.path.dirname(__file__), "..", "data") + os.sep
OBSFILE = DATA + "SEPT078M1.21O"

SIGS = [rSigRnx("GC1C"), rSigRnx("GC2W"),
        rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("GS1C"), rSigRnx("GS2W"),
        rSigRnx("EC1X"), rSigRnx("EC5X"),
        rSigRnx("EL1X"), rSigRnx("EL5X"),
        rSigRnx("ES1X"), rSigRnx("ES5X"),
        rSigRnx("JC1C"), rSigRnx("JC2S"),
        rSigRnx("JL1C"), rSigRnx("JL2S"),
        rSigRnx("JS1C"), rSigRnx("JS2S")]


def decode_epochs(obsfile, nep=2):
    """Decode the first nep epochs, returning them as a list."""
    dec = rnxdec()
    dec.setSignals(SIGS)

    assert dec.decode_obsh(obsfile) >= 0, f"header decode failed: {obsfile}"
    dec.autoSubstituteSignals()

    epochs = []
    for _ in range(nep):
        obs = dec.decode_obs()
        if obs is None or obs.sat is None or len(obs.sat) == 0:
            break
        epochs.append(obs)
    dec.fobs.close()
    return epochs


@pytest.fixture
def gzipped_obsfile(tmp_path):
    """A gzip-compressed copy of the bundled observation file."""
    dst = tmp_path / "SEPT078M1.21O.gz"
    with open(OBSFILE, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return str(dst)


def check_epochs(epochs):
    assert len(epochs) == 2, "expected two decodable epochs"
    for obs in epochs:
        assert len(obs.sat) > 0
        # Each satellite's observations line up with its system's signal list.
        for i, sat in enumerate(obs.sat):
            sys, _ = sat2prn(sat)
            for typ, arr in ((uTYP.C, obs.P), (uTYP.L, obs.L), (uTYP.S, obs.S)):
                sigs = obs.sig[sys][typ]
                assert arr.shape[1] >= len(sigs), (
                    f"{typ!r} array too narrow for {len(sigs)} signals")
                assert i < arr.shape[0]


def test_rnx_uncompressed():
    check_epochs(decode_epochs(OBSFILE))


def test_rnx_gzipped(gzipped_obsfile):
    check_epochs(decode_epochs(gzipped_obsfile))


def test_rnx_gzipped_matches_uncompressed(gzipped_obsfile):
    """The gzip path must decode to the same values as the plain file."""
    plain = decode_epochs(OBSFILE)
    packed = decode_epochs(gzipped_obsfile)

    assert len(plain) == len(packed)
    for a, b in zip(plain, packed):
        assert list(a.sat) == list(b.sat)
        assert a.t.time == b.t.time
        assert (a.P == b.P).all()
        assert (a.L == b.L).all()
        assert (a.S == b.S).all()
