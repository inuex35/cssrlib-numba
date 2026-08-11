"""
 test of broadcast ephemeris decoding and satellite position computation
"""

import gzip
import os
import shutil

import numpy as np
import pytest

from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav, epoch2time, prn2sat, uGNSS, timeadd, ecef2pos
from cssrlib.models.ephemeris import findeph, eph2pos

DATA = os.path.join(os.path.dirname(__file__), "..", "data") + os.sep
NAVFILE = DATA + "30340780.21q"

# The QZSS navigation file covers 2021-03-19 (doy 078).
T0 = [2021, 3, 19, 0, 0, 0]
QZS_PRN = 194

# QZS-1 flies an eccentric (e ~ 0.075) inclined geosynchronous orbit, so the
# geocentric radius swings either side of the 42164 km geosynchronous value
# rather than sitting on it.
GEO_RADIUS_MIN_M = 38.0e6
GEO_RADIUS_MAX_M = 46.0e6


def load_nav(navfile):
    dec = rnxdec()
    return dec.decode_nav(navfile, Nav())


@pytest.fixture
def gzipped_navfile(tmp_path):
    dst = tmp_path / "30340780.21q.gz"
    with open(NAVFILE, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return str(dst)


def test_findeph_and_eph2pos():
    """A QZSS ephemeris resolves to a plausible geosynchronous position."""
    nav = load_nav(NAVFILE)
    assert len(nav.eph) > 0, "no ephemerides decoded"

    t = epoch2time(T0)
    sat = prn2sat(uGNSS.QZS, QZS_PRN)
    eph = findeph(nav.eph, t, sat)
    assert eph is not None, f"no ephemeris found for PRN {QZS_PRN}"

    rs, vs, dts = eph2pos(t, eph, True)
    assert rs.shape == (3,) and vs.shape == (3,)
    assert np.all(np.isfinite(rs)) and np.all(np.isfinite(vs))

    # Geocentric radius within the QZSS orbit's apogee/perigee band.
    r = np.linalg.norm(rs)
    assert GEO_RADIUS_MIN_M < r < GEO_RADIUS_MAX_M, f"radius {r:.0f} m"

    # Orbital speed is a few km/s, and the clock offset is small.
    assert 1.0e3 < np.linalg.norm(vs) < 5.0e3
    assert abs(dts) < 1.0e-3


def test_ground_track_stays_over_asia_pacific():
    """QZSS ground track over a day stays in its designed longitude band."""
    nav = load_nav(NAVFILE)
    t0 = epoch2time(T0)
    sat = prn2sat(uGNSS.QZS, QZS_PRN)
    eph = findeph(nav.eph, t0, sat)
    assert eph is not None

    n = 24 * 3600 // 300
    lat = np.zeros(n)
    lon = np.zeros(n)
    for i in range(n):
        rs, _ = eph2pos(timeadd(t0, i * 300), eph)
        pos = ecef2pos(rs)
        lat[i] = np.rad2deg(pos[0])
        lon[i] = np.rad2deg(pos[1])

    assert np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
    # Quasi-zenith orbit: figure-eight centred near 135E.
    assert 100.0 < np.median(lon) < 170.0
    assert np.max(np.abs(lat)) < 60.0
    # It really does move (not a stuck propagation).
    assert np.ptp(lat) > 10.0


def test_gzipped_navfile_matches_plain(gzipped_navfile):
    """The gzip path decodes to the same ephemerides as the plain file."""
    nav_plain = load_nav(NAVFILE)
    nav_gz = load_nav(gzipped_navfile)

    assert len(nav_plain.eph) == len(nav_gz.eph)

    t = epoch2time(T0)
    sat = prn2sat(uGNSS.QZS, QZS_PRN)
    e1 = findeph(nav_plain.eph, t, sat)
    e2 = findeph(nav_gz.eph, t, sat)
    assert (e1 is None) == (e2 is None)
    if e1 is not None:
        np.testing.assert_allclose(eph2pos(t, e1)[0], eph2pos(t, e2)[0])
