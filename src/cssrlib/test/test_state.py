"""Tests for the EKF state-vector layout.

These pin down the packing that used to live as three one-line methods on
the engine, and cover the two configurations whose old formulas were wrong.
"""

import numpy as np
import pytest

from cssrlib.gnss import uGNSS
from cssrlib.estimation.layout import StateLayout

MAXSAT = int(uGNSS.MAXSAT)


def test_position_only_sizing():
    lay = StateLayout(pmode=0, nf=2, ntrop=0, niono=0)
    assert lay.npos == 3
    assert lay.na == 3 and lay.nq == 3
    assert lay.nx == 3 + MAXSAT * 2
    assert lay.position == slice(0, 3)


def test_velocity_sizing():
    lay = StateLayout(pmode=1, nf=2, ntrop=0, niono=0)
    assert lay.npos == 6 and lay.na == 6
    assert lay.nx == 6 + MAXSAT * 2


def test_ambiguity_indices_are_contiguous_per_band():
    lay = StateLayout(pmode=0, nf=2, ntrop=0, niono=0)

    first = lay.ambiguity(1, 0)
    assert first == lay.na
    assert lay.ambiguity(MAXSAT, 0) == lay.na + MAXSAT - 1
    # Band 1 starts immediately after band 0.
    assert lay.ambiguity(1, 1) == lay.na + MAXSAT
    assert lay.ambiguity(MAXSAT, lay.nf - 1) == lay.nx - 1


def test_every_ambiguity_index_is_unique_and_in_range():
    lay = StateLayout(pmode=0, nf=3, ntrop=0, niono=0)
    seen = {lay.ambiguity(s, f)
            for s in range(1, MAXSAT + 1) for f in range(lay.nf)}
    assert len(seen) == MAXSAT * lay.nf
    assert min(seen) == lay.na and max(seen) == lay.nx - 1


# (ntrop, niono) across all four combinations. The old formulas
# `IT = na - MAXSAT - 1` and `II = na - MAXSAT + s - 1` baked in a
# full-length ionospheric block, so they only agreed with the sizing code
# when niono == MAXSAT.
@pytest.mark.parametrize("ntrop,niono", [
    (0, 0),
    (0, MAXSAT),
    (1, 0),          # tropo without iono: old IT went off the front
    (1, MAXSAT),
])
def test_blocks_never_overlap_and_stay_in_range(ntrop, niono):
    lay = StateLayout(pmode=0, nf=2, ntrop=ntrop, niono=niono)
    used = []

    used.extend(range(lay.npos))
    if ntrop:
        idx = lay.tropo()
        assert 0 <= idx < lay.na, f"tropo index {idx} outside the state"
        used.append(idx)
    if niono:
        for s in range(1, MAXSAT + 1):
            idx = lay.iono(s)
            assert 0 <= idx < lay.na, f"iono index {idx} outside the state"
            used.append(idx)

    assert len(set(used)) == len(used), "state blocks overlap"
    # The fixed-size block is exactly filled.
    assert sorted(set(used)) == list(range(lay.na))


def test_tropo_sits_directly_after_position():
    lay = StateLayout(pmode=0, nf=2, ntrop=1, niono=0)
    assert lay.tropo() == lay.npos == 3
    assert lay.na == 4


def test_iono_follows_tropo():
    lay = StateLayout(pmode=0, nf=2, ntrop=1, niono=MAXSAT)
    assert lay.tropo() == 3
    assert lay.iono(1) == 4
    assert lay.iono(MAXSAT) == 3 + MAXSAT
    assert lay.na == 4 + MAXSAT


def test_engine_accessors_delegate_to_the_layout():
    """gnssobs.IB / II / IT must agree with the layout that sized nav."""
    from cssrlib.gnss import Nav
    from cssrlib.engine.rtk import rtkpos

    nav = Nav(nf=2)
    rtk = rtkpos(nav, np.zeros(3))

    assert rtk.layout.na == nav.na
    assert rtk.layout.nx == nav.nx
    assert len(nav.x) == rtk.layout.nx

    for sat in (1, 32, MAXSAT):
        for f in range(nav.nf):
            assert rtk.IB(sat, f, nav.na) == rtk.layout.ambiguity(sat, f)


def test_from_nav_round_trips():
    from cssrlib.gnss import Nav
    from cssrlib.engine.rtk import rtkpos

    nav = Nav(nf=2)
    rtk = rtkpos(nav, np.zeros(3))

    rebuilt = StateLayout.from_nav(nav)
    assert (rebuilt.na, rebuilt.nx, rebuilt.nq) == (
        rtk.layout.na, rtk.layout.nx, rtk.layout.nq)
