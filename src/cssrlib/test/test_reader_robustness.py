"""Two ways the readers used to damage or hang on their inputs.

Both are about state that belongs to somebody else: a header record that
runs off the end of the file, and a receiver position handed in by the
caller that the tide correction wrote back into.
"""

import os
import tempfile

import numpy as np
import pytest

import cssrlib.rinex as rn
from cssrlib.gnss import uTideModel
from cssrlib.models.ephemeris import satposs
from cssrlib.test.golden_harness import setup

TRUNCATED = [
    "     3.04           OBSERVATION DATA    M                   "
    "RINEX VERSION / TYPE",
    "TRUNC               cssrlib test        20210319 000000 UTC "
    "PGM / RUN BY / DATE ",
    "G   20 C1C L1C S1C C2W L2W S2W C5Q L5Q S5Q C1W L1W S1W C2L  "
    "SYS / # / OBS TYPES ",
]


def test_a_header_that_stops_mid_record_raises_instead_of_hanging():
    """SYS / # / OBS TYPES continues onto further lines until nsig signals
    have been read. At EOF readline() returns '' forever and the count never
    advances, so the reader spun until it was killed."""
    path = os.path.join(tempfile.mkdtemp(), "truncated.obs")
    with open(path, "w") as fh:
        fh.write("\n".join(TRUNCATED) + "\n")

    with pytest.raises(ValueError, match="ends mid-record"):
        rn.rnxdec().decode_obsh(path)


@pytest.mark.parametrize("as_array", [True, False], ids=["ndarray", "list"])
def test_qcedit_does_not_move_the_position_it_was_given(as_array):
    """qcedit adds the solid-earth tide to the receiver position.

    It used to add it to the caller's array. single_differences passes
    nav.rb, so with tidecorr enabled the base station coordinate accumulated
    a tidal displacement every epoch -- 0.13 m after three of them, growing
    without bound. Held as a list it escaped by accident, because numpy
    rebinds instead of extending; that is not a guarantee to rely on.
    """
    dec, decb, rtk = setup()
    rtk.nav.tidecorr = uTideModel.SIMPLE
    rtk.base_nav.tidecorr = uTideModel.SIMPLE

    rb = (np.array(rtk.base_nav.rb, dtype=float) if as_array
          else list(rtk.base_nav.rb))
    rtk.base_nav.rb = rb
    before = [float(v) for v in np.ravel(rb)]

    for _ in range(3):
        obs = dec.decode_obs()
        obsb = decb.decode_obs()
        rs, vs, dts, svh, _ = satposs(obs, rtk.nav)
        rtk.single_differences(obs, obsb, rs, dts, svh)

    after = [float(v) for v in np.ravel(rtk.base_nav.rb)]
    assert len(after) == 3, f"the base coordinate grew to {len(after)} elements"
    assert after == before, (
        f"base coordinate moved {max(abs(a - b) for a, b in zip(after, before)):.6f} m")
