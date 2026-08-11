"""The base receiver has its own state instead of its own engine.

rtkpos used to deepcopy the whole Nav to represent the base -- with a
detach-and-restore dance around the log handle, because deepcopy cannot
pickle an open file -- and then swap self.nav in and out via a _use_nav
context manager so the engine could edit base observations while pretending
to be the base. The base now owns a ReceiverState, passed to qcedit as an
argument.
"""

import numpy as np
import pytest

from cssrlib.gnss import Nav, ReceiverState
from cssrlib.engine.rtk import rtkpos
from cssrlib.test.golden_harness import setup


def test_engine_no_longer_swaps_its_own_nav():
    assert not hasattr(rtkpos, "_use_nav"), (
        "_use_nav is back: the engine is impersonating the base again")


def test_base_receiver_state_is_separate_storage():
    rtk = rtkpos(Nav(nf=2), np.zeros(3))

    assert isinstance(rtk.base_rcv, ReceiverState)
    assert rtk.base_rcv is not rtk.nav.rcv

    rtk.base_rcv.edt[3, 0] = 1
    rtk.base_rcv.gf[3] = 2.5
    assert rtk.nav.rcv.edt[3, 0] == 0
    assert rtk.nav.rcv.gf[3] == 0.0


def test_navigation_data_and_config_are_shared_not_copied():
    """No deepcopy in the default path -- both receivers read one table."""
    rtk = rtkpos(Nav(nf=2), np.zeros(3))

    assert rtk.base_nav is rtk.nav
    assert rtk.base_nav.data is rtk.nav.data
    assert rtk.base_nav.cfg is rtk.nav.cfg

    # A GLONASS channel learned while processing either receiver is visible
    # to both. It used to land only in the base's private copy.
    rtk.nav.glo_ch[7] = -3
    assert rtk.base_nav.glo_ch[7] == -3


def test_qcedit_records_into_the_receiver_it_is_given():
    dec, decb, rtk = setup()
    from cssrlib.models.ephemeris import satposs

    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    rs, _, dts, svh, _ = satposs(obs, rtk.nav)
    rsb, _, dtsb, svhb, _ = satposs(obsb, rtk.base_nav)

    rtk.qcedit(obsb, rsb, dtsb, svhb, rr=rtk.nav.rb, rcv=rtk.base_rcv)
    base_edt = rtk.base_rcv.edt.copy()
    assert rtk.nav.rcv.edt.sum() == 0, "base edits leaked into the rover"

    rtk.qcedit(obs, rs, dts, svh, rcv=rtk.nav.rcv)
    assert np.array_equal(rtk.base_rcv.edt, base_edt), (
        "rover edits overwrote the base's results")
    assert rtk.nav.rcv.edt.sum() > 0, "the rover recorded nothing"


def test_single_differences_keeps_the_two_receivers_apart():
    dec, decb, rtk = setup()
    obs = dec.decode_obs()
    obsb = decb.decode_obs()

    dd = rtk.prepare_double_difference_measurements(obs, obsb)
    assert dd is not None and len(dd["sat"]) >= 4

    # Both receivers were edited, into their own tables.
    assert rtk.nav.rcv.edt.sum() > 0
    assert rtk.base_rcv.edt.sum() > 0
    assert rtk.base_rcv is not rtk.nav.rcv


def test_base_nav_override_still_gets_its_own_data():
    """The opt-in path keeps working, without copying the whole Nav."""
    other = Nav(nf=2)
    other.leaps = 37
    other.cnr_min = 40

    rtk = rtkpos(Nav(nf=2), np.zeros(3), base_nav=other)

    assert rtk.base_nav is not rtk.nav
    assert rtk.base_nav.data is not rtk.nav.data
    assert rtk.base_nav.leaps == 37
    assert rtk.base_nav.cnr_min == 40
    # The rover keeps its own values.
    assert rtk.nav.leaps == 18
    # And the override shares the base's receiver state.
    assert rtk.base_nav.rcv is rtk.base_rcv


def test_no_open_file_handle_is_deepcopied():
    """The old code had to detach nav.fout before deepcopy could run."""
    import tempfile
    import os

    fd, path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    try:
        rtk = rtkpos(Nav(nf=2), np.zeros(3), logfile=path)
        assert rtk.nav.fout is not None
        # Constructing the base did not require touching the handle.
        assert rtk.base_rcv is not None
        rtk.nav.fout.close()
    finally:
        os.unlink(path)
