"""Editing is per band, and it does not write on the caller's observations.

qcedit records a verdict per band -- missing PR, missing CP, low C/N0 -- and
then used to collapse it: ``if np.any(edt[i, :nf_sys]): edt[i, :] = 1`` and
drop the satellite. Every edt row was therefore uniformly 0 or 1 and the
per-band tests were decorative. On the bundled dataset that discarded G01's
36 dB-Hz L1 in 20 satellite-epochs out of 250, because G01's L2 sits at
14 dB-Hz.

The mask is also no longer applied by writing zeros into ``obs`` / ``obsb``.
Those belong to the caller, and ``sync_obs_hold`` hands the same base record
back for several rover epochs by design.
"""

import copy

import numpy as np
import pytest

from cssrlib.gnss import sat2id, sat2prn, uTYP
from cssrlib.models.ephemeris import satposs
from cssrlib.test.golden_harness import setup

# G01 carries a healthy L1 and an unusable L2 throughout the bundled file.
SPLIT_BAND_SAT = "G01"


def first_epoch():
    dec, decb, rtk = setup()
    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    return dec, decb, rtk, obs, obsb


def test_the_dataset_still_has_a_split_band_satellite():
    """Guard the fixture: one good band, one band below cnr_min."""
    _, _, rtk, obs, _ = first_epoch()
    i = next(k for k, s in enumerate(obs.sat) if sat2id(s) == SPLIT_BAND_SAT)

    assert obs.S[i, 0] >= rtk.nav.cnr_min, "L1 should be healthy"
    assert obs.S[i, 1] < rtk.nav.cnr_min, (
        f"{SPLIT_BAND_SAT} L2 is no longer below cnr_min; this file no "
        f"longer exercises per-band editing")


def test_qcedit_keeps_the_satellite_for_its_good_band():
    _, _, rtk, obs, _ = first_epoch()
    rs, vs, dts, svh, _ = satposs(obs, rtk.nav)

    sat_ed = rtk.qcedit(obs, rs, dts, svh)

    sat_no = next(int(s) for s in obs.sat if sat2id(s) == SPLIT_BAND_SAT)
    assert sat_no in sat_ed, (
        f"{SPLIT_BAND_SAT} was dropped although its L1 is usable -- the "
        f"all-or-nothing gate is back")

    row = rtk.nav.rcv.edt[sat_no - 1]
    assert row[0] == 0, "the good band was edited out with the bad one"
    assert row[1] != 0, "the band below cnr_min was not edited"


def test_some_edt_row_is_genuinely_mixed():
    """The property the collapse destroyed: rows need not be uniform."""
    _, _, rtk, obs, _ = first_epoch()
    rs, vs, dts, svh, _ = satposs(obs, rtk.nav)
    rtk.qcedit(obs, rs, dts, svh)

    rows = rtk.nav.rcv.edt[np.asarray(obs.sat) - 1, :rtk.nav.nf]
    mixed = [(sat2id(obs.sat[k]), rows[k].tolist())
             for k in range(len(obs.sat))
             if 0 < int(rows[k].sum()) < rows.shape[1]]

    assert mixed, (
        "every edt row is uniform; per-band editing is not reaching the mask")


def test_single_differences_does_not_touch_the_caller_observations():
    dec, decb, rtk = setup()
    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    rs, vs, dts, svh, _ = satposs(obs, rtk.nav)

    before = (obs.L.copy(), obs.P.copy(), obsb.L.copy(), obsb.P.copy())
    rtk.single_differences(obs, obsb, rs, dts, svh)

    assert np.array_equal(obs.L, before[0]), "rover L was rewritten"
    assert np.array_equal(obs.P, before[1]), "rover P was rewritten"
    assert np.array_equal(obsb.L, before[2]), "base L was rewritten"
    assert np.array_equal(obsb.P, before[3]), "base P was rewritten"


def test_a_held_base_record_survives_being_reused():
    """sync_obs_hold's documented case: one base record, several rover epochs."""
    dec, decb, rtk = setup()
    dec.decode_obs()
    obsb = decb.decode_obs()
    pristine = copy.deepcopy(obsb)

    counts = []
    for _ in range(5):
        obs = dec.decode_obs()
        rs, vs, dts, svh, _ = satposs(obs, rtk.nav)
        iu, _ = rtk.single_differences(obs, obsb, rs, dts, svh)
        counts.append(len(iu))
        assert np.array_equal(obsb.L, pristine.L), (
            "the held base record was degraded by being used")
        assert np.array_equal(obsb.P, pristine.P)

    assert len(set(counts)) == 1 or counts == sorted(counts, reverse=True), (
        f"satellite count wandered across reuses of one base record: {counts}")


def test_dd_exposes_the_masks_that_obs_sd_already_applied():
    """A caller reading raw obs/obsb must be able to reproduce the editing."""
    dec, decb, rtk = setup()
    obs = dec.decode_obs()
    obsb = decb.decode_obs()

    dd = rtk.prepare_double_difference_measurements(obs, obsb)
    assert dd is not None

    sat, iu, ir = dd['sat'], dd['iu'], dd['ir']
    assert dd['edt'].shape == (len(sat), rtk.nav.nf)
    assert dd['edtb'].shape == (len(sat), rtk.nav.nf)

    masked = dd['edt'] | dd['edtb']
    nf = min(rtk.nav.nf, dd['obs_sd'].L.shape[1])
    for k in range(len(sat)):
        for f in range(nf):
            if masked[k, f]:
                assert dd['obs_sd'].L[k, f] == 0.0, (
                    f"{sat2id(sat[k])} band {f} is masked but survived in "
                    f"obs_sd")

    # And the split-band satellite is present with exactly one band.
    row = next(k for k in range(len(sat))
               if sat2id(sat[k]) == SPLIT_BAND_SAT)
    assert not masked[row, 0] and masked[row, 1]
