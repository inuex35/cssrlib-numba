"""Editing is recorded per band, admitted per satellite, and never written
back onto the caller's observations.

qcedit's per-band tests -- missing PR, missing CP, low C/N0 -- record a
verdict per band, and the strict gate then drops any satellite with an
edited band among the ones its system selected. The gate was relaxed to
per-band admission once and measured to be wrong twice: in 2026-07
accepting L5-less GPS / B1I-only BeiDou-2 poisoned the urban float
solution, and in 2026-08 the relaxed gate collapsed the tokyo run2
tightly-coupled pipeline (284 fixes in 300 epochs to 0; float RMS 0.13 m to
5.2 m), with the gate's restoration alone recovering the baseline
print-identically. These tests pin the strict policy on the bundled
split-band satellite so the relaxation cannot come back quietly.

What did survive from that episode: the masks are applied to local copies
rather than written into ``obs`` / ``obsb`` (``sync_obs_hold`` reuses one
base record across rover epochs by design), and ``DDMeasurements`` carries
``edt`` / ``edtb`` so a consumer reading the raw arrays can see the same
verdicts ``obs_sd`` already has applied.
"""

import copy

import numpy as np

from cssrlib.gnss import sat2id
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
        f"longer exercises the split-band gate")


def test_one_bad_selected_band_drops_the_satellite():
    """The strict gate: a split-band satellite does not survive."""
    _, _, rtk, obs, _ = first_epoch()
    rs, vs, dts, svh, _ = satposs(obs, rtk.nav)

    sat_ed = rtk.qcedit(obs, rs, dts, svh)

    sat_no = next(int(s) for s in obs.sat if sat2id(s) == SPLIT_BAND_SAT)
    assert sat_no not in sat_ed, (
        f"{SPLIT_BAND_SAT} survived with a band below cnr_min -- the "
        f"per-band admission relaxation is back, which collapsed the "
        f"tokyo tightly-coupled pipeline when it was measured")

    # And the whole row is marked, so every per-band consumer skips it.
    assert np.all(rtk.nav.rcv.edt[sat_no - 1] > 0)


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


def test_dd_exposes_masks_consistent_with_obs_sd():
    """edt / edtb describe exactly the bands obs_sd carries as zero.

    Under the strict gate a surviving satellite has no edited selected
    band, so for the bundled file the masks are clean on the carried bands
    -- and the split-band satellite is simply absent.
    """
    dec, decb, rtk = setup()
    obs = dec.decode_obs()
    obsb = decb.decode_obs()

    dd = rtk.prepare_double_difference_measurements(obs, obsb)
    assert dd is not None

    sat = dd['sat']
    assert dd['edt'].shape == (len(sat), rtk.nav.nf)
    assert dd['edtb'].shape == (len(sat), rtk.nav.nf)
    assert SPLIT_BAND_SAT not in {sat2id(s) for s in sat}

    masked = dd['edt'] | dd['edtb']
    nf = min(rtk.nav.nf, dd['obs_sd'].L.shape[1])
    for k in range(len(sat)):
        for f in range(nf):
            if masked[k, f]:
                assert dd['obs_sd'].L[k, f] == 0.0, (
                    f"{sat2id(sat[k])} band {f} is masked but survived in "
                    f"obs_sd")
