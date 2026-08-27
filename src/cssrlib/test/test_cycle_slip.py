"""The cycle-slip flag, and the filter that must honour it.

qcedit stopped raising ``edt`` for LLI / GF cycle slips -- an LLI is a
slip notification, not a bad observation -- and started recording
``ReceiverState.slip`` instead, whose docstring says "Cleared by udstate
after the reset is applied". udstate never read it: the word ``slip`` did
not appear in the filter at all, so on the PPP / PPP-RTK path a slipped
satellite kept its pre-slip ambiguity.

The bundled base station supplies the real event: every satellite in
``3034078M1.21O`` carries LLI=1 at 12:00:18.
"""

import numpy as np
import pytest

from cssrlib.gnss import uGNSS
from cssrlib.models.ephemeris import satposs
from cssrlib.test.golden_harness import setup

# Epoch index (0-based, 1 Hz from 12:00:00) at which the base receiver
# reports loss of lock on every satellite and both bands.
BASE_LLI_EPOCH = 18


def test_the_bundled_base_really_does_report_loss_of_lock():
    """Guard the fixture the two slip tests below are built on."""
    dec, decb, _ = setup()
    seen = {}
    for ep in range(BASE_LLI_EPOCH + 2):
        dec.decode_obs()
        obsb = decb.decode_obs()
        lli = np.asarray(obsb.lli)
        if lli.any():
            seen[ep] = int(np.count_nonzero(lli))

    assert BASE_LLI_EPOCH in seen, (
        f"base LLI epoch moved; nonzero LLI seen at {sorted(seen)}")
    assert seen[BASE_LLI_EPOCH] > 20, seen


def drive_to(rtk, dec, decb, stop):
    """Run the PPP driver's per-epoch sequence up to (not including) ``stop``.

    Returns the epoch's ``(obs_, slipped)`` at ``stop`` with udstate not yet
    called, so a test can intervene between qcedit and the filter.
    """
    for ep in range(stop + 1):
        obs = dec.decode_obs()
        obsb = decb.decode_obs()
        rs, vs, dts, svh, nsat = satposs(obs, rtk.nav)
        if nsat < 6:
            continue
        rtk.qcedit(obs, rs, dts, svh)
        iu, obs_ = rtk.single_differences(obs, obsb, rs, dts, svh)
        if len(iu) < 6:
            continue
        if ep == stop:
            return obs_, np.argwhere(rtk.nav.slip > 0)
        rtk.udstate(obs_)
    pytest.fail(f"never reached epoch {stop}")


def slipped_with_an_ambiguity(rtk, obs_, slipped):
    """The (sat, band, index) triples the reset can actually be seen on."""
    present = {int(s) for s in obs_.sat}
    out = []
    for i, f in slipped:
        sat_ = int(i) + 1
        if sat_ not in present:
            continue
        j = rtk.IB(sat_, int(f), rtk.nav.na)
        if rtk.nav.x[j] != 0.0:
            out.append((sat_, int(f), j))
    return out


def test_udstate_resets_the_ambiguity_on_a_cycle_slip():
    """A slip flag must clear the ambiguity, and be consumed doing it."""
    dec, decb, rtk = setup()
    obs_, slipped = drive_to(rtk, dec, decb, BASE_LLI_EPOCH)

    assert slipped.size > 0, "the LLI epoch raised no slip flag"
    targets = slipped_with_an_ambiguity(rtk, obs_, slipped)
    assert targets, "test is vacuous: no slipped satellite carried an ambiguity"

    before = {j: rtk.nav.x[j] for _, _, j in targets}

    rtk.udstate(obs_)

    # udstate resets and re-initializes in the same pass, so the ambiguity
    # is not left at zero -- it is replaced by a fresh (cp - pr/lam) seed.
    # What must not happen is the pre-slip value surviving untouched.
    for sat_, band, j in targets:
        assert rtk.nav.x[j] != before[j], (
            f"udstate ignored nav.slip and carried sat {sat_} band {band}'s "
            f"pre-slip ambiguity ({before[j]:.3f} cyc) through the slip")
        assert rtk.nav.slip[sat_ - 1, band] == 0, (
            f"sat {sat_} band {band}: slip flag not consumed; it would reset "
            f"again on every following epoch")


def test_the_slip_flag_is_what_causes_that_reset():
    """Control for the test above.

    udstate resets on outage and on edt as well, and this harness drives it
    without ``process`` (which is what clears outc), so "the ambiguity is
    zero afterwards" alone does not implicate the slip. Two runs reach the
    same epoch identically; one has its slip flags cleared just before the
    filter. The difference is the slip and nothing else.
    """
    dec_a, decb_a, rtk_a = setup()
    obs_a, slipped = drive_to(rtk_a, dec_a, decb_a, BASE_LLI_EPOCH)
    targets = slipped_with_an_ambiguity(rtk_a, obs_a, slipped)
    assert targets

    dec_b, decb_b, rtk_b = setup()
    obs_b, _ = drive_to(rtk_b, dec_b, decb_b, BASE_LLI_EPOCH)
    assert np.array_equal(rtk_a.nav.x, rtk_b.nav.x), "runs diverged"

    before = {j: rtk_a.nav.x[j] for _, _, j in targets}

    rtk_b.nav.slip[:, :] = 0          # the only difference
    rtk_a.udstate(obs_a)
    rtk_b.udstate(obs_b)

    for sat_, band, j in targets:
        assert rtk_a.nav.x[j] != before[j], (
            f"sat {sat_} band {band}: the slipped run did not re-seed")
        assert rtk_b.nav.x[j] == before[j], (
            f"sat {sat_} band {band} was re-seeded even with the slip flags "
            f"cleared -- this test is not measuring the slip path")
        assert rtk_a.nav.x[j] != rtk_b.nav.x[j]
