"""
Regression tests for library-usability fixes aimed at embedding cssrlib in
external estimators (e.g. GTSAM factor-graph RTK):

  * mlambda raises a catchable Exception, not SystemExit
  * qcedit tolerates a constellation with fewer bands than nav.nf
  * prepare_double_difference_measurements returns a dict that also supports
    attribute access
"""

import numpy as np

from cssrlib.core.mlambda import mlambda, ldldecom, LambdaError
from cssrlib.gnss import Nav, Obs, rSigRnx, uGNSS, uTYP, prn2sat, gpst2time
from cssrlib.engine.rtk import DDMeasurements, rtkpos


def test_mlambda_raises_catchable_exception():
    """Non positive-definite covariance -> LambdaError, never SystemExit."""
    Q = np.array([[1.0, 0.0], [0.0, -1.0]])  # not positive definite

    try:
        ldldecom(Q)
        raised = None
    except BaseException as exc:  # noqa: BLE001 - intentional broad catch
        raised = exc
    assert isinstance(raised, LambdaError)
    assert not isinstance(raised, SystemExit)

    # Catchable as a plain Exception (and as LinAlgError).
    try:
        mlambda(np.zeros(2), Q)
    except Exception as exc:  # noqa: BLE001
        assert isinstance(exc, np.linalg.LinAlgError)
    else:
        raise AssertionError("mlambda should have raised")


def test_qcedit_handles_short_band_system():
    """A system with fewer bands than nav.nf must not raise IndexError.

    GLONASS here carries a single band in an nf=2 setup, so the frequency
    loop runs past the end of its signal lists. qcedit must treat the
    missing slot as absent rather than indexing off the end, and must not
    punish those satellites for a band their system never selected.
    """
    nav = Nav(nf=2)  # dual-frequency setup
    obs = Obs()
    obs.t = gpst2time(2148, 259200.0)
    nav.t = obs.t
    obs.sig = {
        uGNSS.GPS: {uTYP.C: [rSigRnx("GC1C"), rSigRnx("GC2W")],
                    uTYP.L: [rSigRnx("GL1C"), rSigRnx("GL2W")],
                    uTYP.S: [rSigRnx("GS1C"), rSigRnx("GS2W")]},
        uGNSS.GLO: {uTYP.C: [rSigRnx("RC1C")],   # single band < nf
                    uTYP.L: [rSigRnx("RL1C")],
                    uTYP.S: [rSigRnx("RS1C")]},
    }

    sats = [prn2sat(uGNSS.GPS, 1), prn2sat(uGNSS.GPS, 2),
            prn2sat(uGNSS.GLO, 1), prn2sat(uGNSS.GLO, 2)]
    obs.sat = np.array(sats, dtype=int)
    n = len(sats)
    obs.P = np.full((n, 2), 2.2e7)
    obs.L = np.full((n, 2), 1.1e8)
    obs.S = np.full((n, 2), 45.0)
    obs.lli = np.zeros((n, 2), dtype=int)
    # GLONASS carries one band only.
    obs.P[2:, 1] = 0.0
    obs.L[2:, 1] = 0.0
    obs.S[2:, 1] = 0.0

    rr = np.array([-3962108.7, 3381309.5, 3668678.6])
    rtk = rtkpos(nav, rr)   # allocates nav.x / nav.edt / nav.el
    nav.x[0:3] = rr
    nav.t = obs.t

    # Satellites straight overhead, so nothing is cut by the elevation mask.
    rs = np.tile(rr / np.linalg.norm(rr) * 2.0e7, (n, 1))
    dts = np.zeros(n)
    svh = np.zeros(n, dtype=int)

    sat_ed = rtk.qcedit(obs, rs, dts, svh)     # must not raise IndexError

    assert nav.edt.shape[1] == 2
    # The single-band GLONASS satellites survive on their one selected band.
    for s in sats[2:]:
        assert s in sat_ed, "single-band GLO satellite was dropped"
        assert nav.edt[s - 1, 0] == 0


def test_dd_measurements_dual_access():
    """DDMeasurements behaves as a dict and supports attribute access."""
    dd = DDMeasurements({"rs": np.zeros((3, 3)),
                         "iu": np.array([1, 2]),
                         "pos_pred": np.ones(3)})

    assert isinstance(dd, dict)
    assert dd["iu"].tolist() == [1, 2]      # dict access
    assert dd.iu.tolist() == [1, 2]         # attribute access
    assert dd.get("missing", 42) == 42
    assert "rs" in dd

    try:
        dd.nope
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown field should raise AttributeError")


def test_auto_detect_signals():
    """auto_detect_signals builds matching rover/base lists from headers."""
    import os
    import cssrlib.rinex as rn
    from cssrlib.rinex import auto_detect_signals

    data = os.path.join(os.path.dirname(__file__), "..", "data") + os.sep
    dec = rn.rnxdec()
    dec.decode_obsh(data + "SEPT078M1.21O")
    decb = rn.rnxdec()
    decb.decode_obsh(data + "3034078M1.21O")

    sigs, sigsb = auto_detect_signals(dec.sig_map, decb.sig_map, max_freq=2)
    assert len(sigs) > 0 and len(sigs) == len(sigsb)
    # Every band carries pseudorange + carrier + SNR (C, L, S).
    assert len(sigs) % 3 == 0
    types = {s.typ for s in sigs}
    assert {uTYP.C, uTYP.L, uTYP.S} <= types

    # Single-receiver form: no base -> empty second list.
    only, none = auto_detect_signals(dec.sig_map, max_freq=2)
    assert len(only) > 0 and none == []

    # rnxdec.autoSignals one-liner applies them to both decoders.
    d2 = rn.rnxdec(); d2.decode_obsh(data + "SEPT078M1.21O")
    db2 = rn.rnxdec(); db2.decode_obsh(data + "3034078M1.21O")
    s2, sb2 = d2.autoSignals(db2, max_freq=2)
    assert s2 == sigs and d2.sig_tab and db2.sig_tab


if __name__ == "__main__":
    test_mlambda_raises_catchable_exception()
    test_qcedit_handles_short_band_system()
    test_dd_measurements_dual_access()
    test_auto_detect_signals()
    print("OK")
