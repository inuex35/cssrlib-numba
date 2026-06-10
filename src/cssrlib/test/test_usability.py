"""
Regression tests for library-usability fixes aimed at embedding cssrlib in
external estimators (e.g. GTSAM factor-graph RTK):

  * mlambda raises a catchable Exception, not SystemExit
  * qcedit tolerates a constellation with fewer bands than nav.nf
  * prepare_double_difference_measurements returns a dict that also supports
    attribute access
"""

import numpy as np

from cssrlib.mlambda import mlambda, ldldecom, LambdaError
from cssrlib.gnss import Nav, Obs, rSigRnx, uGNSS, uTYP
from cssrlib.pppssr import _qcedit_system_cache, _sig_label
from cssrlib.rtk import DDMeasurements


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


def test_qcedit_cache_handles_short_band_system():
    """A system with fewer bands than nav.nf must not raise IndexError."""
    nav = Nav(nf=2)  # dual-frequency setup
    obs = Obs()
    obs.sig = {
        uGNSS.GPS: {uTYP.C: [rSigRnx("GC1C"), rSigRnx("GC2W")],
                    uTYP.L: [rSigRnx("GL1C"), rSigRnx("GL2W")],
                    uTYP.S: [rSigRnx("GS1C"), rSigRnx("GS2W")]},
        uGNSS.GLO: {uTYP.C: [rSigRnx("RC1C")],   # single band < nf
                    uTYP.L: [rSigRnx("RL1C")],
                    uTYP.S: [rSigRnx("RS1C")]},
    }

    cache = _qcedit_system_cache(obs, nav)

    # cnr_thresholds always has length nf, regardless of available bands.
    assert cache[uGNSS.GPS][3].shape == (2,)
    assert cache[uGNSS.GLO][3].shape == (2,)
    # Out-of-range band label falls back instead of raising.
    assert _sig_label(cache[uGNSS.GLO][2], 1) == "f1"
    assert _sig_label(cache[uGNSS.GPS][2], 1) == "S2W"


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


if __name__ == "__main__":
    test_mlambda_raises_catchable_exception()
    test_qcedit_cache_handles_short_band_system()
    test_dd_measurements_dual_access()
    print("OK")
