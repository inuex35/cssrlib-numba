"""Configuration factories replace configuration-by-subclassing.

rtkpos and ppprtkpos used to differ only in the values they assigned to nav
after calling super().__init__. Expressing that as inheritance is what let
rtkpos override base_process with a return contract gnssobs.process did not
expect -- a Liskov violation that surfaced as a singular matrix rather than
a type error.
"""

import numpy as np
import pytest

from cssrlib.config import (base_config, ppp_config, ppprtk_config,
                            rtk_config)
from cssrlib.gnss import Nav, ProcConfig, uTideModel
from cssrlib.ppprtk import ppprtkpos
from cssrlib.rtk import rtkpos

FACTORIES = (base_config, ppp_config, ppprtk_config, rtk_config)


@pytest.mark.parametrize("factory", FACTORIES,
                         ids=lambda f: f.__name__)
def test_factory_returns_a_complete_config(factory):
    cfg = factory(nf=2)
    assert isinstance(cfg, ProcConfig)
    # Every field a run needs is populated, none left as None by accident.
    for field in ("nf", "pmode", "elmin", "elmaskar", "armode", "parmode",
                  "thresar", "trop_opt", "iono_opt", "phw_opt", "eratio",
                  "err", "sig_p0", "sig_qp", "sig_qb"):
        assert getattr(cfg, field) is not None, f"{field} unset"


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_nf_flows_into_the_error_budget(factory):
    for nf in (1, 2, 3):
        cfg = factory(nf=nf)
        assert cfg.nf == nf
        assert cfg.eratio.shape == (nf,)


def test_rtk_and_ppprtk_differ_only_where_documented():
    """The whole reason the two classes existed, as a diff."""
    rtk = rtk_config(nf=2)
    ppprtk = ppprtk_config(nf=2)

    differing = set()
    for field in vars(rtk):
        a, b = getattr(rtk, field), getattr(ppprtk, field)
        try:
            same = bool(np.all(a == b))
        except Exception:
            same = a is b
        if not same:
            differing.add(field)

    assert differing == {
        "trop_opt",     # 0 (cancels in DD) vs 2 (from cssr)
        "iono_opt",     # 0 vs 2
        "phw_opt",      # 0 vs 2
        "armode",       # 1 continuous vs 3 fix-and-hold
        "tidecorr",     # NONE vs IERS2010
    }, f"unexpected difference set: {sorted(differing)}"


def test_rtk_config_disables_what_cancels_in_a_double_difference():
    cfg = rtk_config(nf=2)
    assert cfg.trop_opt == 0 and cfg.iono_opt == 0 and cfg.phw_opt == 0
    assert cfg.tidecorr == uTideModel.NONE


def test_ppprtk_config_takes_the_atmosphere_from_the_ssr():
    cfg = ppprtk_config(nf=2)
    assert cfg.trop_opt == 2 and cfg.iono_opt == 2
    assert cfg.armode == 3


def test_ppp_config_estimates_the_atmosphere():
    cfg = ppp_config(nf=2)
    assert cfg.trop_opt == 1 and cfg.iono_opt == 1


def test_static_mode_gets_its_own_position_process_noise():
    kinematic = rtk_config(nf=2, pmode=1)
    static = rtk_config(nf=2, pmode=0)
    assert static.sig_qp > kinematic.sig_qp
    assert static.sig_qv is None and kinematic.sig_qv is not None


def test_overrides_are_applied_last():
    cfg = rtk_config(nf=2, elmin=np.deg2rad(5.0), armode=0)
    assert cfg.elmin == pytest.approx(np.deg2rad(5.0))
    assert cfg.armode == 0


def test_unknown_override_is_rejected():
    with pytest.raises(AttributeError, match="no field"):
        rtk_config(nf=2, elmni=0.1)      # typo


@pytest.mark.parametrize("cls,factory", [(rtkpos, rtk_config),
                                         (ppprtkpos, ppprtk_config)],
                         ids=["rtkpos", "ppprtkpos"])
def test_engine_uses_its_factory_by_default(cls, factory):
    engine = cls(Nav(nf=2), np.zeros(3))
    expected = factory(nf=2)
    for field in ("trop_opt", "iono_opt", "phw_opt", "armode", "thresar",
                  "tidecorr"):
        assert getattr(engine.nav, field) == getattr(expected, field)


def test_a_custom_config_is_honoured():
    engine = rtkpos(Nav(nf=2), np.zeros(3),
                    cfg=rtk_config(nf=2, armode=0, thresar=9.0))
    assert engine.nav.armode == 0
    assert engine.nav.thresar == 9.0


def test_caller_configuration_survives_the_factory():
    """nav.rb is set before the engine is built; it must not be discarded.

    Replacing nav.cfg outright wiped it, which showed up as a division by
    zero in ecef2pos when qcedit was handed a base position of [0, 0, 0].
    """
    nav = Nav(nf=2)
    nav.rb = [-3959400.631, 3385704.533, 3667523.111]
    nav.excl_sat = [5]
    nav.elmin = np.deg2rad(7.5)

    engine = rtkpos(nav, np.zeros(3))

    assert engine.nav.rb == [-3959400.631, 3385704.533, 3667523.111]
    assert engine.nav.excl_sat == [5]
    assert engine.nav.elmin == pytest.approx(np.deg2rad(7.5))
    # Fields the caller left alone still come from the factory.
    assert engine.nav.armode == rtk_config(nf=2).armode


def test_initial_position_variance_matches_the_configured_sigma():
    """sig_p0 now reports the value P was actually built from.

    rtkpos and ppprtkpos used to assign sig_p0 = 30.0 *after*
    gnssobs.__init__ had already set P from the base value of 100.0, and
    nothing read sig_p0 again -- so the 30 never took effect and the filter
    started with a 100 m position sigma while the source said 30.
    """
    for cls in (rtkpos, ppprtkpos):
        engine = cls(Nav(nf=2), np.zeros(3))
        assert engine.nav.sig_p0 == 100.0
        assert engine.nav.P[0, 0] == pytest.approx(engine.nav.sig_p0 ** 2)
