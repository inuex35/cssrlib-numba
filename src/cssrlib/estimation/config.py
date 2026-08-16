"""Processing configurations.

An RTK session and a PPP-RTK session differ in *settings*, not in kind:
which corrections apply, which states are estimated, how aggressive the
ambiguity resolver is. That difference used to be expressed by subclassing
the engine, which is what let ``rtkpos`` override ``base_process`` with a
return contract its parent did not expect -- a Liskov violation that showed
up as a singular matrix rather than as a type error.

Each factory here returns a fully populated :class:`~cssrlib.gnss.ProcConfig`.
Pass one to the engine instead of picking a subclass.

    nav = Nav(nf=2)
    engine = rtkpos(nav, pos0, cfg=rtk_config(nf=2))

Keyword overrides are applied last, so a caller can adjust a single field
without copying the whole block::

    rtk_config(nf=2, elmin=np.deg2rad(10.0))
"""

import numpy as np

from cssrlib.gnss import ProcConfig, uIonoModel, uTropoModel, uTideModel


def _apply(cfg, overrides):
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(
                f"ProcConfig has no field {key!r}; check the spelling, or add "
                f"it to ProcConfig if it is genuinely new configuration")
        setattr(cfg, key, value)
    return cfg


def base_config(nf=2, pmode=1, **overrides):
    """The engine's own defaults, shared by every mode.

    These are the values ``gnssobs.__init__`` used to assign to ``nav``
    directly, before any subclass got a chance to change them.
    """
    cfg = ProcConfig(nf=nf)

    cfg.pmode = pmode
    cfg.ephopt = 2                       # SSR-APC
    cfg.trpModel = uTropoModel.SAAST
    cfg.ionoModel = uIonoModel.KLOBUCHAR
    cfg.csmooth = False

    cfg.tidecorr = uTideModel.IERS2010
    cfg.thresar = 3.0
    cfg.armode = 0
    cfg.elmaskar = np.deg2rad(20.0)
    cfg.elmin = np.deg2rad(15.0)
    cfg.parmode = 2                      # 1: full ILS, 2: partial AR
    cfg.par_P0 = 0.995

    # Position process noise depends on whether the receiver is static.
    if pmode == 0:
        cfg.sig_qp = 100.0 / np.sqrt(1)
        cfg.sig_qv = None
    else:
        cfg.sig_qp = 0.01 / np.sqrt(1)
        cfg.sig_qv = 1.0 / np.sqrt(1)

    return _apply(cfg, overrides)


def ppp_config(nf=2, pmode=1, **overrides):
    """Global PPP: estimate the troposphere and the ionosphere."""
    cfg = base_config(nf=nf, pmode=pmode)
    cfg.trop_opt = 1     # estimate
    cfg.iono_opt = 1     # estimate
    cfg.phw_opt = 1      # full phase wind-up model
    return _apply(cfg, overrides)


def ppprtk_config(nf=2, pmode=1, **overrides):
    """PPP-RTK (QZSS CLAS): troposphere and ionosphere come from the SSR."""
    cfg = base_config(nf=nf, pmode=pmode)
    cfg.trop_opt = 2     # from cssr
    cfg.iono_opt = 2     # from cssr
    cfg.phw_opt = 2      # local/regional wind-up model

    cfg.eratio = np.ones(nf) * 50
    cfg.err = [0, 0.01, 0.005] / np.sqrt(2)
    cfg.thresar = 2.0
    cfg.armode = 3       # fix and hold
    return _apply(cfg, overrides)


def rtk_config(nf=2, pmode=1, **overrides):
    """Short-baseline RTK against a base station.

    The atmosphere and phase wind-up cancel in the rover-base double
    difference, so none of them are modelled, and solid Earth tides are
    switched off for the same reason.
    """
    cfg = base_config(nf=nf, pmode=pmode)
    cfg.trop_opt = 0
    cfg.iono_opt = 0
    cfg.phw_opt = 0

    cfg.eratio = np.ones(nf) * 50
    cfg.err = [0, 0.01, 0.005] / np.sqrt(2)
    cfg.thresar = 2.0
    cfg.armode = 1       # continuous

    cfg.tidecorr = uTideModel.NONE

    # RTKLIB-compatible ambiguity-resolution extras.
    cfg.maxtdiff = 30.0      # [s] max age of base observations
    cfg.rtklib_mode = False
    cfg.arfilter = True
    cfg.minfixsats = 4
    return _apply(cfg, overrides)
