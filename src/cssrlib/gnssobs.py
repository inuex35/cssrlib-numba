"""The GNSS positioning engine.

Composition point. The engine's behaviour lives in four mixins, one per
concern, so that changing the observation model does not mean reading the
ambiguity resolver:

    cssrlib.qc          quality control
    cssrlib.residuals   the observation model, and its Numba kernels
    cssrlib.ambiguity   ambiguity resolution
    cssrlib.ekf         the filter and the epoch driver

What stays here is what ties them together: the state-vector layout, the
configuration merge, and the constructor.

The kernels and log formats are re-exported below so that
``from cssrlib.gnssobs import _ddidx_core`` and friends keep resolving.
"""

import numpy as np

from cssrlib.gnss import uGNSS, uTYP
from cssrlib.state import StateLayout

from cssrlib.qc import QualityControlMixin
from cssrlib.residuals import ObservationModelMixin
from cssrlib.ambiguity import AmbiguityMixin
from cssrlib.ekf import FilterMixin

# Re-exports: these moved to the modules above, but callers and tests import
# them from here.
from cssrlib.ambiguity import _ddidx_core                       # noqa: F401
from cssrlib.residuals import (_ddcov_numpy,                    # noqa: F401
                               _tropmapf_dispatch_ppp,          # noqa: F401
                               _sdres_variance,                 # noqa: F401
                               _sdres_core,                     # noqa: F401
                               _sdres_build_plan)               # noqa: F401
from cssrlib.residuals import (fmt_ztd, fmt_ion,                # noqa: F401
                               fmt_res, fmt_amb,                # noqa: F401
                               MIN_SIN_EL,                      # noqa: F401
                               TROPO_MODEL_SAAST,               # noqa: F401
                               TROPO_MODEL_HOPF)                # noqa: F401


class gnssobs(QualityControlMixin, ObservationModelMixin, AmbiguityMixin,
              FilterMixin):
    """ class for PPP processing """

    nav = None
    VAR_HOLDAMB = 0.001

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, cfg=None,
                 trop_opt=1, iono_opt=1, phw_opt=1):
        """ initialize variables for PPP

        ``cfg`` is a :class:`~cssrlib.gnss.ProcConfig`, normally built by one
        of the factories in :mod:`cssrlib.config`. It replaces the config
        already on ``nav``. The ``trop_opt`` / ``iono_opt`` / ``phw_opt``
        keywords are the older way of saying the same thing and still work
        when no ``cfg`` is given.
        """
        from cssrlib.config import base_config

        self.nav = nav

        if cfg is None:
            cfg = base_config(nf=nav.nf, pmode=nav.pmode,
                              trop_opt=trop_opt, iono_opt=iono_opt,
                              phw_opt=phw_opt)
        self._apply_config(cfg)

        # Position (+ optional velocity), zenith tropo delay and
        # slant ionospheric delay states
        #
        # One object owns where every unknown sits; IB/II/IT below read it
        # rather than re-deriving the arithmetic.
        self.layout = StateLayout(
            pmode=self.nav.pmode,
            nf=self.nav.nf,
            ntrop=(1 if self.nav.trop_opt == 1 else 0),
            niono=(uGNSS.MAXSAT if self.nav.iono_opt == 1 else 0))
        self.layout.apply_to(self.nav)

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.phw = np.zeros(uGNSS.MAXSAT)
        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Observation noise, initial sigmas, process noise and the
        # processing options all now come from cfg (see cssrlib.config);
        # they used to be assigned here and then partly re-assigned by
        # whichever subclass had been instantiated.

        # Initial state vector
        #
        self.nav.x[0:3] = pos0
        if self.nav.pmode >= 1:  # kinematic
            self.nav.x[3:6] = 0.0  # velocity

        # Diagonal elements of covariance matrix
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True

        dP[0:3] = self.nav.sig_p0**2
        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            dP[3:6] = self.nav.sig_v0**2

        # Tropo delay
        if self.nav.trop_opt == 1:  # trop is estimated
            if self.nav.pmode >= 1:  # kinematic
                dP[6] = self.nav.sig_ztd0**2
            else:
                dP[3] = self.nav.sig_ztd0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2

        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2

        if self.nav.trop_opt == 1:  # trop is estimated
            # Tropo delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[6] = self.nav.sig_qztd**2
            else:
                self.nav.q[3] = self.nav.sig_qztd**2

        if self.nav.iono_opt == 1:  # iono is estimated
            # Iono delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[7:7+uGNSS.MAXSAT] = self.nav.sig_qion**2
            else:
                self.nav.q[4:4+uGNSS.MAXSAT] = self.nav.sig_qion**2

        # ambiguity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[7+uGNSS.MAXSAT:7 +
                       (uGNSS.MAXSAT*self.nav.nf+1)] = self.nav.sig_qb**2
        else:
            self.nav.q[4+uGNSS.MAXSAT:4 +
                       (uGNSS.MAXSAT*self.nav.nf+1)] = self.nav.sig_qb**2

        # Logging level
        #
        self.monlevel = 0
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def _apply_config(self, cfg):
        """Merge a configuration into ``nav``, letting the caller win.

        Callers routinely configure ``nav`` before handing it to an engine --
        ``nav.rb = <base station ECEF>`` above all. Replacing ``nav.cfg``
        outright would silently discard that, so each field is only taken
        from ``cfg`` where the caller has left ``nav`` at the stock default.
        Anything explicitly set survives.
        """
        from cssrlib.gnss import ProcConfig

        stock = ProcConfig(nf=self.nav.nf)
        target = self.nav.cfg

        for field, value in vars(cfg).items():
            try:
                untouched = bool(np.all(getattr(target, field)
                                        == getattr(stock, field)))
            except Exception:
                untouched = getattr(target, field) is getattr(stock, field)
            if untouched:
                setattr(target, field, value)


    def IB(self, s, f, na=3):
        """ return index of phase ambiguity """
        return self.layout.ambiguity(s, f, na)

    def II(self, s, na):
        """ return index of slant ionospheric delay estimate """
        return self.layout.iono(s, na)

    def IT(self, na):
        """ return index of zenith tropospheric delay estimate """
        return self.layout.tropo(na)

    @staticmethod
    def nsig_sys(obs, sys):
        """Number of frequency slots this constellation actually carries.

        May be < ``nf`` under a mixed-nf configuration (e.g. GPS L1/L2/L5
        while ``nf=4`` for Galileo E1/E5a/E5b/E6). The unused high slots are
        zero-padded in the obs arrays and treated as absent observations by
        the residual/state loops, so the constellations need not share the
        same signal count.
        """
        return len(obs.sig[sys][uTYP.L])
