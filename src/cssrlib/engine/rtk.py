"""
module for RTK positioning

"""

from cssrlib.engine.gnssobs import gnssobs
import numpy as np
from copy import copy, deepcopy
from cssrlib.models.ephemeris import satposs
from cssrlib.gnss import sat2prn, uGNSS, uTYP, rCST, uTideModel
from cssrlib.gnss import Nav, ReceiverState
from cssrlib.estimation.config import rtk_config


class DDMeasurements(dict):
    """Return value of :meth:`rtkpos.prepare_double_difference_measurements`.

    A plain ``dict`` (so existing ``result['rs']`` access, ``isinstance(...,
    dict)``, ``.items()``, ``.get()`` etc. keep working) that additionally
    supports attribute access, i.e. ``dd.rs`` as well as ``dd['rs']``. This
    makes the contract self-documenting and IDE-friendly when the result is
    consumed by an external estimator such as a GTSAM factor graph.

    Fields
    ------
    rs, vs, dts, svh : np.ndarray
        Rover satellite ECEF position [m], velocity [m/s], clock offset [s]
        and health, aligned with ``obs.sat``.
    rsb, vsb, dtsb, svhb : np.ndarray
        Same quantities for the base receiver.
    iu : np.ndarray of int
        Indices into ``obs.sat`` of satellites common to rover and base.
    ir : np.ndarray of int
        Indices into ``obsb.sat`` of the same common satellites.
    sat : np.ndarray of int
        Common satellite numbers (``obs.sat[iu]``).
    el : np.ndarray of float
        Rover elevation angles [rad] for ``sat`` at ``pos_pred``.
    obs_sd : Obs
        Single-difference (rover - base) observations at ``sat``: ``obs_sd.L``
        and ``obs_sd.P`` carry the per-frequency differences.
    y, e : np.ndarray or None
        Zero-difference base residuals / line-of-sight (``None`` when
        ``dd_only=True``).
    yu, eu, elu : np.ndarray or None
        Zero-difference rover residuals / line-of-sight / elevations over the
        full ``obs.sat`` set (``None`` when ``compute_zdres=False``).
    pos_pred : np.ndarray
        Receiver position used to linearise the geometry [m].
    """

    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class rtkpos(gnssobs):
    """ class for RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, base_nav=None,
                 cfg=None):
        """ initialize variables for RTK

        Everything that used to distinguish this class from ppprtkpos is now
        in :func:`cssrlib.estimation.config.rtk_config`; pass ``cfg`` to adjust it.
        """
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         cfg=rtk_config(nf=nav.nf, pmode=nav.pmode)
                         if cfg is None else cfg)

        # The base is a second receiver, not a second engine: it needs its
        # own per-receiver bookkeeping (edt / el / gf / slip) and nothing
        # else. Ephemerides, corrections and the processing configuration
        # are the same for both, so they are shared rather than deepcopied.
        # Sharing nav.glo_ch is a small fix in itself -- the base used to
        # populate its own copy of the GLONASS channel table, leaving the
        # rover's empty for satellites only the base had seen.
        self.base_rcv = ReceiverState(nf=self.nav.nf)

        if base_nav is None:
            self.base_nav = self.nav
        else:
            # Opt-in: the base runs on different navigation data. Only the
            # data and configuration are copied -- no file handles involved,
            # so none of the old detach-and-restore dance is needed.
            self.base_nav = Nav(nf=self.nav.nf)
            self.base_nav.data = deepcopy(self.nav.data)
            self.base_nav.cfg = deepcopy(self.nav.cfg)
            self.base_nav.rcv = self.base_rcv
            self._override_nav(self.base_nav, base_nav)

        if self.base_nav is not self.nav:
            self.base_nav.tidecorr = uTideModel.NONE

    def single_differences(self, obs, obsb, rs, dts, svh,
                           rsb=None, dtsb=None, svhb=None):
        """Build rover-base single differences for DD-only pipelines.

        Runs quality control on both receivers, intersects the common
        satellites and differences their L / P observations. The right
        entry point when the state estimate is maintained outside cssrlib
        (e.g. in a GTSAM factor graph).

        Returns (iu, obs_), where obs_ carries rover-base differenced
        L / P at the common satellite set. Pre-computed base satellite
        states (rsb / dtsb / svhb) may be passed to skip satposs.
        """
        rover = self.nav.rcv
        base = self.base_rcv

        if rsb is None or dtsb is None or svhb is None:
            rsb, _, dtsb, svhb, _ = satposs(obsb, self.base_nav)

        # Which receiver each edit belongs to is now an argument, not a
        # temporary rebinding of self.nav.
        sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb,
                               rr=self.base_nav.rb, rcv=base)
        sat_ed_u = self.qcedit(obs, rs, dts, svh, rcv=rover)

        np.maximum(rover.slip, base.slip, out=rover.slip)

        # Per-frequency editing: a sat survives qcedit if ANY band passed,
        # so zero out the bands that failed on either receiver. Consumers
        # (SD diff below, DD factor builders) already skip zero L / P.
        nfu = min(self.nav.nf, obs.L.shape[1])
        eu = rover.edt[np.asarray(obs.sat) - 1, :nfu] > 0
        obs.L[:, :nfu][eu] = 0.0
        obs.P[:, :nfu][eu] = 0.0
        nfr = min(self.nav.nf, obsb.L.shape[1])
        er = base.edt[np.asarray(obsb.sat) - 1, :nfr] > 0
        obsb.L[:, :nfr][er] = 0.0
        obsb.P[:, :nfr][er] = 0.0

        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        iu, ir = self._common_indices(obs, obsb, sat_ed)

        obs_ = copy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = self._build_frequency_diff(obs.L[iu, :], obsb.L[ir, :])
        obs_.P = self._build_frequency_diff(obs.P[iu, :], obsb.P[ir, :])
        return iu, obs_

    def _common_indices(self, obs, obsb, sat_ed):
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        return iu, ir

    @staticmethod
    def _row_has_nonzero(row):
        return np.any(row != 0)

    def update_ambiguities(self, obs):
        """Initialize / reset the float ambiguity states for this epoch.

        Does the per-satellite ambiguity bookkeeping (cycle-slip / outage
        resets and fresh-ambiguity initialization) without any Kalman
        time/measurement update. Intended for external solvers (e.g. GTSAM)
        that reuse cssrlib's ambiguity bookkeeping but own the state.
        """
        ns = len(obs.sat)
        sat = obs.sat
        for f in range(self.nav.nf):
            for i in range(uGNSS.MAXSAT):
                self.nav.outc[i, f] += 1
                sat_ = i + 1
                sys_i, _ = sat2prn(sat_)
                reset = (
                    self.nav.outc[i, f] > self.nav.maxout
                    or self._row_has_nonzero(self.nav.edt[i, :])
                    or self._row_has_nonzero(self.nav.slip[i, :])
                )
                if sys_i not in obs.sig:
                    continue
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0
                    self.nav.slip[i, f] = 0

            for i in range(ns):
                sat_i = sat[i]
                if self._row_has_nonzero(self.nav.edt[sat_i-1, :]):
                    continue
                sys_i, _ = sat2prn(sat_i)
                if sys_i not in obs.sig:
                    continue
                sig = obs.sig[sys_i][uTYP.L][f]
                fi = (
                    sig.frequency(self.nav.glo_ch.get(sat_i, 0))
                    if sys_i == uGNSS.GLO else sig.frequency()
                )
                lam = rCST.CLIGHT / fi if fi > 0 else 0.0
                cp, pr = obs.L[i, f], obs.P[i, f]
                if cp == 0 or pr == 0 or lam == 0:
                    continue

                # This band was observed, so it is not an outage: cancel the
                # increment made above. Without this nothing ever cleared
                # outc on this path -- it only reached zero by way of the
                # reset it triggered -- so a continuously tracked satellite
                # sawtoothed 1..maxout and had its ambiguity wiped and
                # re-seeded from the pseudorange every maxout+1 epochs. The
                # EKF driver does the same thing off vsat (see
                # FilterMixin.process); this is the DD-only equivalent.
                self.nav.outc[sat_i-1, f] = 0

                j = self.IB(sat_i, f, self.nav.na)
                if self.nav.x[j] == 0.0:
                    self.initx(cp - pr/lam, self.nav.sig_n0**2, j)

        # Slip flags consumed: clear so the next qcedit starts clean.
        # Without this, any sat that ever sees an LLI/GF slip stays flagged
        # forever and triggers an ambiguity reset every subsequent epoch —
        # wiping the freshly initialized N before AR can ever ratio-test.
        self.nav.slip[:] = 0

    def prepare_double_difference_measurements(
        self, obs, obsb, pos_pred=None, cs=None, orb=None, bsx=None,
        rs=None, vs=None, dts=None, svh=None,
        rsb=None, vsb=None, dtsb=None, svhb=None,
        dd_only=True, compute_zdres=False,
    ):
        """Prepare rover/base double-difference observations (no EKF).

        Computes the building blocks an external estimator (e.g. a GTSAM
        factor graph) needs to form double-difference RTK factors. The
        minimal core always runs the DD-only path; ``cs``/``orb``/``bsx`` and
        the ``dd_only``/``compute_zdres`` flags are accepted for backward
        compatibility but ignored (SSR, precise orbit and the undifferenced
        zdres path were removed). Rover elevations come from ``qcedit``
        (``nav.el``), and ``y``/``e``/``yu``/``eu``/``elu`` are ``None``.

        Parameters
        ----------
        obs, obsb : Obs
            Rover and base observations for the epoch.
        pos_pred : array-like of shape (3,), optional
            Receiver ECEF position. Defaults to ``nav.x[0:3]``.
        rs, vs, dts, svh / rsb, vsb, dtsb, svhb : np.ndarray, optional
            Pre-computed rover / base satellite states; pass them to skip the
            ``satposs`` call when the caller already has them.

        Returns
        -------
        DDMeasurements or None
            Mapping of satellite states, common-satellite indices, DD
            observations and rover elevations (see :class:`DDMeasurements`
            for the full field list). ``None`` when there are too few
            satellites or no base observations.
        """
        if len(obs.sat) == 0 or obsb is None or len(obsb.sat) == 0:
            return None

        rs, vs, dts, svh, nsat, pos_pred = self._prepare_sat_states(
            obs, pos_pred=pos_pred, rs=rs, vs=vs, dts=dts, svh=svh)
        self.nav.nsat[0] = len(obs.sat)
        self.nav.nsat[1] = nsat
        if nsat < 4:
            return None

        if rsb is None or dtsb is None or svhb is None:
            rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.base_nav)

        iu, obs_sd = self.single_differences(
            obs, obsb, rs, dts, svh, rsb=rsb, dtsb=dtsb, svhb=svhb
        )
        ns = len(iu)
        self.nav.nsat[2] = ns
        if ns < 4:
            return None

        sat = obs.sat[iu]
        ir = np.intersect1d(obsb.sat, sat, True, True)[1]

        # Elevations come from qcedit (no zdres in the minimal DD-only core).
        el = self.nav.el[sat-1].copy()
        self.nav.sat = sat
        self.nav.el[sat-1] = el

        return DDMeasurements({
            'rs': rs, 'vs': vs, 'dts': dts, 'svh': svh,
            'rsb': rsb, 'vsb': vsb, 'dtsb': dtsb, 'svhb': svhb,
            'y': None, 'e': None, 'yu': None, 'eu': None, 'elu': None,
            'iu': iu, 'ir': ir, 'sat': sat, 'el': el,
            'obs_sd': obs_sd, 'pos_pred': pos_pred,
        })

    def _override_nav(self, target, source):
        for attr in (
            'eph',
            'geph',
            'seph',
            'peph',
            'ion',
            'ion_gim',
            'ion_region',
            'excl_sat',
            'leaps',
            'glo_ch',
        ):
            if hasattr(source, attr):
                setattr(target, attr, deepcopy(getattr(source, attr)))

        for attr in (
            'cnr_min',
            'cnr_min_gpy',
            'thresslip',
            'elmin',
            'armode',
            'pmode',
            'ephopt',
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _build_frequency_diff(self, rover, base):
        nf = self.nav.nf
        ns, cols = rover.shape
        result = np.zeros((ns, nf))

        # assume first column corresponds to primary frequency (e.g., L1)
        primary_mask = (rover[:, 0] != 0.0) & (base[:, 0] != 0.0)
        result[primary_mask, 0] = rover[primary_mask, 0] - base[primary_mask, 0]

        if nf <= 1 or cols <= 1 or base.shape[1] <= 1:
            return result

        cols_2nd = min(cols, base.shape[1])
        secondary_mask = (rover[:, 1:cols_2nd] != 0.0) & (base[:, 1:cols_2nd] != 0.0)
        valid_rows = np.any(secondary_mask, axis=1)
        if not np.any(valid_rows):
            return result

        secondary_cols = np.argmax(secondary_mask[valid_rows], axis=1) + 1
        row_idx = np.nonzero(valid_rows)[0]
        result[row_idx, 1] = (
            rover[row_idx, secondary_cols] - base[row_idx, secondary_cols]
        )

        return result

    # Deprecated aliases for the previous names (the "_external" / "_dd_only"
    # suffixes lost their meaning once the internal EKF was removed).
    manage_ambiguities_external = update_ambiguities
    base_process_dd_only = single_differences
