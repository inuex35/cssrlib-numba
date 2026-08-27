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
    edt, edtb : np.ndarray of bool
        Per-band editing masks for ``sat``, rover and base side, shaped
        ``(len(sat), nf)``. True means qcedit rejected that band. Editing is
        per band -- a satellite is kept while any band is usable -- so a
        consumer reading the raw ``obs`` / ``obsb`` arrays must apply these;
        ``obs_sd`` already has them applied. ``obs`` and ``obsb`` themselves
        are never modified.
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

        Configuration comes from
        :func:`cssrlib.estimation.config.rtk_config`; pass ``cfg`` to adjust it.
        """
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         cfg=rtk_config(nf=nav.nf, pmode=nav.pmode)
                         if cfg is None else cfg)

        # The base gets its own per-receiver bookkeeping (edt / el /
        # gf / slip); ephemerides, corrections and configuration are
        # shared with the rover.
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

        # Mask the bands that failed qcedit on either receiver, into
        # local copies: obs / obsb belong to the caller and the base
        # record is reused across rover epochs.
        nfu = min(self.nav.nf, obs.L.shape[1])
        eu = rover.edt[np.asarray(obs.sat) - 1, :nfu] > 0
        Lu, Pu = obs.L.copy(), obs.P.copy()
        Lu[:, :nfu][eu] = 0.0
        Pu[:, :nfu][eu] = 0.0

        nfr = min(self.nav.nf, obsb.L.shape[1])
        er = base.edt[np.asarray(obsb.sat) - 1, :nfr] > 0
        Lr, Pr = obsb.L.copy(), obsb.P.copy()
        Lr[:, :nfr][er] = 0.0
        Pr[:, :nfr][er] = 0.0

        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        iu, ir = self._common_indices(obs, obsb, sat_ed)

        obs_ = copy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = self._build_frequency_diff(Lu[iu, :], Lr[ir, :])
        obs_.P = self._build_frequency_diff(Pu[iu, :], Pr[ir, :])
        # Rover-side columns follow the common-satellite indexing too.
        for name in ('S', 'D', 'lli'):
            col = getattr(obs, name, None)
            if isinstance(col, np.ndarray) and col.ndim == 2                     and col.shape[0] == len(obs.sat):
                setattr(obs_, name, col[iu, :])
        return iu, obs_

    def _common_indices(self, obs, obsb, sat_ed):
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        return iu, ir

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
                # Per band: qcedit records edt and slip per band, and a
                # satellite now survives on its good bands. Reading the whole
                # row here would let one degraded band reset the ambiguity of
                # every other band on that satellite.
                reset = (
                    self.nav.outc[i, f] > self.nav.maxout
                    or self.nav.edt[i, f] > 0
                    or self.nav.slip[i, f] > 0
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
                if self.nav.edt[sat_i-1, f] > 0:
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

                # Observed band: clear the outage counter incremented
                # above (the EKF driver does the same off vsat).
                self.nav.outc[sat_i-1, f] = 0

                j = self.IB(sat_i, f, self.nav.na)
                if self.nav.x[j] == 0.0:
                    self.initx(cp - pr/lam, self.nav.sig_n0**2, j)

        # Slip flags consumed: clear so the next qcedit starts clean.
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

        # Per-band editing masks over the common satellites, so a caller
        # reading obs / obsb directly can drop the same bands obs_sd did.
        nf = self.nav.nf
        edt_u = self.nav.rcv.edt[sat-1, :nf] > 0
        edt_r = self.base_rcv.edt[sat-1, :nf] > 0

        return DDMeasurements({
            'rs': rs, 'vs': vs, 'dts': dts, 'svh': svh,
            'rsb': rsb, 'vsb': vsb, 'dtsb': dtsb, 'svhb': svhb,
            'y': None, 'e': None, 'yu': None, 'eu': None, 'elu': None,
            'iu': iu, 'ir': ir, 'sat': sat, 'el': el,
            'edt': edt_u, 'edtb': edt_r,
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
        """Rover-minus-base per band, band n against band n.

        Differenced per band, so each column names one frequency. A band
        contributes only where both receivers observed it; anything else
        stays zero, which is how the consumers spell "absent".
        """
        nf = self.nav.nf
        result = np.zeros((rover.shape[0], nf))

        cols = min(nf, rover.shape[1], base.shape[1])
        if cols <= 0:
            return result

        r = rover[:, :cols]
        b = base[:, :cols]
        both = (r != 0.0) & (b != 0.0)
        result[:, :cols] = np.where(both, r - b, 0.0)
        return result

    # Deprecated aliases for the previous names (the "_external" / "_dd_only"
    # suffixes lost their meaning once the internal EKF was removed).
    manage_ambiguities_external = update_ambiguities
    base_process_dd_only = single_differences
