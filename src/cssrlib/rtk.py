"""
module for RTK positioning

"""

from cssrlib.pppssr import pppos
import numpy as np
from copy import copy, deepcopy
from contextlib import contextmanager
from cssrlib.ephemeris import satposs
from cssrlib.gnss import sat2prn, uGNSS, uTYP, rCST


class rtkpos(pppos):
    """ class for RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, base_nav=None):
        """ initialize variables for PPP-RTK """

        # trop, iono from cssr
        # phase windup model is local/regional
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         trop_opt=0, iono_opt=0, phw_opt=0)

        # Temporarily detach the log file handle: deepcopy cannot pickle an
        # open TextIOWrapper. Share the same handle between rover/base nav.
        fout = getattr(self.nav, 'fout', None)
        self.nav.fout = None
        try:
            self.base_nav = deepcopy(self.nav)
        finally:
            self.nav.fout = fout
        self.base_nav.fout = fout
        if base_nav is not None:
            self._override_nav(self.base_nav, base_nav)

        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma
        self.nav.sig_p0 = 30.0  # [m]
        self.nav.thresar = 2.0  # AR acceptance threshold
        self.nav.armode = 1     # AR is enabled
        self.nav.maxtdiff = 30.0  # [s] max age of base obs (RTKLIB maxtdiff)

    def base_process_dd_only(self, obs, obsb, rs, dts, svh,
                             rsb=None, dtsb=None, svhb=None):
        """Light variant of base_process for DD-only pipelines.

        Skips zdres on the base receiver (y / e arrays are unused by
        callers that build their own DD residuals downstream), so this
        is the right entry point when the rover-side state estimate is
        being maintained outside cssrlib (e.g. in a GTSAM factor graph).

        Returns (iu, obs_), where obs_ carries rover-base differenced
        L / P at the common satellite set. Pre-computed base satellite
        states (rsb / dtsb / svhb) may be passed to skip satposs.
        """
        nav_rover = self.nav
        nav_base = self.base_nav

        if rsb is None or dtsb is None or svhb is None:
            rsb, _, dtsb, svhb, _ = satposs(obsb, nav_base)

        with self._use_nav(nav_base):
            sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=nav_base.rb)
        with self._use_nav(nav_rover):
            sat_ed_u = self.qcedit(obs, rs, dts, svh)

        np.maximum(nav_rover.slip, nav_base.slip, out=nav_rover.slip)

        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        iu, ir = self._common_indices(obs, obsb, sat_ed)

        obs_ = copy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = self._build_frequency_diff(obs.L[iu, :], obsb.L[ir, :])
        obs_.P = self._build_frequency_diff(obs.P[iu, :], obsb.P[ir, :])
        return iu, obs_

    def base_process(self, obs, obsb, rs, dts, svh,
                     rsb=None, vsb=None, dtsb=None, svhb=None):
        """ processing for base station in RTK.

        Pre-computed base satellite states (rsb/vsb/dtsb/svhb) may be passed
        to avoid recomputing satposs when the caller has them already.
        """
        nav_rover = self.nav
        nav_base = self.base_nav

        if rsb is None or vsb is None or dtsb is None or svhb is None:
            rsb, vsb, dtsb, svhb, _ = satposs(obsb, nav_base)
        with self._use_nav(nav_base):
            yr, er, elr = self.zdres(
                obsb, None, None, rsb, vsb, dtsb, nav_base.rb, 0)

        # Editing observations (base/rover)
        with self._use_nav(nav_base):
            sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=nav_base.rb)
        with self._use_nav(nav_rover):
            sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # Propagate base-side cycle-slip flags into the rover so udstate's
        # ambiguity reset triggers on either-side slips. Without this a
        # base-only LLI/GF slip would silently corrupt the DD ambiguity.
        np.maximum(nav_rover.slip, nav_base.slip, out=nav_rover.slip)

        # define common satellite between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        iu, ir = self._common_indices(obs, obsb, sat_ed)
        ns = len(iu)

        y = np.zeros((ns*2, self.nav.nf*2))
        e = np.zeros((ns*2, 3))

        y[ns:, :] = yr[ir, :]
        e[ns:, :] = er[ir, :]

        # Shallow copy is safe: callers only read obs_.sat / obs_.L / obs_.P
        # / obs_.sig / obs_.t — none of which are mutated downstream.
        obs_ = copy(obs)
        obs_.sat = obs.sat[iu]

        rover_L = obs.L[iu, :]
        base_L = obsb.L[ir, :]
        obs_.L = self._build_frequency_diff(rover_L, base_L)

        rover_P = obs.P[iu, :]
        base_P = obsb.P[ir, :]
        obs_.P = self._build_frequency_diff(rover_P, base_P)

        return y, e, iu, obs_

    def _common_indices(self, obs, obsb, sat_ed):
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        return iu, ir

    @staticmethod
    def _row_has_nonzero(row):
        return np.any(row != 0)

    def manage_ambiguities_external(self, obs):
        """Update ambiguity/reset state without running udstate/kfupdate.

        Intended for external solvers (e.g. GTSAM) that reuse cssrlib's
        ambiguity bookkeeping but own the state propagation/update step.
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
                j = self.IB(sat_i, f, self.nav.na)
                if self.nav.x[j] == 0.0:
                    self.initx(cp - pr/lam, self.nav.sig_n0**2, j)

        # Slip flags consumed: clear so the next qcedit starts clean.
        # Mirrors udstate's slip[:] = 0 at end. Without this, any sat that
        # ever sees an LLI/GF slip stays flagged forever and triggers an
        # ambiguity reset every subsequent epoch — wiping the freshly
        # initialized N before AR can ever ratio-test.
        self.nav.slip[:] = 0

    def prepare_relative_measurements(
        self, obs, obsb, pos_pred=None, cs=None, orb=None, bsx=None,
        rs=None, vs=None, dts=None, svh=None,
        rsb=None, vsb=None, dtsb=None, svhb=None,
        dd_only=False, compute_zdres=True,
    ):
        """Prepare rover/base relative observations without EKF update.

        Returns a dict with satellite states, common-satellite indices,
        DD observations, and rover elevations at `pos_pred`.
        """
        if len(obs.sat) == 0 or obsb is None or len(obsb.sat) == 0:
            return None

        if rs is None or vs is None or dts is None or svh is None:
            rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        else:
            nsat = int(np.count_nonzero(~np.isnan(dts)))
        self.nav.nsat[0] = len(obs.sat)
        self.nav.nsat[1] = nsat
        if nsat < 4:
            return None

        if rsb is None or dtsb is None or svhb is None or (not dd_only and vsb is None):
            rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)

        if dd_only:
            iu, obs_sd = self.base_process_dd_only(
                obs, obsb, rs, dts, svh, rsb=rsb, dtsb=dtsb, svhb=svhb
            )
            y = None
            e = None
        else:
            y, e, iu, obs_sd = self.base_process(
                obs, obsb, rs, dts, svh,
                rsb=rsb, vsb=vsb, dtsb=dtsb, svhb=svhb,
            )
        ns = len(iu)
        self.nav.nsat[2] = ns
        if ns < 4:
            return None

        sat = obs.sat[iu]
        ir = np.intersect1d(obsb.sat, sat, True, True)[1]

        if pos_pred is None:
            pos_pred = self.nav.x[0:3].copy()
        if compute_zdres:
            yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, pos_pred)
            el = elu[iu]
        else:
            yu = None
            eu = None
            elu = None
            el = self.nav.el[sat-1].copy()
        self.nav.sat = sat
        self.nav.el[sat-1] = el

        return {
            'rs': rs, 'vs': vs, 'dts': dts, 'svh': svh,
            'rsb': rsb, 'vsb': vsb, 'dtsb': dtsb, 'svhb': svhb,
            'y': y, 'e': e, 'yu': yu, 'eu': eu, 'elu': elu,
            'iu': iu, 'ir': ir, 'sat': sat, 'el': el,
            'obs_sd': obs_sd, 'pos_pred': pos_pred,
        }

    @contextmanager
    def _use_nav(self, nav):
        original_nav = self.nav
        self.nav = nav
        try:
            yield
        finally:
            self.nav = original_nav

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
