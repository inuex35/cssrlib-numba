"""
module for RTK positioning

"""

from cssrlib.pppssr import pppos
import numpy as np
from copy import copy, deepcopy
from contextlib import contextmanager
from cssrlib.ephemeris import satposs


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
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]

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
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
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

        if nf <= 1:
            return result

        for sat_idx in range(ns):
            for col in range(1, cols):
                if col >= base.shape[1]:
                    break
                if rover[sat_idx, col] != 0.0 and base[sat_idx, col] != 0.0:
                    result[sat_idx, 1] = rover[sat_idx, col] - base[sat_idx, col]
                    break

        return result
