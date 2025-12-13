"""
module for RTK positioning

"""

from cssrlib.pppssr import pppos
import numpy as np
from copy import deepcopy
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

        self.base_nav = deepcopy(self.nav)
        if base_nav is not None:
            self._override_nav(self.base_nav, base_nav)

        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma
        self.nav.sig_p0 = 30.0  # [m]
        self.nav.thresar = 2.0  # AR acceptance threshold
        self.nav.armode = 1     # AR is enabled

    def base_process(self, obs, obsb, rs, dts, svh):
        """ processing for base station in RTK """
        nav_rover = self.nav
        nav_base = self.base_nav

        rsb, vsb, dtsb, svhb, _ = satposs(obsb, nav_base)
        with self._use_nav(nav_base):
            yr, er, elr = self.zdres(
                obsb, None, None, rsb, vsb, dtsb, nav_base.rb, 0)

        # Editing observations (base/rover)
        with self._use_nav(nav_base):
            sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=nav_base.rb)
        with self._use_nav(nav_rover):
            sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # define common satellite between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        ns = len(iu)

        y = np.zeros((ns*2, self.nav.nf*2))
        e = np.zeros((ns*2, 3))

        y[ns:, :] = yr[ir, :]
        e[ns:, :] = er[ir, :]

        obs_ = deepcopy(obs)
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
