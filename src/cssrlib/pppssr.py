"""
module for standard PPP positioning
"""

import numpy as np
from numba import njit

from cssrlib.gnss import sat2id, sat2prn, uTYP, uGNSS, rCST, SAT_SYS_ARR
from cssrlib.gnss import uTropoModel, ecef2pos, time2str, timediff
from cssrlib.gnss import uIonoModel, uTideModel
from cssrlib.mlambda import mlambda
from cssrlib.geometry import geodist, satazel

# format definition for logging
fmt_ztd = "{}         ztd      ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
fmt_ion = "{} {}-{} ion {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f}\n"
fmt_res = "{} {}-{} res {} ({:3d}) {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
fmt_amb = "{} {}-{} amb {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f} {:10.3f}\n"

MIN_SIN_EL = 0.1 * rCST.D2R


@njit(cache=True)
def _ddidx_core(sat_arr, nav_x, nav_vsat, nav_el, sys_lookup,
                 na, nf, MAXSAT, GNSSMAX, elmaskar):
    """Inner loop of ddidx — find ref sat per (system, freq) and the
    DD pair indices into the ambiguity slot of nav.x.

    Returns (ix, fix). The caller copies fix into nav.fix and uses ix
    to build the SD→DD transformation. Replaces the Python-set membership
    check with an O(1) presence array indexed by sat number.
    """
    sat_present = np.zeros(MAXSAT + 2, dtype=np.bool_)
    for s in sat_arr:
        si = int(s)
        if 0 < si <= MAXSAT:
            sat_present[si] = True

    fix = np.zeros((MAXSAT, nf), dtype=np.int64)
    ix = np.zeros((MAXSAT, 2), dtype=np.int64)
    nb = 0

    for m in range(GNSSMAX):
        k = na
        for f in range(nf):
            i_ref = -1
            for i in range(k, k + MAXSAT):
                sat_i = i - k + 1
                if sys_lookup[sat_i] != m:
                    continue
                if (not sat_present[sat_i]
                        or nav_x[i] == 0.0
                        or nav_vsat[sat_i - 1, f] == 0):
                    continue
                if nav_el[sat_i - 1] >= elmaskar:
                    fix[sat_i - 1, f] = 2
                    i_ref = i
                    break
                else:
                    fix[sat_i - 1, f] = 1
            if i_ref >= 0:
                for j in range(k, k + MAXSAT):
                    sat_j = j - k + 1
                    if sys_lookup[sat_j] != m:
                        continue
                    if (j == i_ref
                            or not sat_present[sat_j]
                            or nav_x[j] == 0.0
                            or nav_vsat[sat_j - 1, f] == 0):
                        continue
                    if nav_el[sat_j - 1] >= elmaskar:
                        ix[nb, 0] = i_ref
                        ix[nb, 1] = j
                        nb += 1
                        fix[sat_j - 1, f] = 2
            k += MAXSAT

    return ix[:nb].copy(), fix


def _sig_label(sigs, f):
    """Signal id for band ``f``, or a generic fallback if out of range.

    A constellation may expose fewer bands than ``nav.nf``, leaving the
    per-system signal lists shorter than the frequency loop; this keeps the
    log message safe instead of raising IndexError.
    """
    return sigs[f].str() if f < len(sigs) else "f{:d}".format(f)


@njit(cache=True)
def _qc_signal_checks(P_row, L_row, S_row, lli_row, cnr_thresholds):
    nf = P_row.size
    result = np.zeros(nf, dtype=np.int64)
    for f in range(nf):
        if lli_row[f] == 1:
            result[f] = 1
            continue
        if P_row[f] == 0.0:
            result[f] = 2
            continue
        if L_row[f] == 0.0:
            result[f] = 3
            continue
        if S_row[f] < cnr_thresholds[f]:
            result[f] = 4
            continue
    return result


@njit(cache=True)
def _gf_slip_check(L1, L2, lam1, lam2, gf_prev, thresslip):
    gf1 = 0.0
    slip = False
    if L1 != 0.0 and L2 != 0.0:
        gf1 = L1*lam1 - L2*lam2
        if gf_prev != 0.0 and gf1 != 0.0 and abs(gf1-gf_prev) > thresslip:
            slip = True
    return gf1, slip


def _qcedit_system_cache(obs, nav):
    cache = {}
    nf = nav.nf
    for sys, sigs_by_type in obs.sig.items():
        sigs_pr = sigs_by_type[uTYP.C]
        sigs_cp = sigs_by_type[uTYP.L]
        sigs_cn = sigs_by_type[uTYP.S]
        # A constellation may carry fewer bands than nav.nf (e.g. a
        # single-frequency system in a dual-frequency setup). Index sigs_cn
        # defensively so such a system does not raise IndexError here; the
        # absent bands have no observation and get edited out per-satellite
        # in qcedit (invalid PR), while the present bands are still used.
        cnr_thresholds = np.asarray(
            [(nav.cnr_min_gpy
              if (f < len(sigs_cn) and sigs_cn[f].isGPS_PY())
              else nav.cnr_min)
             for f in range(nf)],
            dtype=np.float64,
        )
        gf_pair = None
        if len(sigs_cp) >= 2:
            if sys == uGNSS.GLO:
                gf_pair = "glo"
            else:
                gf_pair = (
                    sigs_cp[0].wavelength() or 0.0,
                    sigs_cp[1].wavelength() or 0.0,
                )
        cache[sys] = (sigs_pr, sigs_cp, sigs_cn, cnr_thresholds, gf_pair)
    return cache


class pppos():
    """ class for PPP processing """

    nav = None
    VAR_HOLDAMB = 0.001

    def __init__(self, nav, pos0=np.zeros(3),
                 logfile=None, trop_opt=1, iono_opt=1, phw_opt=1):
        """ initialize variables for PPP """

        self.nav = nav

        # Number of frequencies (actually signals!)
        #
        self.nav.ephopt = 2  # SSR-APC

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Select iono model
        #
        self.nav.ionoModel = uIonoModel.KLOBUCHAR

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.nav.trop_opt = trop_opt

        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.nav.iono_opt = iono_opt

        # 0: none, 1: full model, 2: local/regional model
        self.nav.phw_opt = phw_opt

        # carrier smoothing
        self.nav.csmooth = False

        # Position (+ optional velocity), zenith tropo delay and
        # slant ionospheric delay states
        #
        self.nav.ntrop = (1 if self.nav.trop_opt == 1 else 0)
        self.nav.niono = (uGNSS.MAXSAT if self.nav.iono_opt == 1 else 0)

        self.nav.na = (3 if self.nav.pmode == 0 else 6)
        self.nav.nq = (3 if self.nav.pmode == 0 else 6)

        self.nav.na += self.nav.ntrop + self.nav.niono
        self.nav.nq += self.nav.ntrop + self.nav.niono

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na+uGNSS.MAXSAT*self.nav.nf

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.phw = np.zeros(uGNSS.MAXSAT)
        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Parameters for PPP
        #
        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]
        self.nav.sig_ztd0 = 0.1  # [m]
        self.nav.sig_ion0 = 10.0  # [m]
        self.nav.sig_n0 = 30.0    # [cyc]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 100.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]
        self.nav.sig_qztd = 0.05/np.sqrt(3600)     # [m/sqrt(s)]
        self.nav.sig_qion = 10.0/np.sqrt(1)        # [m/s/sqrt(s)]
        self.nav.sig_qb = 1e-4/np.sqrt(1)          # [m/s/sqrt(s)]

        # Processing options
        #
        self.nav.tidecorr = uTideModel.IERS2010
        # self.nav.tidecorr = uTideModel.SIMPLE
        self.nav.thresar = 3.0  # AR acceptance threshold
        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.nav.armode = 0
        self.nav.elmaskar = np.deg2rad(20.0)  # elevation mask for AR
        self.nav.elmin = np.deg2rad(15.0)

        self.nav.parmode = 2  # 1: normal, 2: PAR
        self.nav.par_P0 = 0.995  # probability of sussefull AR

        # RTKLIB demo5-faithful AR mode. When True, resamb() uses
        # resamb_lambda_rtklib() which enforces ratio >= thresar and emulates
        # manage_amb_LAMBDA's one-satellite round-robin exclusion (no PAR
        # success-rate bypass). Defaults preserve cssrlib's PAR behavior.
        self.nav.rtklib_mode = False
        self.nav.excsat = 0       # last excluded satellite (1..MAXSAT, 0=none)
        self.nav.prev_ratio1 = 0.0  # ratio before exclusion (previous epoch)
        self.nav.prev_ratio2 = 0.0  # ratio after exclusion (previous epoch)
        self.nav.arfilter = True   # drop newly-acquired sats that hurt ratio
        self.nav.minfixsats = 4    # minimum sats required to attempt AR

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

    def initx(self, x0, v0, i):
        """ initialize x and P for index i """
        self.nav.x[i] = x0
        for j in range(self.nav.nx):
            self.nav.P[j, i] = self.nav.P[i, j] = v0 if i == j else 0

    def IB(self, s, f, na=3):
        """ return index of phase ambiguity """
        idx = na+uGNSS.MAXSAT*f+s-1
        return idx

    def II(self, s, na):
        """ return index of slant ionospheric delay estimate """
        return na-uGNSS.MAXSAT+s-1

    def IT(self, na):
        """ return index of zenith tropospheric delay estimate """
        return na-uGNSS.MAXSAT-1

    def varerr(self, nav, el, f):
        """ variation of measurement """
        s_el = max(np.sin(el), 0.1*rCST.D2R)
        fact = nav.eratio[f-nav.nf] if f >= nav.nf else 1
        a = fact*nav.err[1]
        b = fact*nav.err[2]
        return (a**2+(b/s_el)**2)

    def sysidx(self, satlist, sys_ref):
        """ return index of satellites with sys=sys_ref """
        idx = []
        for k, sat in enumerate(satlist):
            sys, _ = sat2prn(sat)
            if sys == sys_ref:
                idx.append(k)
        return idx

    def restamb(self, bias, nb):
        """ restore SD ambiguity """
        nv = 0
        xa = self.nav.x.copy()
        xa[0:self.nav.na] = self.nav.xa[0:self.nav.na]
        sys_lookup = SAT_SYS_ARR

        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                n = 0
                index = []
                for i in range(uGNSS.MAXSAT):
                    if sys_lookup[i+1] != m or self.nav.fix[i, f] != 2:
                        continue
                    index.append(self.IB(i+1, f, self.nav.na))
                    n += 1
                if n < 2:
                    continue
                xa[index[0]] = self.nav.x[index[0]]
                for i in range(1, n):
                    xa[index[i]] = xa[index[0]]-bias[nv]
                    nv += 1
        return xa

    def ddidx(self, nav, sat):
        """ index for SD to DD transformation matrix D """
        sat_arr = np.ascontiguousarray(sat, dtype=np.int32)
        ix, fix = _ddidx_core(
            sat_arr, nav.x, nav.vsat, nav.el, SAT_SYS_ARR,
            nav.na, nav.nf,
            int(uGNSS.MAXSAT), int(uGNSS.GNSSMAX), nav.elmaskar)
        nav.fix = fix
        return ix

    def resamb_lambda_partial(self, sat, armode=1, P0=0.995, max_drop=5):
        """Partial-AR variant of resamb_lambda.

        Starts with the full DD set from ddidx(). If the ratio test fails,
        drops the DD whose float-integer gap |y - round(y)| is largest and
        retries. Continues up to max_drop drops or until fewer than 4 DDs
        remain. Each dropped DD's target sat gets nav.fix set to 1 so
        restamb() only restores the accepted subset.

        Use instead of resamb_lambda() when multipath-contaminated float
        ambiguities prevent full AR — a contaminated-N subset often
        passes ratio once the worst 1-3 sats are excluded.

        Returns (nb_accepted, xa). nb_accepted=0 means no partial subset
        passed ratio test; -1 means not enough DDs to start with.
        """
        nx = self.nav.nx
        na = self.nav.na
        xa_out = np.zeros(na)
        ix_full = self.ddidx(self.nav, sat)
        if len(ix_full) < 4:
            return -1, -1

        active = np.ones(len(ix_full), dtype=bool)
        # Cache state snapshot so we can restore nav.fix after partial.
        fix_snapshot = self.nav.fix.copy()

        for _drop_iter in range(max_drop + 1):
            sel = np.where(active)[0]
            if len(sel) < 4:
                break
            ix = ix_full[sel]
            y = self.nav.x[ix[:, 0]] - self.nav.x[ix[:, 1]]
            DP = self.nav.P[ix[:, 0], na:nx] - self.nav.P[ix[:, 1], na:nx]
            Qb = DP[:, ix[:, 0] - na] - DP[:, ix[:, 1] - na]
            Qab = self.nav.P[0:na, ix[:, 0]] - self.nav.P[0:na, ix[:, 1]]

            b, s, nfix, Ps = mlambda(y, Qb, parmode=armode, P0=P0)
            if nfix <= 0:
                break

            bias = b[:, 0]
            ratio_ok = (armode == 2 or s[0] <= 0.0 or
                        s[1] / s[0] >= self.nav.thresar)

            if ratio_ok:
                # Demote excluded sats' fix flag from 2 → 1 so restamb()
                # only acts on the accepted subset.
                dropped = np.where(~active)[0]
                for gidx in dropped:
                    t_idx = ix_full[gidx, 1]  # index into nav.x
                    offset = t_idx - na
                    f_t = int(offset // uGNSS.MAXSAT)
                    s_t = int(offset % uGNSS.MAXSAT) + 1
                    # Only demote if no other accepted row uses this target
                    still_used = any(
                        ix_full[gi, 1] == t_idx for gi in sel)
                    if not still_used and 0 < s_t <= uGNSS.MAXSAT:
                        self.nav.fix[s_t - 1, f_t] = 1

                self.nav.xa = self.nav.x[0:na].copy()
                self.nav.Pa = self.nav.P[0:na, 0:na].copy()
                y_res = y - bias
                K = Qab @ np.linalg.inv(Qb)
                self.nav.xa -= K @ y_res
                self.nav.Pa -= K @ Qab.T
                xa_out = self.restamb(bias, len(ix))
                return len(ix), xa_out

            # Drop the worst DD (largest float-integer gap)
            frac = np.abs(y - np.round(y))
            worst_local = int(np.argmax(frac))
            active[sel[worst_local]] = False

        # All attempts failed — restore fix snapshot and return float.
        self.nav.fix = fix_snapshot
        return 0, xa_out

    def resamb_lambda(self, sat, armode=1, P0=0.995):
        """ resolve integer ambiguity using LAMBDA method """
        nx = self.nav.nx
        na = self.nav.na
        xa = np.zeros(na)
        ix = self.ddidx(self.nav, sat)
        nb = len(ix)
        if nb <= 0:
            print("no valid DD")
            return -1, -1

        # y=D*xc, Qb=D*Qc*D', Qab=Qac*D'
        y = self.nav.x[ix[:, 0]]-self.nav.x[ix[:, 1]]
        DP = self.nav.P[ix[:, 0], na:nx]-self.nav.P[ix[:, 1], na:nx]
        Qb = DP[:, ix[:, 0]-na]-DP[:, ix[:, 1]-na]
        Qab = self.nav.P[0:na, ix[:, 0]]-self.nav.P[0:na, ix[:, 1]]

        # MLAMBDA ILS
        b, s, nfix, Ps = mlambda(y, Qb, parmode=armode, P0=P0)
        # Stash s[0],s[1] so wrappers (e.g. resamb_lambda_rtklib) can read
        # the ratio without re-running mlambda.
        self._last_s0 = float(s[0]) if len(s) > 0 else 0.0
        self._last_s1 = float(s[1]) if len(s) > 1 else 0.0
        if nfix > 0 and (armode == 2 or s[0] <= 0.0 or
                         s[1]/s[0] >= self.nav.thresar):
            self.nav.xa = self.nav.x[0:na].copy()
            self.nav.Pa = self.nav.P[0:na, 0:na].copy()
            bias = b[:, 0]
            y -= b[:, 0]
            K = Qab@np.linalg.inv(Qb)
            self.nav.xa -= K@y
            self.nav.Pa -= K@Qab.T

            # restore SD ambiguity
            xa = self.restamb(bias, nb)

        elif armode == 2 and nfix == 0:
            nb = 0
            if self.nav.monlevel > 0:
                self.nav.fout.write(
                    "{:s}  Ps={:3.2f} nfix={:d}\n".
                    format(time2str(self.nav.t), Ps, nfix))
        else:
            nb = 0

        return nb, xa

    def resamb_lambda_rtklib(self, sat):
        """RTKLIB demo5 manage_amb_LAMBDA-equivalent AR.

        Pass 1: full ILS + ratio test (parmode=1, ratio >= nav.thresar).
        Pass 2 (only if pass 1 failed and at least minfixsats sats are
        available): exclude one satellite via round-robin (nav.excsat)
        and retry once. arfilter additionally prefers excluding a
        newly-acquired sat (nav.lock == 0) when its appearance dropped
        the ratio.

        Differs from resamb_lambda_partial(): RTKLIB picks the excluded
        sat by round-robin order across SVs, not by the largest
        float-integer gap, and runs at most one exclusion per epoch.
        """
        # Update lock counters: increment for sats valid this epoch,
        # reset to 0 for the rest. Mirrors RTKLIB ssat[].lock semantics.
        valid = set(int(s) for s in sat)
        for i in range(self.nav.lock.shape[0]):
            sv = i + 1
            for f in range(self.nav.nf):
                if sv in valid and self.nav.vsat[i, f] != 0:
                    self.nav.lock[i, f] += 1
                else:
                    self.nav.lock[i, f] = 0

        nb, xa = self.resamb_lambda(sat, 1, self.nav.par_P0)
        ratio = (0.0 if self._last_s0 <= 0.0
                 else self._last_s1 / self._last_s0)
        if nb > 0:
            self.nav.prev_ratio1 = ratio
            self.nav.prev_ratio2 = ratio
            self.nav.excsat = 0
            return nb, xa
        self.nav.prev_ratio1 = ratio

        if len(sat) < self.nav.minfixsats:
            return 0, xa

        # Round-robin: resume from the sat after nav.excsat.
        sat_arr = [int(s) for s in sat]
        try:
            start = sat_arr.index(self.nav.excsat) + 1
        except ValueError:
            start = 0
        order = sat_arr[start:] + sat_arr[:start]

        exc = 0
        # arfilter: if a newly-locked sat (lock==1, i.e. first epoch the
        # counter was incremented) just dragged the ratio below threshold,
        # prefer dropping it.
        if self.nav.arfilter and ratio < self.nav.thresar \
                and self.nav.prev_ratio2 > 0.0 \
                and ratio < 1.1 * self.nav.prev_ratio2:
            for s_ in order:
                if any(0 < self.nav.lock[s_-1, f] <= 1
                       for f in range(self.nav.nf)):
                    exc = s_
                    break
        if exc == 0:
            for s_ in order:
                if any(self.nav.vsat[s_-1, f] != 0
                       for f in range(self.nav.nf)):
                    exc = s_
                    break
        if exc == 0:
            return 0, xa

        # Exclude by zeroing vsat for one epoch; ddidx() then skips it.
        vsat_row = self.nav.vsat[exc-1, :].copy()
        self.nav.vsat[exc-1, :] = 0
        try:
            sat2 = [s for s in sat if int(s) != exc]
            nb, xa = self.resamb_lambda(sat2, 1, self.nav.par_P0)
        finally:
            self.nav.vsat[exc-1, :] = vsat_row

        if nb > 0:
            self.nav.prev_ratio2 = (
                0.0 if self._last_s0 <= 0.0
                else self._last_s1 / self._last_s0)
            self.nav.excsat = exc
            return nb, xa

        self.nav.excsat = 0
        return 0, xa

    def resamb_lambda_subsets(self, sat):
        """RTKLIB-faithful AR with system-level preferred subset retries.

        Pass 1: full AR over all systems via ``resamb_lambda_rtklib``
        (which already handles its own ratio + 1-sat round-robin
        fallback). If pass 1 produces a strong fix
        (ratio >= nav.thresar + 0.5), return immediately.

        Pass 2: when pass 1 is marginal or failed, try system-level
        subsets that exclude one or two constellations entirely. This
        catches the case where one system is multipath-corrupted and
        dragging the full-set AR ratio below threshold:

          * GPS + GAL + QZS                (drop GLO + BDS)
          * GPS + GAL + QZS + BDS          (drop GLO)
          * GPS + GAL + QZS + GLO          (drop BDS)

        Each subset runs ``resamb_lambda`` once. Among the subsets that
        produce a fix with ratio >= nav.thresar, adopt the one with the
        highest ratio (and prefer it over the pass-1 fix when its ratio
        is strictly higher).

        Inspired by libgnss++ rtk_ar_selection::buildPreferredSubsets
        (rsasaki0109/gnssplusplus-library).
        """
        nb_full, xa_full = self.resamb_lambda_rtklib(sat)
        s0_full, s1_full = self._last_s0, self._last_s1
        ratio_full = (0.0 if s0_full <= 0.0 else s1_full / s0_full)

        # Strong full-set fix → no need to search subsets.
        if nb_full > 0 and ratio_full >= self.nav.thresar + 0.5:
            return nb_full, xa_full

        best_nb, best_xa, best_ratio = nb_full, xa_full, ratio_full

        # Subsets always keep the GPS + GAL + QZS core (most reliable
        # in tokyo-class urban multipath).
        core = {uGNSS.GPS, uGNSS.GAL, uGNSS.QZS}
        subsets = (
            core,
            core | {uGNSS.BDS},
            core | {uGNSS.GLO},
        )

        vsat_snapshot = self.nav.vsat.copy()
        try:
            for keep_sys in subsets:
                # Reset vsat each iteration to undo any prior subset's
                # zeroing.
                self.nav.vsat[:, :] = vsat_snapshot
                sub_sat = []
                for s_int in sat:
                    sys_id, _ = sat2prn(int(s_int))
                    if sys_id in keep_sys:
                        sub_sat.append(int(s_int))
                    else:
                        self.nav.vsat[int(s_int) - 1, :] = 0
                if len(sub_sat) < self.nav.minfixsats:
                    continue
                nb_s, xa_s = self.resamb_lambda(sub_sat, 1, self.nav.par_P0)
                if nb_s <= 0:
                    continue
                s0_s, s1_s = self._last_s0, self._last_s1
                ratio_s = (0.0 if s0_s <= 0.0 else s1_s / s0_s)
                if ratio_s < self.nav.thresar:
                    continue
                if ratio_s > best_ratio:
                    best_nb, best_xa, best_ratio = nb_s, xa_s, ratio_s
        finally:
            self.nav.vsat[:, :] = vsat_snapshot

        # Stash the adopted subset's pseudo-ratio into _last_s0/_last_s1
        # so downstream callers reading the ratio see the chosen value.
        if best_nb > 0:
            self._last_s0 = 1.0
            self._last_s1 = best_ratio
        return best_nb, best_xa


    def qcedit(self, obs, rs, dts, svh, rr=None):
        """ Coarse quality control and editing of observations """

        # Predicted position at next epoch
        #
        tt = timediff(obs.t, self.nav.t)
        if rr is None:
            rr_ = self.nav.x[0:3].copy()
            if self.nav.pmode > 0:
                rr_ += self.nav.x[3:6]*tt
        else:
            # rr may be a plain list (e.g. nav.rb); coerce to a float64 array.
            rr_ = np.asarray(rr, dtype=np.float64)

        # Solid Earth tides were removed with the minimal core (they cancel
        # in the short-baseline rover-base double difference).

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        # Total number of satellites
        #
        ns = uGNSS.MAXSAT

        # Default-edited; we'll reset to 0 only for observed sats whose
        # checks all pass. Saves ~200 redundant iterations over unobserved
        # PRNs in the original loop.
        self.nav.edt = np.ones((ns, self.nav.nf), dtype=int)

        obs_sat_arr = np.asarray(obs.sat)
        sys_lookup = SAT_SYS_ARR
        system_cache = _qcedit_system_cache(obs, self.nav)
        sig_table = obs.sig if hasattr(obs, 'sig') else None

        sat = []
        for j, sat_raw in enumerate(obs_sat_arr):
            sat_i = int(sat_raw)

            i = sat_i - 1
            sys_i = sys_lookup[sat_i]
            # Mark observed sat as not edited; sub-checks below may
            # re-set edt[i, f] = 1 for individual frequencies.
            self.nav.edt[i, :] = 0

            # Check satellite exclusion
            #
            if sat_i in self.nav.excl_sat:
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite excluded\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check for valid orbit and clock offset
            #
            if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - invalid eph\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check satellite health
            #
            if svh[j] > 0:
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite unhealthy\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check elevation angle
            #
            _, e = geodist(rs[j, :], rr_)
            _, el = satazel(pos, e)
            self.nav.el[sat_i-1] = el
            if el < self.nav.elmin:
                self.nav.edt[i][:] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write(
                        "{}  {} - edit - low elevation {:5.1f} deg\n"
                        .format(time2str(obs.t), sat2id(sat_i),
                                np.rad2deg(el)))
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR, sigsCP, sigsCN, cnr_thresholds, gf_pair = system_cache[sys_i]

            P_row = obs.P[j, :self.nav.nf]
            L_row = obs.L[j, :self.nav.nf]
            S_row = obs.S[j, :self.nav.nf]
            lli_row = obs.lli[j, :self.nav.nf]
            qc_codes = _qc_signal_checks(
                np.asarray(P_row, dtype=np.float64),
                np.asarray(L_row, dtype=np.float64),
                np.asarray(S_row, dtype=np.float64),
                np.asarray(lli_row, dtype=np.float64),
                cnr_thresholds,
            )

            for f in range(self.nav.nf):
                code = int(qc_codes[f])
                if code == 0:
                    continue
                # LLI=1 is a cycle-slip notification, not a bad observation:
                # flag the sat for ambiguity reset (consumed by
                # manage_ambiguities_external) but keep the measurement
                # (RTKLIB-style behavior). Other codes drop it.
                if code == 1:
                    self.nav.slip[i, f] = 1
                else:
                    self.nav.edt[i, f] = 1
                if self.nav.monlevel > 0:
                    # Label lists may be shorter than nav.nf for a system
                    # with fewer bands; fall back to a generic band name.
                    if code == 1:
                        msg = "slip {:4s} - LLI".format(_sig_label(sigsCP, f))
                    elif code == 2:
                        msg = "edit {:4s} - invalid PR obs".format(
                            _sig_label(sigsPR, f))
                    elif code == 3:
                        msg = "edit {:4s} - invalid CP obs".format(
                            _sig_label(sigsCP, f))
                    else:
                        msg = "edit {:4s} - low C/N0 {:4.1f} dB-Hz".format(
                            _sig_label(sigsCN, f), obs.S[j, f])
                    self.nav.fout.write("{}  {} - {}\n".format(
                        time2str(obs.t), sat2id(sat_i), msg))

            # cycle-slip detection by geometry-free combination
            if (
                obs.L.shape[1] > 1
                and sig_table
                and sys_i in sig_table
                and uTYP.L in sig_table[sys_i]
                and len(sig_table[sys_i][uTYP.L]) >= 2
            ):
                L1R, L2R = obs.L[j, 0:2]
                sig1, sig2 = sig_table[sys_i][uTYP.L][0:2]
                if gf_pair == "glo":
                    ch = self.nav.glo_ch[sat_i]
                    lam1 = sig1.wavelength(ch)
                    lam2 = sig2.wavelength(ch)
                else:
                    lam1, lam2 = gf_pair
                gf_prev = float(self.nav.gf[sat_i])
                gf1, slip = _gf_slip_check(
                    float(L1R),
                    float(L2R),
                    float(lam1),
                    float(lam2),
                    gf_prev,
                    float(self.nav.thresslip),
                )
                if gf1 != 0.0:
                    self.nav.gf[sat_i] = gf1
                if slip:
                    # GF slip is a cycle-slip event: flag for ambiguity
                    # reset, do not drop the observation.
                    self.nav.slip[i, 0:2] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            " {}  {} - slip {:4s} - GF gf0 {:6.3f} gf1 {:6.3f} gf0-gf1 {:6.3f} \n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sig1.str(), gf_prev, gf1,
                                    gf_prev-gf1))
            else:
                # Single frequency or missing signal metadata: skip GF slip test
                obs.L = np.atleast_2d(obs.L)
                obs.P = np.atleast_2d(obs.P)

            # Store satellite which have passed all tests
            #
            if np.any(self.nav.edt[i, :] > 0):
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)
