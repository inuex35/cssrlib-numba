"""Carrier-phase ambiguity resolution.

The single- to double-difference transformation, the LAMBDA search and its
RTKLIB-compatible variants, and fix-and-hold."""

import numpy as np
from numba import njit

from cssrlib.gnss import sat2prn, uGNSS, rCST
from cssrlib.gnss import SAT_SYS_ARR
from cssrlib.gnss import uTropoModel
from cssrlib.gnss import time2str
from cssrlib.core.mlambda import mlambda

# format definition for logging
fmt_ztd = "{}         ztd      ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
fmt_ion = "{} {}-{} ion {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f}\n"
fmt_res = "{} {}-{} res {} ({:3d}) {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
fmt_amb = "{} {}-{} amb {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f} {:10.3f}\n"

MIN_SIN_EL = 0.1 * rCST.D2R
TROPO_MODEL_SAAST = int(uTropoModel.SAAST)
TROPO_MODEL_HOPF = int(uTropoModel.HOPF)

@njit(cache=True)
def _ddidx_core(sat_arr, nav_x, nav_vsat, nav_el, sys_lookup,
                na, nf, MAXSAT, GNSSMAX, elmaskar):
    """Inner loop of ddidx -- pick the reference satellite per (system, band)
    and collect the DD pair indices into the ambiguity slots of nav.x.

    Returns (ix, fix); the caller stores fix into nav.fix. The Python-list
    membership test of the original is replaced by an O(1) presence array
    indexed by satellite number, and the reference index is tracked
    explicitly instead of relying on the loop variable leaking out of the
    search loop.
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

class ArResult:
    """Outcome of one ambiguity resolution, as data.

    The historical interface leaves half its answer on the engine
    (``_last_s0``/``_last_s1``) and half on ``nav`` (fix marks, xa, Pa);
    a refactor once dropped the stash and nothing failed until the
    accuracy did. This carries the whole answer in the return value.
    """

    __slots__ = ('nb', 'xa', 's0', 's1')

    def __init__(self, nb, xa, s0, s1):
        self.nb, self.xa, self.s0, self.s1 = int(nb), xa, float(s0), float(s1)

    @property
    def ratio(self):
        return 0.0 if self.s0 <= 0.0 else self.s1 / self.s0

    @property
    def fixed(self):
        return self.nb > 0


class AmbiguityMixin:
    """Ambiguity resolution, mixed into :class:`~cssrlib.engine.gnssobs.gnssobs`."""

    def ddidx(self, nav, sat):
        """ index for SD to DD transformation matrix D """
        sat_arr = np.ascontiguousarray(sat, dtype=np.int32)
        ix, fix = _ddidx_core(
            sat_arr, nav.x, nav.vsat, nav.el, SAT_SYS_ARR,
            nav.na, nav.nf,
            int(uGNSS.MAXSAT), int(uGNSS.GNSSMAX), nav.elmaskar)
        nav.fix = fix
        return ix

    def restamb(self, bias, nb, ix=None):
        """ restore SD ambiguity

        ``bias`` is mlambda's fixed double differences, one per row of the
        DD index ``ix`` that ``ddidx`` built -- same rows, same order. When
        ``ix`` is passed the restoration is exactly that correspondence:
        the reference keeps its float value and each target follows from
        its fixed difference.

        Without ``ix`` (older callers), the pairing is re-derived from
        ``nav.fix``, which silently assumes what ddidx happens to do: the
        reference is the lowest-PRN fix==2 satellite of each (system, band)
        and the targets follow in PRN order. Change the reference choice
        and this path scrambles every ambiguity -- measured once at 1190
        fixes collapsing to 1. New callers should pass ``ix``.
        """
        xa = self.nav.x.copy()
        xa[0:self.nav.na] = self.nav.xa[0:self.nav.na]

        if ix is not None:
            for row in range(len(ix)):
                i_ref, j = int(ix[row, 0]), int(ix[row, 1])
                xa[i_ref] = self.nav.x[i_ref]
                xa[j] = xa[i_ref] - bias[row]
            return xa

        nv = 0
        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                n = 0
                index = []
                for i in range(uGNSS.MAXSAT):
                    sys, _ = sat2prn(i+1)
                    if sys != m or self.nav.fix[i, f] != 2:
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

    def resolve_ambiguities(self, sat):
        """Resolve integers and return the whole outcome as an ArResult.

        Chooses the variant by ``nav.rtklib_mode`` (the demo5 retry when
        set, plain LAMBDA otherwise) and packs the ratio pair into the
        result instead of leaving it on the engine. The nav side effects
        documented on the underlying methods still happen -- they are the
        interface to fix-and-hold and validation -- but no caller of this
        method needs to read hidden attributes for the answer.
        """
        if getattr(self.nav, 'rtklib_mode', False):
            nb, xa = self.resamb_lambda_rtklib(sat)
        else:
            nb, xa = self.resamb_lambda(sat, self.nav.parmode,
                                        self.nav.par_P0)
        return ArResult(nb, xa, self._last_s0, self._last_s1)

    def resamb_lambda(self, sat, parmode=1, P0=0.995):
        """ resolve integer ambiguity using LAMBDA method

        parmode selects the LAMBDA search (1: full ILS, 2: partial AR); it is
        nav.parmode, not nav.armode -- the latter switches AR on/off and
        fix-and-hold.

        Inputs, beyond the arguments
        ----------------------------
        ``sat`` is a presence check only; the actual selection is
        ``nav.vsat == 1`` (see ``ddidx``). The float state and covariance
        come from ``nav.x`` / ``nav.P``; elevations from ``nav.el``.

        Side effects -- these are API, callers depend on each
        -----------------------------------------------------
        * ``nav.fix`` is (re)written on **every** call by ``ddidx``, accepted
          or not: 2 for satellites in a double difference, 1 for a candidate
          below ``nav.elmaskar`` encountered before the reference in PRN
          order. ``restamb`` and fix-and-hold read it.
        * ``self._last_s0`` / ``self._last_s1``: the ILS ratio pair, stashed
          for wrappers and gates. A refactor once dropped this and cost a
          silent 0.85 m 3D RMS downstream before it was found.
        * on acceptance only: ``nav.xa`` / ``nav.Pa``, the fixed
          non-ambiguity state via ``xa = x - K (y_float - b)`` -- the sign
          and the content of K are part of the contract too.
        """
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
        b, s, nfix, Ps = mlambda(y, Qb, parmode=parmode, P0=P0)
        # Stash s[0], s[1] so wrappers can read the ratio without re-running
        # mlambda (restored: the DD-minimal core did this here).
        self._last_s0 = float(s[0]) if len(s) > 0 else 0.0
        self._last_s1 = float(s[1]) if len(s) > 1 else 0.0
        if nfix > 0 and (parmode == 2 or s[0] <= 0.0 or
                         s[1]/s[0] >= self.nav.thresar):
            self.nav.xa = self.nav.x[0:na].copy()
            self.nav.Pa = self.nav.P[0:na, 0:na].copy()
            bias = b[:, 0]
            y -= b[:, 0]
            K = Qab@np.linalg.inv(Qb)
            self.nav.xa -= K@y
            self.nav.Pa -= K@Qab.T

            # restore SD ambiguity from the very pairs the search used
            xa = self.restamb(bias, nb, ix=ix)

        elif parmode == 2 and nfix == 0:
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

        It excludes at most one satellite per epoch, picked by
        round-robin order across SVs (RTKLIB-style), rather than by
        the largest float-integer gap.

        Side effects, on top of resamb_lambda's
        ---------------------------------------
        * ``nav.lock`` is updated on **every** call, accepted or not:
          incremented for satellites valid this epoch, reset for the rest.
          Next epoch's ``arfilter`` reads ``lock == 1`` as freshly acquired.
        * ``nav.prev_ratio1`` follows every pass-1 ratio;
          ``nav.prev_ratio2`` only successful ones.
        * ``nav.excsat`` is the round-robin cursor: the excluded satellite
          when the retry fixed, else 0.

        Any drop-in replacement must reproduce all of these -- their absence
        does not fail loudly, it changes which satellite the next epoch
        excludes, and the trajectories quietly part.
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

    def holdamb(self, xa):
        """ hold integer ambiguity """
        nb = self.nav.nx-self.nav.na
        v = np.zeros(nb)
        H = np.zeros((nb, self.nav.nx))
        nv = 0
        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                n = 0
                index = []
                for i in range(uGNSS.MAXSAT):
                    sys, _ = sat2prn(i+1)
                    if sys != m or self.nav.fix[i, f] != 2:
                        continue
                    index.append(self.IB(i+1, f, self.nav.na))
                    n += 1
                    self.nav.fix[i, f] = 3  # hold
                # constraint to fixed ambiguity
                for i in range(1, n):
                    v[nv] = (xa[index[0]]-xa[index[i]]) - \
                        (self.nav.x[index[0]]-self.nav.x[index[i]])
                    H[nv, index[0]] = 1.0
                    H[nv, index[i]] = -1.0
                    nv += 1
        if nv > 0:
            R = np.eye(nv)*self.VAR_HOLDAMB
            # update states with constraints
            self.nav.x, self.nav.P, _ = self.kfupdate(
                self.nav.x, self.nav.P, H[0:nv, :], v[0:nv], R)
        return 0

    def holdamb_flags(self):
        """Mark resolved ambiguities as held (nav.fix[i, f]: 2 → 3) without
        running the Kalman update. Use this in pipelines that overwrite
        nav.x / nav.P from another source (e.g. GTSAM marginals) every
        epoch — the kfupdate result would be discarded anyway. Returns
        the number of held ambiguities for sanity checking.
        """
        n_held = 0
        nf = self.nav.nf
        fix = self.nav.fix
        for i in range(uGNSS.MAXSAT):
            for f in range(nf):
                if fix[i, f] == 2:
                    fix[i, f] = 3
                    n_held += 1
        return n_held

    def sysidx(self, satlist, sys_ref):
        """ return index of satellites with sys=sys_ref """
        idx = []
        for k, sat in enumerate(satlist):
            sys, _ = sat2prn(sat)
            if sys == sys_ref:
                idx.append(k)
        return idx
