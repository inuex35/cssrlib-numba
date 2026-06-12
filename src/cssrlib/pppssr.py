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
from cssrlib.gnss import tropmodel, time2doy
from cssrlib.atmosphere import tropmapf_niell
from cssrlib.constants import CLIGHT, GME

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


@njit(cache=True)
def _range_corrections(trop, iono, antr_pr, antr_cp, ants_pr, ants_cp,
                       cbias, pbias, phw):
    prc = trop + antr_pr + ants_pr + iono - cbias
    cpc = trop + antr_cp + ants_cp - iono - pbias + phw
    return prc, cpc


@njit(cache=True)
def _fill_residual_row(y_row, lam, L_vals, P_vals, col_idx, base_range, cpc_row, prc_row):
    nf = lam.size
    for f in range(nf):
        if col_idx[f] < 0:
            continue
        y_row[f] = L_vals[f]*lam[f] - (base_range + cpc_row[f])
        y_row[f+nf] = P_vals[f] - (base_range + prc_row[f])


@njit(cache=True)
def _tropmapf_dispatch_ppp(doy, pos, el, model):
    if model == TROPO_MODEL_HOPF:
        mapfh = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 72.0) ** 2))
        mapfw = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 120.0) ** 2))
        return mapfh, mapfw
    elif model == TROPO_MODEL_SAAST:
        return tropmapf_niell(doy, pos, el)
    return 0.0, 0.0


@njit(cache=True)
def _shapiro_delay(rsat, rrcv):
    rs = np.linalg.norm(rsat)
    rr = np.linalg.norm(rrcv)
    rrs = np.linalg.norm(rsat - rrcv)
    denom = rs + rr - rrs
    if denom <= 0.0:
        denom = 1e-12
    return (2.0 * GME / (CLIGHT * CLIGHT)) * np.log((rs + rr + rrs) / denom)


@njit(cache=True)
def _zdres_geometry_precompute(rs, rr, pos, elmin, trp_model, doy):
    n = rs.shape[0]
    geom = np.zeros(n, dtype=np.float64)
    los = np.zeros((n, 3), dtype=np.float64)
    el = np.zeros(n, dtype=np.float64)
    mapfh = np.zeros(n, dtype=np.float64)
    mapfw = np.zeros(n, dtype=np.float64)
    relatv = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        rng, los_vec = geodist(rs[i, :], rr)
        geom[i] = rng
        los[i, :] = los_vec
        _, el_val = satazel(pos, los_vec)
        el[i] = el_val
        if el_val < elmin:
            continue
        valid[i] = True
        mf, mw = _tropmapf_dispatch_ppp(doy, pos, el_val, trp_model)
        mapfh[i] = mf
        mapfw[i] = mw
        relatv[i] = _shapiro_delay(rs[i, :], rr)

    return geom, los, el, mapfh, mapfw, relatv, valid


def _zdres_signal_cache(obs, nav):
    """Precompute signal selection arrays for zdres."""

    n = len(obs.P)
    nf = nav.nf
    sys_lookup = SAT_SYS_ARR
    lam_all = np.zeros((n, nf), dtype=np.float64)
    frq_all = np.zeros((n, nf), dtype=np.float64)
    col_idx_all = -np.ones((n, nf), dtype=np.int64)
    L_sel_all = np.zeros((n, nf), dtype=np.float64)
    P_sel_all = np.zeros((n, nf), dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    sys_signal_cache = {}

    for i in range(n):
        sat = obs.sat[i]
        sys = sys_lookup[sat]
        sigsCP = obs.sig[sys][uTYP.L]

        cache_key = (int(sys), int(nav.glo_ch.get(int(sat), 0))
                     if sys == uGNSS.GLO else 0)
        cached = sys_signal_cache.get(cache_key)
        if cached is None:
            if sys == uGNSS.GLO:
                ch = nav.glo_ch[int(sat)]
                lam_full = np.asarray(
                    [sig.wavelength(ch) or 0.0 for sig in sigsCP],
                    dtype=np.float64,
                )
                frq_full = np.asarray(
                    [sig.frequency(ch) or 0.0 for sig in sigsCP],
                    dtype=np.float64,
                )
            else:
                lam_full = np.asarray(
                    [sig.wavelength() or 0.0 for sig in sigsCP],
                    dtype=np.float64,
                )
                frq_full = np.asarray(
                    [sig.frequency() or 0.0 for sig in sigsCP],
                    dtype=np.float64,
                )
            cached = (lam_full, frq_full)
            sys_signal_cache[cache_key] = cached
        else:
            lam_full, frq_full = cached

        max_cols = obs.L.shape[1] if obs.L.ndim == 2 else 0
        if obs.P.ndim == 2:
            max_cols = min(max_cols, obs.P.shape[1])
        if max_cols == 0:
            continue

        L_row = obs.L[i, :] if obs.L.ndim == 2 else obs.L[i]
        P_row = obs.P[i, :] if obs.P.ndim == 2 else obs.P[i]
        L_row_arr = np.asarray(L_row, dtype=np.float64)
        P_row_arr = np.asarray(P_row, dtype=np.float64)

        valid_cols = np.nonzero(
            (L_row_arr[:max_cols] != 0.0) & (P_row_arr[:max_cols] != 0.0)
        )[0]
        if valid_cols.size == 0:
            continue

        count = min(valid_cols.size, nf)
        col_idx_row = col_idx_all[i, :]
        col_idx_row[:count] = valid_cols[:count]
        lam_row = lam_all[i, :]
        frq_row = frq_all[i, :]
        L_sel_row = L_sel_all[i, :]
        P_sel_row = P_sel_all[i, :]

        cols_sel = col_idx_row[:count]
        lam_row[:count] = lam_full[cols_sel]
        frq_row[:count] = frq_full[cols_sel]
        L_sel_row[:count] = L_row_arr[cols_sel]
        P_sel_row[:count] = P_row_arr[cols_sel]

        valid[i] = True

    return lam_all, frq_all, col_idx_all, L_sel_all, P_sel_all, valid


@njit(cache=True)
def _sdres_variance(el: float, col_idx: int, nf: int, eratio: np.ndarray, err: np.ndarray) -> float:
    s_el = np.sin(el)
    if s_el < MIN_SIN_EL:
        s_el = MIN_SIN_EL
    fact = 1.0
    if col_idx >= nf:
        freq_idx = col_idx - nf
        if freq_idx < eratio.size:
            fact = eratio[freq_idx]
        else:
            fact = eratio[-1] if eratio.size > 0 else 1.0
    a = fact * (err[1] if err.size > 1 else err[0])
    b = fact * (err[2] if err.size > 2 else err[-1])
    return a * a + (b / s_el) ** 2


@njit(cache=True)
def _sdres_core(
    mode,
    ns,
    y,
    e,
    x,
    el,
    ref_idx,
    sat_idx,
    col_idx,
    mu_arr,
    lam_ref_arr,
    lam_sat_arr,
    is_phase_arr,
    iono_i_idx,
    iono_j_idx,
    amb_i_idx,
    amb_j_idx,
    trop_idx,
    use_trop,
    use_iono,
    nav_nx,
    mapfw_sd,
    block_idx_arr,
    block_total,
    nf,
    eratio,
    err,
):
    m = ref_idx.size
    v = np.zeros(m, dtype=np.float64)
    H = np.zeros((m, nav_nx), dtype=np.float64)
    Ri = np.zeros(m, dtype=np.float64)
    Rj = np.zeros(m, dtype=np.float64)
    nb = np.zeros(block_total, dtype=np.int64) if block_total > 0 else np.zeros(0, dtype=np.int64)

    for k in range(m):
        ri = ref_idx[k]
        sj = sat_idx[k]
        col = col_idx[k]
        if mode == 0:
            v[k] = (y[ri, col] - y[ri+ns, col]) - (y[sj, col] - y[sj+ns, col])
        else:
            v[k] = y[ri, col] - y[sj, col]

        H[k, 0:3] = -e[ri, :] + e[sj, :]

        if use_trop and trop_idx >= 0:
            diff_map = mapfw_sd[ri] - mapfw_sd[sj]
            H[k, trop_idx] = diff_map
            v[k] -= diff_map * x[trop_idx]

        if use_iono:
            idx_i = iono_i_idx[k]
            idx_j = iono_j_idx[k]
            if idx_i >= 0 and idx_j >= 0:
                mu_val = mu_arr[k]
                H[k, idx_i] = +mu_val
                H[k, idx_j] = -mu_val
                v[k] -= mu_val * (x[idx_i] - x[idx_j])

        if is_phase_arr[k]:
            idx_i = amb_i_idx[k]
            idx_j = amb_j_idx[k]
            lam_i = lam_ref_arr[k]
            lam_j = lam_sat_arr[k]
            if idx_i >= 0:
                H[k, idx_i] = lam_i
            if idx_j >= 0:
                H[k, idx_j] = -lam_j
            if idx_i >= 0 and idx_j >= 0:
                v[k] -= lam_i * (x[idx_i] - x[idx_j])

        Ri[k] = _sdres_variance(el[ri], col, nf, eratio, err)
        Rj[k] = _sdres_variance(el[sj], col, nf, eratio, err)

        if nb.size > 0:
            blk = block_idx_arr[k]
            if blk >= 0 and blk < nb.size:
                nb[blk] += 1

    return v, H, Ri, Rj, nb


def _sdres_build_plan(obs, sat, el, y, nav):
    """Build measurement plan arrays for sdres."""

    nf = nav.nf
    sys_list = list(obs.sig.keys())
    block_stride = nf * 2

    ref_indices = []
    sat_indices = []
    freq_indices = []
    col_indices = []
    block_indices = []
    mu_values = []
    lam_ref_values = []
    lam_sat_values = []
    is_phase_flags = []
    sig_label_indices = []
    sig_label_table = []
    sig_label_map = {}

    ns = len(sat)
    sat_array = np.asarray(sat, dtype=np.int64)
    el_arr = np.asarray(el, dtype=np.float64)

    for sys_idx, sys in enumerate(sys_list):
        sat_idx_list = []
        for k in range(ns):
            sys_k, _ = sat2prn(int(sat_array[k]))
            if sys_k == sys:
                sat_idx_list.append(k)
        if len(sat_idx_list) == 0:
            continue
        ref_pos = sat_idx_list[int(np.argmax(el_arr[sat_idx_list]))]
        if sys == uGNSS.GLO:
            freq0 = obs.sig[sys][uTYP.L][0].frequency(0)
        else:
            freq0 = obs.sig[sys][uTYP.L][0].frequency()

        for f in range(block_stride):
            is_phase = f < nf
            freq_idx = f if is_phase else f - nf
            sig_group = obs.sig[sys][uTYP.L] if is_phase else obs.sig[sys][uTYP.C]
            if freq_idx >= len(sig_group):
                continue
            sig = sig_group[freq_idx]
            block_id = sys_idx * block_stride + f

            for sat_pos in sat_idx_list:
                if sat_pos == ref_pos:
                    continue
                sat_id = int(sat_array[sat_pos])
                if sat_id <= 0 or sat_id > nav.edt.shape[0]:
                    continue
                if np.any(nav.edt[sat_id-1, :] > 0):
                    continue
                if y[ref_pos, f] == 0.0 or y[sat_pos, f] == 0.0:
                    continue

                if sys == uGNSS.GLO:
                    freq = sig.frequency(nav.glo_ch[sat_id])
                else:
                    freq = sig.frequency()
                mu = -(freq0/freq)**2 if is_phase else +(freq0/freq)**2

                if is_phase:
                    ref_sat_id = int(sat_array[ref_pos])
                    if sys == uGNSS.GLO:
                        lam_ref = sig.wavelength(nav.glo_ch[ref_sat_id])
                        lam_sat = sig.wavelength(nav.glo_ch[sat_id])
                    else:
                        lam_ref = sig.wavelength()
                        lam_sat = lam_ref
                else:
                    lam_ref = 0.0
                    lam_sat = 0.0

                ref_indices.append(ref_pos)
                sat_indices.append(sat_pos)
                freq_indices.append(freq_idx)
                col_indices.append(f)
                block_indices.append(block_id)
                mu_values.append(mu)
                lam_ref_values.append(lam_ref)
                lam_sat_values.append(lam_sat)
                is_phase_flags.append(is_phase)
                sig_str = sig.str()
                label_idx = sig_label_map.get(sig_str, -1)
                if label_idx < 0:
                    label_idx = len(sig_label_table)
                    sig_label_map[sig_str] = label_idx
                    sig_label_table.append(sig_str)
                sig_label_indices.append(label_idx)

    return (
        np.asarray(ref_indices, dtype=np.int64),
        np.asarray(sat_indices, dtype=np.int64),
        np.asarray(freq_indices, dtype=np.int64),
        np.asarray(col_indices, dtype=np.int64),
        np.asarray(block_indices, dtype=np.int64),
        np.asarray(mu_values, dtype=np.float64),
        np.asarray(lam_ref_values, dtype=np.float64),
        np.asarray(lam_sat_values, dtype=np.float64),
        np.asarray(is_phase_flags, dtype=np.bool_),
        np.asarray(sys_list, dtype=np.int64),
        np.asarray(sig_label_indices, dtype=np.int64),
        sig_label_table,
    )


def _ddcov_numpy(nb: np.ndarray, Ri: np.ndarray, Rj: np.ndarray, nv: int) -> np.ndarray:
    """Vectorized DD covariance assembly."""

    R = np.zeros((nv, nv), dtype=np.float64)
    if nv == 0 or nb.size == 0:
        return R

    offset = 0
    for count in nb:
        if count <= 0:
            continue
        end = offset + count
        if end > nv:
            end = nv
        rows = slice(offset, end)
        row_vals = Ri[rows]
        block = np.broadcast_to(row_vals[:, None], (end - offset, end - offset)).copy()
        block[np.diag_indices(end - offset)] += Rj[rows]
        R[rows, rows] = block
        offset = end
    return R


@njit(cache=True)
def _zdres_core(
    y_row,
    lam,
    L_vals,
    P_vals,
    col_idx,
    base_range,
    trop,
    iono,
    antr_pr,
    antr_cp,
    ants_pr,
    ants_cp,
    cbias,
    pbias,
    phw,
    ):
    prc_row, cpc_row = _range_corrections(
        trop,
        iono,
        antr_pr,
        antr_cp,
        ants_pr,
        ants_cp,
        cbias,
        pbias,
        phw,
    )
    _fill_residual_row(y_row, lam, L_vals, P_vals, col_idx, base_range, cpc_row, prc_row)
    return prc_row, cpc_row


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
                # update_ambiguities) but keep the measurement
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

    def valpos(self, v, R, thres=4.0):
        """ post-fit residual test """
        nv = len(v)
        fact = thres**2
        for i in range(nv):
            if v[i]**2 <= fact*R[i, i]:
                continue
            if self.nav.monlevel > 1:
                txt = "{:3d} is large: {:8.4f} ({:8.4f})".format(
                    i, v[i], R[i, i])
                if self.nav.fout is None:
                    print(txt)
                else:
                    self.nav.fout.write(txt+"\n")
        return True

    def sdres(self, obs, x, y, e, sat, el):
        """
        SD phase/code residuals

        Parameters
        ----------

        obs : Obs()
            Data structure with observations
        x   :
            State vector elements
        y   :
            Un-differenced corrected observations
        e   :
            Line-of-sight vectors
        sat : np.array of int
            List of satellites
        el  : np.array of float values
            Elevation angles

        Returns
        -------
        v   : np.array of float values
            Residuals of single-difference measurements
        H   : np.array of float values
            Jacobian matrix with partial derivatives of state variables
        R   : np.array of float values
            Covariance matrix of single-difference measurements
        """

        nf = self.nav.nf  # number of frequencies (or signals)
        ns = len(el)  # number of satellites
        nc = len(obs.sig.keys())  # number of constellations

        mode = 1 if len(y) == ns else 0  # 0:DD,1:SD

        nb = np.zeros(2*nc*nf, dtype=int)

        Ri = np.zeros(ns*nf*2)
        Rj = np.zeros(ns*nf*2)

        nv = 0
        b = 0

        H = np.zeros((ns*nf*2, self.nav.nx))
        v = np.zeros(ns*nf*2)

        # Geodetic position
        #
        pos = ecef2pos(x[0:3])
        pos_arr = np.asarray(pos, dtype=np.float64)
        doy = time2doy(obs.t)
        mapfh_sd = np.zeros(ns, dtype=np.float64)
        mapfw_sd = np.zeros(ns, dtype=np.float64)
        for idx_sat in range(ns):
            if el[idx_sat] <= 0.0:
                continue
            mf, mw = _tropmapf_dispatch_ppp(float(doy), pos_arr, float(el[idx_sat]), int(self.nav.trpModel))
            mapfh_sd[idx_sat] = mf
            mapfw_sd[idx_sat] = mw

        (
            ref_idx_arr,
            sat_idx_arr,
            freq_idx_arr,
            col_idx_arr,
            block_idx_arr,
            mu_arr,
            lam_ref_arr,
            lam_sat_arr,
            is_phase_arr,
            sys_list_arr,
            sig_label_idx_arr,
            sig_label_table,
        ) = _sdres_build_plan(obs, sat, el, y, self.nav)

        block_stride = nf * 2
        block_total = int(sys_list_arr.size * block_stride) if sys_list_arr.size > 0 else 0
        meas_count = col_idx_arr.size

        if meas_count == 0:
            R = self.ddcov(np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), 0)
            return np.zeros(0), np.zeros((0, self.nav.nx)), R

        use_trop = 1 if self.nav.ntrop > 0 else 0
        trop_idx = self.IT(self.nav.na) if use_trop else -1

        use_iono = 1 if self.nav.niono > 0 else 0
        iono_i_idx = -np.ones(meas_count, dtype=np.int64)
        iono_j_idx = -np.ones(meas_count, dtype=np.int64)
        if use_iono:
            for idx_meas in range(meas_count):
                sat_i_id = sat[ref_idx_arr[idx_meas]]
                sat_j_id = sat[sat_idx_arr[idx_meas]]
                iono_i_idx[idx_meas] = self.II(sat_i_id, self.nav.na)
                iono_j_idx[idx_meas] = self.II(sat_j_id, self.nav.na)

        amb_i_idx = -np.ones(meas_count, dtype=np.int64)
        amb_j_idx = -np.ones(meas_count, dtype=np.int64)
        for idx_meas in range(meas_count):
            if is_phase_arr[idx_meas]:
                sat_i_id = sat[ref_idx_arr[idx_meas]]
                sat_j_id = sat[sat_idx_arr[idx_meas]]
                freq_idx = freq_idx_arr[idx_meas]
                amb_i_idx[idx_meas] = self.IB(sat_i_id, freq_idx, self.nav.na)
                amb_j_idx[idx_meas] = self.IB(sat_j_id, freq_idx, self.nav.na)

        v, H, Ri, Rj, nb = _sdres_core(
            int(mode),
            int(ns),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(e, dtype=np.float64),
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(el, dtype=np.float64),
            np.ascontiguousarray(ref_idx_arr, dtype=np.int64),
            np.ascontiguousarray(sat_idx_arr, dtype=np.int64),
            np.ascontiguousarray(col_idx_arr, dtype=np.int64),
            np.ascontiguousarray(mu_arr, dtype=np.float64),
            np.ascontiguousarray(lam_ref_arr, dtype=np.float64),
            np.ascontiguousarray(lam_sat_arr, dtype=np.float64),
            np.ascontiguousarray(is_phase_arr, dtype=np.bool_),
            np.ascontiguousarray(iono_i_idx, dtype=np.int64),
            np.ascontiguousarray(iono_j_idx, dtype=np.int64),
            np.ascontiguousarray(amb_i_idx, dtype=np.int64),
            np.ascontiguousarray(amb_j_idx, dtype=np.int64),
            int(trop_idx),
            int(use_trop),
            int(use_iono),
            int(self.nav.nx),
            np.ascontiguousarray(mapfw_sd, dtype=np.float64),
            np.ascontiguousarray(block_idx_arr, dtype=np.int64),
            int(block_total),
            int(nf),
            np.ascontiguousarray(self.nav.eratio, dtype=np.float64),
            np.ascontiguousarray(self.nav.err, dtype=np.float64),
        )

        for idx_meas in range(meas_count):
            if is_phase_arr[idx_meas]:
                freq_idx = int(freq_idx_arr[idx_meas])
                sat_i_id = sat[ref_idx_arr[idx_meas]] - 1
                sat_j_id = sat[sat_idx_arr[idx_meas]] - 1
                if 0 <= sat_i_id < self.nav.vsat.shape[0]:
                    self.nav.vsat[sat_i_id, freq_idx] = 1
                if 0 <= sat_j_id < self.nav.vsat.shape[0]:
                    self.nav.vsat[sat_j_id, freq_idx] = 1

        if self.nav.monlevel > 2:
            if use_trop:
                for idx_meas in range(meas_count):
                    diff_map = mapfw_sd[ref_idx_arr[idx_meas]] - mapfw_sd[sat_idx_arr[idx_meas]]
                    self.nav.fout.write(
                        fmt_ztd.format(
                            time2str(obs.t),
                            trop_idx,
                            trop_idx,
                            diff_map,
                            x[trop_idx],
                            np.sqrt(self.nav.P[trop_idx, trop_idx]),
                        )
                    )
            if use_iono:
                for idx_meas in range(meas_count):
                    label = sig_label_table[sig_label_idx_arr[idx_meas]]
                    idx_i = iono_i_idx[idx_meas]
                    idx_j = iono_j_idx[idx_meas]
                    if idx_i < 0 or idx_j < 0:
                        continue
                    sat_i_id = sat[ref_idx_arr[idx_meas]]
                    sat_j_id = sat[sat_idx_arr[idx_meas]]
                    self.nav.fout.write(
                        fmt_ion.format(
                            time2str(obs.t),
                            sat2id(sat_i_id),
                            sat2id(sat_j_id),
                            label,
                            idx_i,
                            idx_j,
                            mu_arr[idx_meas],
                            x[idx_i],
                            x[idx_j],
                            np.sqrt(self.nav.P[idx_i, idx_i]),
                            np.sqrt(self.nav.P[idx_j, idx_j]),
                        )
                    )
                for idx_meas in range(meas_count):
                    if not is_phase_arr[idx_meas]:
                        continue
                    idx_i = amb_i_idx[idx_meas]
                    idx_j = amb_j_idx[idx_meas]
                    if idx_i < 0 or idx_j < 0:
                        continue
                    label = sig_label_table[sig_label_idx_arr[idx_meas]]
                    sat_i_id = sat[ref_idx_arr[idx_meas]]
                    sat_j_id = sat[sat_idx_arr[idx_meas]]
                    self.nav.fout.write(
                        fmt_amb.format(
                            time2str(obs.t),
                            sat2id(sat_i_id),
                            sat2id(sat_j_id),
                            label,
                            idx_i,
                            idx_j,
                            lam_ref_arr[idx_meas],
                            lam_sat_arr[idx_meas],
                            x[idx_i],
                        x[idx_j],
                        np.sqrt(self.nav.P[idx_i, idx_i]),
                        np.sqrt(self.nav.P[idx_j, idx_j]),
                    )
                )

        if self.nav.monlevel > 1:
            for idx_meas in range(meas_count):
                label = sig_label_table[sig_label_idx_arr[idx_meas]]
                sat_i_id = sat[ref_idx_arr[idx_meas]]
                sat_j_id = sat[sat_idx_arr[idx_meas]]
                self.nav.fout.write(
                    fmt_res.format(
                        time2str(obs.t),
                        sat2id(sat_i_id),
                        sat2id(sat_j_id),
                        label,
                        idx_meas,
                        v[idx_meas],
                        np.sqrt(Ri[idx_meas]),
                        np.sqrt(Rj[idx_meas]),
                    )
                )

        R = self.ddcov(nb, Ri, Rj, meas_count)

        return v, H, R

    def ddcov(self, nb, Ri, Rj, nv):
        """ DD measurement error covariance """
        return _ddcov_numpy(nb, Ri, Rj, nv)

    def holdamb_flags(self):
        """Mark resolved ambiguities as held (nav.fix[i, f]: 2 → 3) without
        running the Kalman update. Use this in pipelines that overwrite
        nav.x / nav.P from another source (e.g. GTSAM marginals) every
        epoch — the kfupdate result would be discarded anyway. Returns
        the number of held ambiguities for sanity checking.
        """
        n_held = 0
        sys_lookup = SAT_SYS_ARR
        nf = self.nav.nf
        fix = self.nav.fix
        for i in range(uGNSS.MAXSAT):
            for f in range(nf):
                if fix[i, f] == 2:
                    fix[i, f] = 3
                    n_held += 1
        return n_held

    def zdres(self, obs, cs, bsx, rs, vs, dts, rr, rtype=1):
        """Non-differential residuals for short-baseline RTK.

        Minimal-core variant: solid-earth tides, antenna PCO/PCV, phase
        wind-up, ionosphere and SSR/OSB biases are all disabled for the
        broadcast-ephemeris double-difference RTK configuration
        (trop_opt=0, iono_opt=0, phw_opt=0, tidecorr=NONE), where each of
        those terms is identically zero or cancels in the rover-base DD.
        Only geometry, satellite clock, Shapiro relativity and the slant
        tropospheric delay remain. cs/bsx are accepted for API
        compatibility and ignored.
        """
        _c = rCST.CLIGHT

        nf = self.nav.nf
        n = len(obs.P)
        y = np.zeros((n, nf * 2))
        el = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = np.asarray(rr, dtype=np.float64).copy()

        # Geodetic position (no tide displacement: tides cancel in the
        # rover-base double difference at RTK baselines).
        pos = ecef2pos(rr_)
        pos_arr = np.asarray(pos, dtype=np.float64)

        # Zenith hydrostatic / wet tropospheric delays at the user position.
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)
        doy = time2doy(obs.t)

        rs_matrix = np.asarray(rs, dtype=np.float64)
        if rs_matrix.ndim == 1:
            rs_matrix = rs_matrix.reshape(1, -1)
        rs_arr = np.ascontiguousarray(rs_matrix[:, 0:3])
        rr_vec = np.ascontiguousarray(rr_)
        (geom_all, los_all, el_all, mapfh_all, mapfw_all,
         relatv_all, valid_mask) = _zdres_geometry_precompute(
            rs_arr, rr_vec, pos_arr, float(self.nav.elmin),
            int(self.nav.trpModel), float(doy))

        (lam_all, frq_all, col_idx_all, L_sel_all, P_sel_all,
         signal_valid_mask) = _zdres_signal_cache(obs, self.nav)

        zero_nf = np.zeros(nf, dtype=np.float64)
        for i in range(n):
            sat = obs.sat[i]
            # Skip edited observations
            if np.any(self.nav.edt[sat - 1, :] > 0):
                continue
            if not signal_valid_mask[i] or not valid_mask[i]:
                continue

            col_idx_arr = col_idx_all[i, :]
            lam_vec = lam_all[i, :]
            L_sel_vec = L_sel_all[i, :]
            P_sel_vec = P_sel_all[i, :]

            e[i, :] = los_all[i, :]
            el[i] = el_all[i]
            trop = mapfh_all[i] * trop_hs + mapfw_all[i] * trop_wet
            base_range = geom_all[i] + relatv_all[i] - _c * dts[i]

            _zdres_core(
                y[i], lam_vec, L_sel_vec, P_sel_vec, col_idx_arr,
                float(base_range), float(trop),
                zero_nf, zero_nf, zero_nf, zero_nf, zero_nf,
                zero_nf, zero_nf, zero_nf)

        return y, e, el
