"""
module for standard PPP positioning
"""

import numpy as np
from numba import njit

from cssrlib.ephemeris import satposs
from cssrlib.gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS, rCST
from cssrlib.gnss import uTropoModel, ecef2pos, tropmodel, time2str, timediff
from cssrlib.gnss import gpst2utc, uIonoModel, time2doy
from cssrlib.ppp import tidedisp, tidedispIERS2010, uTideModel
from cssrlib.ppp import shapiro, windupcorr
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.cssrlib import sCType
from cssrlib.cssrlib import sCSSRTYPE as sc
from cssrlib.mlambda import mlambda
from cssrlib.atmosphere import tropmapf_niell
from cssrlib.constants import CLIGHT, GME
from cssrlib.geometry import geodist, satazel

# format definition for logging
fmt_ztd = "{}         ztd      ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
fmt_ion = "{} {}-{} ion {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f}\n"
fmt_res = "{} {}-{} res {} ({:3d}) {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
fmt_amb = "{} {}-{} amb {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f} {:10.3f}\n"

MIN_SIN_EL = 0.1 * rCST.D2R

_SIG0_TABLE = {
    sc.QZS_MADOCA: {
        uGNSS.GPS: (rSigRnx("GC1W"), rSigRnx("GC2W")),
        uGNSS.GLO: (rSigRnx("RC1C"), rSigRnx("RC2C")),
        uGNSS.GAL: (rSigRnx("EC1C"), rSigRnx("EC5Q")),
        uGNSS.QZS: (rSigRnx("JC1C"), rSigRnx("JC2S")),
    },
    sc.GAL_HAS_SIS: {
        uGNSS.GPS: (rSigRnx("GC1W"), rSigRnx("GC2W")),
        uGNSS.GAL: (rSigRnx("EC1C"), rSigRnx("EC7Q")),
    },
    sc.GAL_HAS_IDD: {
        uGNSS.GPS: (rSigRnx("GC1C"),),
        uGNSS.GLO: (rSigRnx("RC1C"),),
        uGNSS.GAL: (rSigRnx("EC1C"),),
        uGNSS.BDS: (rSigRnx("CC2I"),),
        uGNSS.QZS: (rSigRnx("JC1C"),),
    },
    sc.IGS_SSR: {
        uGNSS.GPS: (rSigRnx("GC1C"),),
        uGNSS.GLO: (rSigRnx("RC1C"),),
        uGNSS.GAL: (rSigRnx("EC1C"),),
        uGNSS.BDS: (rSigRnx("CC2I"),),
        uGNSS.QZS: (rSigRnx("JC1C"),),
    },
    sc.RTCM3_SSR: {
        uGNSS.GPS: (rSigRnx("GC1C"),),
        uGNSS.GLO: (rSigRnx("RC1C"),),
        uGNSS.GAL: (rSigRnx("EC1C"),),
        uGNSS.BDS: (rSigRnx("CC2I"),),
        uGNSS.QZS: (rSigRnx("JC1C"),),
    },
    sc.BDS_PPP: {
        uGNSS.GPS: (rSigRnx("GC1W"), rSigRnx("GC2W")),
        uGNSS.BDS: (rSigRnx("CC6I"),),
    },
    sc.QZS_CLAS: {
        uGNSS.GPS: (rSigRnx("GC1W"), rSigRnx("GC2W")),
    },
    sc.PVS_PPP: {
        uGNSS.GPS: (rSigRnx("GC1C"), rSigRnx("GC5Q")),
        uGNSS.GAL: (rSigRnx("EC1C"), rSigRnx("EC5Q")),
        uGNSS.SBS: (rSigRnx("SC1C"), rSigRnx("SC5Q")),
    },
    sc.SBAS_L1: {
        uGNSS.GPS: (rSigRnx("GC1C"), rSigRnx("GC5Q")),
        uGNSS.GAL: (rSigRnx("EC1C"), rSigRnx("EC5Q")),
        uGNSS.SBS: (rSigRnx("SC1C"), rSigRnx("SC5Q")),
    },
    sc.SBAS_L5: {
        uGNSS.GPS: (rSigRnx("GC1C"), rSigRnx("GC5Q")),
        uGNSS.GAL: (rSigRnx("EC1C"), rSigRnx("EC5Q")),
        uGNSS.SBS: (rSigRnx("SC1C"), rSigRnx("SC5Q")),
    },
}

TROPO_MODEL_SAAST = int(uTropoModel.SAAST)
TROPO_MODEL_HOPF = int(uTropoModel.HOPF)


@njit(cache=True)
def _gather_or_zero(values, indices):
    n = indices.size
    out = np.zeros(n)
    size = values.size
    for i in range(n):
        idx = indices[i]
        if idx >= 0 and idx < size:
            out[i] = values[idx]
    return out


@njit(cache=True)
def _range_corrections(trop, iono, antr_pr, antr_cp, ants_pr, ants_cp,
                       cbias, pbias, phw):
    prc = trop + antr_pr + ants_pr + iono - cbias
    cpc = trop + antr_cp + ants_cp - iono - pbias + phw
    return prc, cpc


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


@njit(cache=True)
def _fill_residual_row(y_row, lam, L_vals, P_vals, col_idx, base_range, cpc_row, prc_row):
    nf = lam.size
    for f in range(nf):
        if col_idx[f] < 0:
            continue
        y_row[f] = L_vals[f]*lam[f] - (base_range + cpc_row[f])
        y_row[f+nf] = P_vals[f] - (base_range + prc_row[f])


@njit(cache=True)
def _compute_bias_bsx(osb_values, ns2m, nf):
    out = np.zeros(nf, dtype=np.float64)
    count = nf if nf < osb_values.size else osb_values.size
    for i in range(count):
        out[i] = -ns2m * osb_values[i]
    return out


@njit(cache=True)
def _combine_cssr_bias(global_bias, regional_bias, nf, flip):
    out = np.zeros(nf, dtype=np.float64)
    gsize = global_bias.size
    rsize = regional_bias.size
    scale = -1.0 if flip else 1.0
    for i in range(nf):
        val = 0.0
        if i < gsize:
            val += global_bias[i]
        if i < rsize:
            val += regional_bias[i]
        out[i] = scale * val
    return out


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
    lam_all = np.zeros((n, nf), dtype=np.float64)
    frq_all = np.zeros((n, nf), dtype=np.float64)
    col_idx_all = -np.ones((n, nf), dtype=np.int64)
    L_sel_all = np.zeros((n, nf), dtype=np.float64)
    P_sel_all = np.zeros((n, nf), dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        sat = obs.sat[i]
        sys, _ = sat2prn(sat)
        sigsCP = obs.sig[sys][uTYP.L]

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

        for f_idx in range(count):
            col = col_idx_row[f_idx]
            if col < 0:
                continue
            if sys == uGNSS.GLO:
                lam_row[f_idx] = sigsCP[col].wavelength(nav.glo_ch[sat])
                frq_row[f_idx] = sigsCP[col].frequency(nav.glo_ch[sat])
            else:
                lam_row[f_idx] = sigsCP[col].wavelength()
                frq_row[f_idx] = sigsCP[col].frequency()
            if col < L_row_arr.size:
                L_sel_row[f_idx] = L_row_arr[col]
            if col < P_row_arr.size:
                P_sel_row[f_idx] = P_row_arr[col]

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


def antModelRx_fast(nav, pos, e_vec, sigs, rtype):
    """Return contiguous receiver antenna corrections with NaNs zeroed."""

    vals = antModelRx(nav, pos, e_vec, sigs, rtype)
    if vals is None:
        return np.zeros(len(sigs), dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


def antModelTx_fast(nav, e_vec, sigs, sat, time, rs, sig0=None):
    """Return contiguous satellite antenna corrections with NaNs zeroed."""

    vals = antModelTx(nav, e_vec, sigs, sat, time, rs, sig0)
    if vals is None:
        return np.zeros(len(sigs), dtype=np.float64)
    arr = np.asarray(vals, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


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

    def udstate(self, obs):
        """ time propagation of states and initialize """

        # First call (nav.t never set): timediff would yield the full TOW
        # (~3.5e5 s) and Phi/process-noise propagation would explode P. Treat
        # this as dt=0 so the time-update is a no-op until obs.t is recorded
        # at the end of process().
        if getattr(self.nav.t, 'time', 0) == 0:
            self.nav.t = obs.t
        tt = timediff(obs.t, self.nav.t)

        ns = len(obs.sat)
        sys = []
        sat = obs.sat
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel,ztd,ion,amb
        #
        nx = self.nav.nx
        Phi = np.eye(nx)
        # if self.nav.niono > 0:
        #    ni = self.nav.na-uGNSS.MAXSAT
        #    Phi[ni:self.nav.na, ni:self.nav.na] = np.zeros(
        #        (uGNSS.MAXSAT, uGNSS.MAXSAT))
        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
        self.nav.P[0:nx, 0:nx] = Phi@self.nav.P[0:nx, 0:nx]@Phi.T

        # Process noise
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*tt

        # Update Kalman filter state elements
        #
        for f in range(self.nav.nf):

            # Reset phase-ambiguity if instantaneous AR
            # or expire obs outage counter
            #
            for i in range(uGNSS.MAXSAT):

                sat_ = i+1
                sys_i, _ = sat2prn(sat_)

                self.nav.outc[i, f] += 1
                reset = (self.nav.outc[i, f] >
                         self.nav.maxout or np.any(self.nav.edt[i, :] > 0))
                if sys_i not in obs.sig.keys():
                    continue

                # Reset ambiguity estimate
                #
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0

                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - reset ambiguity  {}\n"
                            .format(time2str(obs.t), sat2id(sat_),
                                    obs.sig[sys_i][uTYP.L][f]))

                if self.nav.niono > 0:
                    # Reset slant ionospheric delay estimate
                    #
                    j = self.II(sat_, self.nav.na)
                    if reset and self.nav.x[j] != 0.0:
                        self.initx(0.0, 0.0, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write("{}  {} - reset ionosphere\n"
                                                .format(time2str(obs.t),
                                                        sat2id(sat_)))

            # Ambiguity
            #
            bias = np.zeros(ns)
            ion = np.zeros(ns)
            f1 = 0

            """
            offset = 0
            na = 0
            """
            for i in range(ns):

                # Do not initialize invalid observations
                #
                if np.any(self.nav.edt[sat[i]-1, :] > 0):
                    continue

                if self.nav.nf > 1 and self.nav.niono > 0:
                    # Get dual-frequency pseudoranges for this constellation
                    #
                    sig1 = obs.sig[sys[i]][uTYP.C][0]
                    sig2 = obs.sig[sys[i]][uTYP.C][1]

                    pr1 = obs.P[i, 0]
                    pr2 = obs.P[i, 1]

                    # Skip zero observations
                    #
                    if pr1 == 0.0 or pr2 == 0.0:
                        continue

                    if sys[i] == uGNSS.GLO:
                        if sat[i] not in self.nav.glo_ch:
                            print("glonass channel not found: {:d}"
                                  .format(sat[i]))
                            continue
                        f1 = sig1.frequency(self.nav.glo_ch[sat[i]])
                        f2 = sig2.frequency(self.nav.glo_ch[sat[i]])
                    else:
                        f1 = sig1.frequency()
                        f2 = sig2.frequency()

                    # Get iono delay at frequency of first signal
                    #
                    ion[i] = (pr1-pr2)/(1.0-(f1/f2)**2)

                # Get pseudorange and carrier-phase observation of signal f
                #
                sig = obs.sig[sys[i]][uTYP.L][f]

                if sys[i] == uGNSS.GLO:
                    fi = sig.frequency(self.nav.glo_ch[sat[i]])
                else:
                    fi = sig.frequency()

                lam = rCST.CLIGHT/fi

                cp = obs.L[i, f]
                pr = obs.P[i, f]
                if cp == 0.0 or pr == 0.0 or lam is None:
                    continue

                bias[i] = cp - pr/lam + 2.0*ion[i]/lam*(f1/fi)**2

                """
                amb = nav.x[IB(sat[i], f, nav.na)]
                if amb != 0.0:
                    offset += bias[i] - amb
                    na += 1
                """
            """
            # Adjust phase-code coherency
            #
            if na > 0:
                db = offset/na
                for i in range(uGNSS.MAXSAT):
                    if nav.x[IB(i+1, f, nav.na)] != 0.0:
                        nav.x[IB(i+1, f, nav.na)] += db
            """

            # Initialize ambiguity
            #
            for i in range(ns):

                sys_i, _ = sat2prn(sat[i])

                j = self.IB(sat[i], f, self.nav.na)
                if bias[i] != 0.0 and self.nav.x[j] == 0.0:

                    self.initx(bias[i], self.nav.sig_n0**2, j)

                    if self.nav.monlevel > 0:
                        sig = obs.sig[sys_i][uTYP.L][f]
                        self.nav.fout.write(
                            "{}  {} - init  ambiguity  {} {:12.3f}\n"
                            .format(time2str(obs.t), sat2id(sat[i]),
                                    sig, bias[i]))

                if self.nav.niono > 0:
                    j = self.II(sat[i], self.nav.na)
                    if ion[i] != 0 and self.nav.x[j] == 0.0:

                        self.initx(ion[i], self.nav.sig_ion0**2, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write(
                                "{}  {} - init  ionosphere      {:12.3f}\n"
                                .format(time2str(obs.t), sat2id(sat[i]),
                                        ion[i]))

        return 0

    def find_bias(self, cs, sigref, sat, inet=0):
        """ find satellite signal bias from correction """
        nf = len(sigref)
        v = np.zeros(nf)

        if nf == 0:
            return v

        ctype = sigref[0].typ
        if ctype == uTYP.C:
            if cs.lc[inet].cbias is None or \
                    sat not in cs.lc[inet].cbias.keys():
                return v
            sigc = cs.lc[inet].cbias[sat]
        else:
            if cs.lc[inet].pbias is None or \
                    sat not in cs.lc[inet].pbias.keys():
                return v
            sigc = cs.lc[inet].pbias[sat]

        # work-around for Galileo HAS: L2P -> L2W
        if cs.cssrmode in [sc.GAL_HAS_SIS, sc.GAL_HAS_IDD]:
            if ctype == uTYP.C and rSigRnx('GC2P') in sigc.keys():
                sigc[rSigRnx('GC2W')] = sigc[rSigRnx('GC2P')]
            if ctype == uTYP.L and rSigRnx('GL2P') in sigc.keys():
                sigc[rSigRnx('GL2W')] = sigc[rSigRnx('GL2P')]

        for k, sig in enumerate(sigref):
            if sig in sigc.keys():
                v[k] = sigc[sig]
            elif sig.toAtt('X') in sigc.keys():
                v[k] = sigc[sig.toAtt('X')]
        return v

    def zdres(self, obs, cs, bsx, rs, vs, dts, rr, rtype=1):
        """ non-differential residual """

        _c = rCST.CLIGHT
        ns2m = _c*1e-9

        nf = self.nav.nf
        n = len(obs.P)
        y = np.zeros((n, nf*2))
        el = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()

        # Solid Earth tide corrections
        #
        if self.nav.tidecorr == uTideModel.SIMPLE:
            pos = ecef2pos(rr_)
            disp = tidedisp(gpst2utc(obs.t), pos)
        elif self.nav.tidecorr == uTideModel.IERS2010:
            pos = ecef2pos(rr_)
            disp = tidedispIERS2010(gpst2utc(obs.t), pos)
        else:
            disp = np.zeros(3)
        rr_ += disp

        # Geodetic position
        #
        pos = ecef2pos(rr_)
        pos_arr = np.asarray(pos, dtype=np.float64)

        # Zenith tropospheric dry and wet delays at user position
        #
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos,
                                         model=self.nav.trpModel)
        doy = time2doy(obs.t)
        rs_matrix = np.asarray(rs, dtype=np.float64)
        if rs_matrix.ndim == 1:
            rs_matrix = rs_matrix.reshape(1, -1)
        rs_arr = np.ascontiguousarray(rs_matrix[:, 0:3])
        rr_vec = np.ascontiguousarray(np.asarray(rr_, dtype=np.float64))
        geom_all, los_all, el_all, mapfh_all, mapfw_all, relatv_all, valid_mask = _zdres_geometry_precompute(
            rs_arr,
            rr_vec,
            pos_arr,
            float(self.nav.elmin),
            int(self.nav.trpModel),
            float(doy),
        )

        inet_sat_index = {}
        if self.nav.trop_opt == 2 or self.nav.iono_opt == 2:  # from cssr
            inet = cs.find_grid_index(pos)
            dlat, dlon = cs.get_dpos(pos)
            if inet > 0:
                sat_array = np.array(cs.lc[inet].sat_n, dtype=np.int64)
                for idx, sat_id in enumerate(sat_array):
                    inet_sat_index[int(sat_id)] = idx
        else:
            inet = -1

        if self.nav.trop_opt == 2:  # trop from cssr
            trph, trpw = cs.get_trop(dlat, dlon)
            trop_hs0, trop_wet0, _ = tropmodel(obs.t, [pos[0], pos[1], 0],
                                               model=self.nav.trpModel)
            r_hs = trop_hs/trop_hs0
            r_wet = trop_wet/trop_wet0

        if self.nav.iono_opt == 2:  # iono from cssr
            stec = cs.get_stec(dlat, dlon)

        cpc = np.zeros((n, nf))
        prc = np.zeros((n, nf))
        (
            lam_all,
            frq_all,
            col_idx_all,
            L_sel_all,
            P_sel_all,
            signal_valid_mask,
        ) = _zdres_signal_cache(obs, self.nav)

        for i in range(n):

            sat = obs.sat[i]
            sys, _ = sat2prn(sat)

            # Skip edited observations
            #
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            if inet > 0 and sat not in cs.lc[inet].sat_n:
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR = obs.sig[sys][uTYP.C]
            sigsCP = obs.sig[sys][uTYP.L]
            if not signal_valid_mask[i]:
                continue

            col_idx_arr = col_idx_all[i, :]
            lam_vec = lam_all[i, :]
            frq_vec = frq_all[i, :]
            L_sel_vec = L_sel_all[i, :]
            P_sel_vec = P_sel_all[i, :]

            cbias = np.zeros(self.nav.nf, dtype=np.float64)
            pbias = np.zeros(self.nav.nf, dtype=np.float64)

            if self.nav.ephopt == 4:
                cbias_vals = np.asarray(
                    [bsx.getosb(sat, obs.t, s) for s in sigsPR],
                    dtype=np.float64,
                )
                cbias = _compute_bias_bsx(
                    np.ascontiguousarray(cbias_vals, dtype=np.float64),
                    float(ns2m),
                    int(self.nav.nf),
                )
                if sys != uGNSS.GLO:
                    pbias_vals = np.asarray(
                        [bsx.getosb(sat, obs.t, s) for s in sigsCP],
                        dtype=np.float64,
                    )
                    pbias = _compute_bias_bsx(
                        np.ascontiguousarray(pbias_vals, dtype=np.float64),
                        float(ns2m),
                        int(self.nav.nf),
                    )
            elif cs is not None:
                cbias_global = np.zeros(len(sigsPR), dtype=np.float64)
                cbias_regional = np.zeros(len(sigsPR), dtype=np.float64)
                pbias_global = np.zeros(len(sigsCP), dtype=np.float64)
                pbias_regional = np.zeros(len(sigsCP), dtype=np.float64)
                if cs.lc[0].cstat & (1 << sCType.CBIAS):
                    cbias_global += self.find_bias(cs, sigsPR, sat)
                if inet > 0 and cs.lc[inet].cstat & (1 << sCType.CBIAS):
                    cbias_regional += self.find_bias(cs, sigsPR, sat, inet)
                if cs.lc[0].cstat & (1 << sCType.PBIAS):
                    pbias_global += self.find_bias(cs, sigsCP, sat)
                if inet > 0 and cs.lc[inet].cstat & (1 << sCType.PBIAS):
                    pbias_regional += self.find_bias(cs, sigsCP, sat, inet)
                flip = cs.cssrmode in (sc.QZS_CLAS, sc.BDS_PPP, sc.PVS_PPP)
                cbias = _combine_cssr_bias(
                    np.ascontiguousarray(cbias_global, dtype=np.float64),
                    np.ascontiguousarray(cbias_regional, dtype=np.float64),
                    int(self.nav.nf),
                    bool(flip),
                )
                pbias = _combine_cssr_bias(
                    np.ascontiguousarray(pbias_global, dtype=np.float64),
                    np.ascontiguousarray(pbias_regional, dtype=np.float64),
                    int(self.nav.nf),
                    bool(flip),
                )

            # Check for invalid biases
            #
            if np.isnan(cbias).any() or np.isnan(pbias).any():
                if self.nav.monlevel > 3:
                    print("skip invalid cbias/pbias for sat={:d}".format(sat))
                continue

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            if not valid_mask[i]:
                continue

            r = geom_all[i]
            e[i, :] = los_all[i, :]
            el[i] = el_all[i]

            # Shapiro relativistic effect
            #
            relatv = relatv_all[i]

            # Tropospheric delay mapping functions
            #
            mapfh = mapfh_all[i]
            mapfw = mapfw_all[i]

            # Tropospheric delay
            #
            if self.nav.trop_opt == 2:  # from cssr
                trop = mapfh*trph*r_hs+mapfw*trpw*r_wet
            else:
                trop = mapfh*trop_hs + mapfw*trop_wet

            # Ionospheric delay
            #
            if self.nav.iono_opt == 2 and inet > 0:
                idx_l = inet_sat_index.get(int(sat), -1)
                if idx_l >= 0:
                    iono = 40.3e16/(frq_vec*frq_vec)*stec[idx_l]
                else:
                    iono = np.zeros(nf)
            else:
                iono = np.zeros(nf)

            # Phase wind-up effect
            #
            if self.nav.phw_opt > 0:
                phw_mode = (False if self.nav.phw_opt == 2 else True)
                self.nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :],
                                                 rr_, self.nav.phw[sat-1],
                                                 full=phw_mode)

                # cycle -> m
                phw = lam_vec*self.nav.phw[sat-1]
            else:
                phw = np.zeros(nf)

            # Select APC reference signals
            #
            sig0 = None
            if cs is not None and cs.cssrmode in _SIG0_TABLE:
                sig0 = _SIG0_TABLE[cs.cssrmode].get(sys, None)

            # Receiver/satellite antenna offset
            #
            if self.nav.rcv_ant is None:
                antrPR = np.zeros(nf)
                antrCP = np.zeros(nf)
            else:
                ant_rx_pr = antModelRx_fast(self.nav, pos, e[i, :], sigsPR, rtype)
                ant_rx_cp = antModelRx_fast(self.nav, pos, e[i, :], sigsCP, rtype)
                antrPR = _gather_or_zero(ant_rx_pr, col_idx_arr)
                antrCP = _gather_or_zero(ant_rx_cp, col_idx_arr)

            antsPR = np.zeros(nf)
            antsCP = np.zeros(nf)
            if self.nav.ephopt == 4:
                antsPR_all = antModelTx_fast(
                    self.nav, e[i, :], sigsPR, sat, obs.t, rs[i, :]
                )
                antsCP_all = antModelTx_fast(
                    self.nav, e[i, :], sigsCP, sat, obs.t, rs[i, :]
                )
                antsPR = _gather_or_zero(antsPR_all, col_idx_arr)
                antsCP = _gather_or_zero(antsCP_all, col_idx_arr)
            elif cs is not None and cs.cssrmode in (
                sc.QZS_MADOCA, sc.GAL_HAS_SIS, sc.GAL_HAS_IDD,
                sc.IGS_SSR, sc.RTCM3_SSR, sc.BDS_PPP, sc.PVS_PPP
            ) and sig0 is not None:
                antsPR_all = antModelTx_fast(
                    self.nav, e[i, :], sigsPR, sat, obs.t, rs[i, :], sig0
                )
                antsCP_all = antModelTx_fast(
                    self.nav, e[i, :], sigsCP, sat, obs.t, rs[i, :], sig0
                )
                antsPR = _gather_or_zero(antsPR_all, col_idx_arr)
                antsCP = _gather_or_zero(antsCP_all, col_idx_arr)

            # Check for invalid values
            #
            if antrPR is None or antrCP is None or \
               antsPR is None or antsCP is None:
                continue

            iono_vec = np.ascontiguousarray(iono, dtype=np.float64)
            antr_pr_vec = np.ascontiguousarray(antrPR, dtype=np.float64)
            antr_cp_vec = np.ascontiguousarray(antrCP, dtype=np.float64)
            ants_pr_vec = np.ascontiguousarray(antsPR, dtype=np.float64)
            ants_cp_vec = np.ascontiguousarray(antsCP, dtype=np.float64)
            cbias_vec = np.ascontiguousarray(cbias, dtype=np.float64)
            pbias_vec = np.ascontiguousarray(pbias, dtype=np.float64)
            phw_vec = np.ascontiguousarray(phw, dtype=np.float64)

            base_range = r + relatv - _c*dts[i]
            prc_row, cpc_row = _zdres_core(
                y[i],
                lam_vec,
                L_sel_vec,
                P_sel_vec,
                col_idx_arr,
                float(base_range),
                float(trop),
                iono_vec,
                antr_pr_vec,
                antr_cp_vec,
                ants_pr_vec,
                ants_cp_vec,
                cbias_vec,
                pbias_vec,
                phw_vec,
            )
            prc[i, :] = prc_row
            cpc[i, :] = cpc_row

        return y, e, el

    def _valid_double_diff(self, y, iu, ir, f):
        return y[f + iu*self.nav.nf*2] != 0.0 and y[f + ir*self.nav.nf*2] != 0.0

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

    def kfupdate(self, x, P, H, v, R):
        """
        Kalman filter measurement update.

        Parameters:
        x (ndarray): State estimate vector
        P (ndarray): State covariance matrix
        H (ndarray): Observation model matrix
        v (ndarray): Innovation vector
                     (residual between measurement and prediction)
        R (ndarray): Measurement noise covariance

        Returns:
        x (ndarray): Updated state estimate vector
        P (ndarray): Updated state covariance matrix
        S (ndarray): Innovation covariance matrix
        """

        PHt = P@H.T
        S = H@PHt+R
        # K = PHt @ inv(S) computed via solve to avoid the explicit
        # inverse (~2x cheaper for symmetric positive-definite S).
        K = np.linalg.solve(S.T, PHt.T).T
        x += K@v
        IKH = np.eye(P.shape[0])-K@H
        P = IKH@P@IKH.T + K@R@K.T  # Joseph stabilized version

        return x, P, S

    def restamb(self, bias, nb):
        """ restore SD ambiguity """
        nv = 0
        xa = self.nav.x.copy()
        xa[0:self.nav.na] = self.nav.xa[0:self.nav.na]

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

    def ddidx(self, nav, sat):
        """ index for SD to DD transformation matrix D """
        nb = 0
        n = uGNSS.MAXSAT
        na = nav.na
        ix = np.zeros((n, 2), dtype=int)
        nav.fix = np.zeros((n, nav.nf), dtype=int)
        for m in range(uGNSS.GNSSMAX):
            k = na
            for f in range(nav.nf):
                for i in range(k, k+n):
                    sat_i = i-k+1
                    sys, _ = sat2prn(sat_i)
                    if (sys != m):
                        continue
                    if sat_i not in sat or nav.x[i] == 0.0 \
                       or nav.vsat[sat_i-1, f] == 0:
                        continue
                    if nav.el[sat_i-1] >= nav.elmaskar:
                        nav.fix[sat_i-1, f] = 2
                        break
                    else:
                        nav.fix[sat_i-1, f] = 1
                for j in range(k, k+n):
                    sat_j = j-k+1
                    sys, _ = sat2prn(sat_j)
                    if (sys != m):
                        continue
                    if i == j or sat_j not in sat or nav.x[j] == 0.0 \
                       or nav.vsat[sat_j-1, f] == 0:
                        continue
                    if nav.el[sat_j-1] >= nav.elmaskar:
                        ix[nb, :] = [i, j]
                        nb += 1
                        nav.fix[sat_j-1, f] = 2
                k += n
        ix = np.resize(ix, (nb, 2))
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
                # Demote excluded sats' fix flag from 2 → 1 so restamb() and
                # holdamb() only act on the accepted subset.
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
            rr_ = rr

        # Solid Earth tide corrections
        #
        if self.nav.tidecorr == uTideModel.SIMPLE:
            pos = ecef2pos(rr_)
            disp = tidedisp(gpst2utc(obs.t), pos)
        elif self.nav.tidecorr == uTideModel.IERS2010:
            pos = ecef2pos(rr_)
            disp = tidedispIERS2010(gpst2utc(obs.t), pos)
        else:
            disp = np.zeros(3)
        rr_ += disp

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        # Total number of satellites
        #
        ns = uGNSS.MAXSAT

        # Reset previous editing results
        #
        self.nav.edt = np.zeros((ns, self.nav.nf), dtype=int)

        # Build O(1) lookups instead of scanning obs.sat each iteration.
        obs_sat_arr = np.asarray(obs.sat)
        sat_to_idx = {int(s): k for k, s in enumerate(obs_sat_arr)}

        # Loop over all satellites
        #
        sat = []
        for i in range(ns):

            sat_i = i+1
            sys_i, _ = sat2prn(sat_i)

            if sat_i not in sat_to_idx:
                self.nav.edt[i, :] = 1
                continue

            # Check satellite exclusion
            #
            if sat_i in self.nav.excl_sat:
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite excluded\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            j = sat_to_idx[sat_i]

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
            sigsPR = obs.sig[sys_i][uTYP.C]
            sigsCP = obs.sig[sys_i][uTYP.L]
            sigsCN = obs.sig[sys_i][uTYP.S]

            cnr_thresholds = np.zeros(self.nav.nf, dtype=np.float64)
            for f in range(self.nav.nf):
                cnr_thresholds[f] = (self.nav.cnr_min_gpy
                                     if sigsCN[f].isGPS_PY()
                                     else self.nav.cnr_min)

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
                self.nav.edt[i, f] = 1
                if self.nav.monlevel > 0:
                    if code == 1:
                        msg = "edit {:4s} - LLI".format(sigsCP[f].str())
                    elif code == 2:
                        msg = "edit {:4s} - invalid PR obs".format(
                            sigsPR[f].str())
                    elif code == 3:
                        msg = "edit {:4s} - invalid CP obs".format(
                            sigsCP[f].str())
                    else:
                        msg = "edit {:4s} - low C/N0 {:4.1f} dB-Hz".format(
                            sigsCN[f].str(), obs.S[j, f])
                    self.nav.fout.write("{}  {} - {}\n".format(
                        time2str(obs.t), sat2id(sat_i), msg))

            # cycle-slip detection by geometry-free combination
            sig_table = obs.sig if hasattr(obs, 'sig') else None
            sys, _ = sat2prn(sat_i)
            if (
                obs.L.shape[1] > 1
                and sig_table
                and sys in sig_table
                and uTYP.L in sig_table[sys]
                and len(sig_table[sys][uTYP.L]) >= 2
            ):
                L1R, L2R = obs.L[j, 0:2]
                sig1, sig2 = sig_table[sys][uTYP.L][0:2]
                if sys == uGNSS.GLO:
                    lam1 = sig1.wavelength(self.nav.glo_ch[sat_i])
                    lam2 = sig2.wavelength(self.nav.glo_ch[sat_i])
                else:
                    lam1 = sig1.wavelength()
                    lam2 = sig2.wavelength()
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
                    self.nav.edt[i, 0:2] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            " {}  {} - edit {:4s} - GF slip gf0 {:6.3f} gf1 {:6.3f} gf0-gf1 {:6.3f} \n"
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

    def base_process(self, obs, obsb, rs, dts, svh):
        """ processing for base station in RTK
            (implemented in rtkpos) """
        return None, None, None, None

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        PPP/PPP-RTK/RTK positioning
        """

        # Skip empty epochs
        #
        if len(obs.sat) == 0:
            return

        self.nav.nsat[0] = len(obs.sat)

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        self.nav.nsat[1] = nsat

        if nsat < 6:
            print(" too few satellites < 6: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        if obsb is None:  # PPP/PPP-RTK
            # Select satellites having passed quality control
            #
            # index of valid sats in obs.sat
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            ns = len(iu)
            y = np.zeros((ns, self.nav.nf*2))
            e = np.zeros((ns, 3))

            obs_ = obs
        else:  # RTK
            y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
            ns = len(iu)

        self.nav.nsat[2] = ns

        if ns < 6:
            print(" too few satellites < 6: ns={:d}".format(ns))
            return

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs_)

        xa = np.zeros(self.nav.nx)
        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 6 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        # Non-differential residuals after measurement update
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        y = yu[iu, :]
        e = eu[iu, :]
        ny = y.shape[0]
        if ny < 6:
            return -1

        # Residuals for float solution
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        if self.valpos(v, R):
            self.nav.x = xp
            self.nav.P = Pp
            self.nav.ns = 0
            for i in range(ns):
                j = sat[i]-1
                for f in range(self.nav.nf):
                    if self.nav.vsat[j, f] == 0:
                        continue
                    self.nav.outc[j, f] = 0
                    if f == 0:
                        self.nav.ns += 1
        else:
            self.nav.smode = 0

        self.nav.smode = 5  # 4: fixed ambiguities, 5: float ambiguities

        if self.nav.armode > 0:
            nb, xa = self.resamb_lambda(sat, self.nav.parmode, self.nav.par_P0)
            if nb > 0:
                # Use position with fixed ambiguities xa
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[0:3])
                y = yu[iu, :]
                e = eu[iu, :]
                v, H, R = self.sdres(obs, xa, y, e, sat, el)
                # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
                if self.valpos(v, R):
                    if self.nav.armode == 3:     # fix and hold
                        self.holdamb(xa)    # hold fixed ambiguity
                    self.nav.smode = 4           # fix
                else:
                    pass
            else:
                pass

        # Store epoch for solution
        #
        self.nav.t = obs.t

        return 0
