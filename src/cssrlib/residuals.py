"""The observation model: undifferenced and single-difference residuals.

zdres builds corrected residuals for one receiver; sdres differences them
between satellites and forms the design matrix. The Numba kernels that
carry the inner loops live here with them."""

import numpy as np
from numba import njit

from cssrlib.gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS, rCST
from cssrlib.gnss import uTropoModel, ecef2pos, tropmodel, geodist, satazel
from cssrlib.gnss import time2str, gpst2utc, tropmapf
from cssrlib.gnss import time2doy
from cssrlib.atmosphere import tropmapf_niell
from cssrlib.ppp import tidedisp, tidedispIERS2010, uTideModel
from cssrlib.ppp import shapiro, windupcorr
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.cssrlib import sCType
from cssrlib.cssrlib import sCSSRTYPE as sc

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
def _tropmapf_dispatch_ppp(doy, pos, el, model):
    if model == TROPO_MODEL_HOPF:
        mapfh = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 72.0) ** 2))
        mapfw = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 120.0) ** 2))
        return mapfh, mapfw
    elif model == TROPO_MODEL_SAAST:
        return tropmapf_niell(doy, pos, el)
    return 0.0, 0.0


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


def _ddcov_numpy(nb, Ri, Rj, nv):
    """Vectorized DD measurement-error covariance assembly."""
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
        n_blk = end - offset
        block = np.broadcast_to(Ri[rows][:, None], (n_blk, n_blk)).copy()
        block[np.diag_indices(n_blk)] += Rj[rows]
        R[rows, rows] = block
        offset = end
    return R

class ObservationModelMixin:
    """Residual computation, mixed into :class:`~cssrlib.gnssobs.gnssobs`."""

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

        # Zenith tropospheric dry and wet delays at user position
        #
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos,
                                         model=self.nav.trpModel)

        if self.nav.trop_opt == 2 or self.nav.iono_opt == 2:  # from cssr
            inet = cs.find_grid_index(pos)
            dlat, dlon = cs.get_dpos(pos)
            cs.inet = inet
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

            # Wavelength
            #
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat])
                                for s in sigsCP])
                frq = np.array([s.frequency(self.nav.glo_ch[sat])
                               for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])
                frq = np.array([s.frequency() for s in sigsCP])

            # Per-signal arrays span this constellation's signal count (nsf);
            # the unused high slots of the nf-wide prc/cpc/y rows stay zero
            # (see nsig_sys).
            nsf = len(sigsCP)
            cbias = np.zeros(nsf)
            pbias = np.zeros(nsf)

            if self.nav.ephopt == 4:  # from Bias-SINEX

                # Code and phase signal bias, converted from [ns] to [m]
                # Note: IGS uses sign convention different with RTCM
                cbias = np.array(
                    [-bsx.getosb(sat, obs.t, s)*ns2m for s in sigsPR])
                if sys != uGNSS.GLO:
                    pbias = np.array(
                        [-bsx.getosb(sat, obs.t, s)*ns2m for s in sigsCP])

            elif cs is not None:  # from CSSR

                if cs.lc[0].cstat & (1 << sCType.CBIAS) == (1 << sCType.CBIAS):
                    cbias = self.find_bias(cs, sigsPR, sat)

                if inet > 0 and cs.lc[inet].cstat & (1 << sCType.CBIAS) == \
                        (1 << sCType.CBIAS):
                    cbias += self.find_bias(cs, sigsPR, sat, inet)

                if cs.lc[0].cstat & (1 << sCType.PBIAS) == (1 << sCType.PBIAS):
                    pbias = self.find_bias(cs, sigsCP, sat)

                if inet > 0 and cs.lc[inet].cstat & (1 << sCType.PBIAS) == \
                        (1 << sCType.PBIAS):
                    pbias += self.find_bias(cs, sigsCP, sat, inet)

                # note: some services use sign convention different with RTCM
                if cs.cssrmode in [sc.QZS_CLAS, sc.BDS_PPP, sc.PVS_PPP]:
                    pbias = -pbias
                    cbias = -cbias

            # Check for invalid biases
            #
            if np.isnan(cbias).any() or np.isnan(pbias).any():
                if self.nav.monlevel > 3:
                    print("skip invalid cbias/pbias for sat={:d}".format(sat))
                continue

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            r, e[i, :] = geodist(rs[i, :], rr_)
            _, el[i] = satazel(pos, e[i, :])
            if el[i] < self.nav.elmin:
                continue

            # Shapiro relativistic effect
            #
            relatv = shapiro(rs[i, :], rr_)

            # Tropospheric delay mapping functions
            #
            mapfh, mapfw = tropmapf(obs.t, pos, el[i],
                                    model=self.nav.trpModel)

            # Tropospheric delay
            #
            if self.nav.trop_opt == 2:  # from cssr
                trop = mapfh*trph*r_hs+mapfw*trpw*r_wet
            else:
                trop = mapfh*trop_hs + mapfw*trop_wet

            # Ionospheric delay
            #
            if self.nav.iono_opt == 2 and inet > 0:  # from cssr
                idx_l = cs.lc[inet].sat_n.index(sat)
                iono = np.array([40.3e16/(f*f)*stec[idx_l] for f in frq])
            else:
                iono = np.zeros(nsf)

            # Phase wind-up effect
            #
            if self.nav.phw_opt > 0:
                phw_mode = (False if self.nav.phw_opt == 2 else True)
                self.nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :],
                                                 rr_, self.nav.phw[sat-1],
                                                 full=phw_mode)

                # cycle -> m
                phw = lam*self.nav.phw[sat-1]
            else:
                phw = np.zeros(nsf)

            # Select APC reference signals
            #
            sig0 = None
            if cs is not None:

                if cs.cssrmode == sc.QZS_MADOCA:

                    if sys == uGNSS.GPS:
                        sig0 = (rSigRnx("GC1W"), rSigRnx("GC2W"))
                    elif sys == uGNSS.GLO:
                        sig0 = (rSigRnx("RC1C"), rSigRnx("RC2C"))
                    elif sys == uGNSS.GAL:
                        sig0 = (rSigRnx("EC1C"), rSigRnx("EC5Q"))
                    elif sys == uGNSS.QZS:
                        sig0 = (rSigRnx("JC1C"), rSigRnx("JC2S"))

                elif cs.cssrmode == sc.GAL_HAS_SIS:

                    if sys == uGNSS.GPS:
                        sig0 = (rSigRnx("GC1W"), rSigRnx("GC2W"))
                    elif sys == uGNSS.GAL:
                        sig0 = (rSigRnx("EC1C"), rSigRnx("EC7Q"))

                elif cs.cssrmode in (sc.GAL_HAS_IDD, sc.IGS_SSR, sc.RTCM3_SSR):

                    if sys == uGNSS.GPS:
                        sig0 = (rSigRnx("GC1C"),)
                    elif sys == uGNSS.GLO:
                        sig0 = (rSigRnx("RC1C"),)
                    elif sys == uGNSS.GAL:
                        sig0 = (rSigRnx("EC1C"),)
                    elif sys == uGNSS.BDS:
                        sig0 = (rSigRnx("CC2I"),)
                    elif sys == uGNSS.QZS:
                        sig0 = (rSigRnx("JC1C"),)

                elif cs.cssrmode == sc.BDS_PPP:

                    if sys == uGNSS.GPS:
                        sig0 = (rSigRnx("GC1W"), rSigRnx("GC2W"))
                    elif sys == uGNSS.BDS:
                        sig0 = (rSigRnx("CC6I"),)

                elif cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                    if sys == uGNSS.GPS:
                        sig0 = (rSigRnx("GC1C"), rSigRnx("GC5Q"))
                    elif sys == uGNSS.GAL:
                        sig0 = (rSigRnx("EC1C"), rSigRnx("EC5Q"))
                    elif sys == uGNSS.SBS:
                        sig0 = (rSigRnx("SC1C"), rSigRnx("SC5Q"))

            # Receiver/satellite antenna offset
            #
            if self.nav.rcv_ant is None:
                # ndarray, not a list: these are summed element-wise with the
                # trop / iono terms below, where a list would concatenate (or
                # raise on float + list).
                antrPR = np.zeros(len(sigsPR))
                antrCP = np.zeros(len(sigsCP))
            else:
                antrPR = antModelRx(self.nav, pos, e[i, :], sigsPR, rtype)
                antrCP = antModelRx(self.nav, pos, e[i, :], sigsCP, rtype)

            if self.nav.ephopt == 4:

                antsPR = antModelTx(
                    self.nav, e[i, :], sigsPR, sat, obs.t, rs[i, :])
                antsCP = antModelTx(
                    self.nav, e[i, :], sigsCP, sat, obs.t, rs[i, :])

            elif cs is not None and cs.cssrmode in (sc.QZS_MADOCA,
                                                    sc.GAL_HAS_SIS,
                                                    sc.GAL_HAS_IDD,
                                                    sc.IGS_SSR,
                                                    sc.RTCM3_SSR,
                                                    sc.BDS_PPP,
                                                    sc.PVS_PPP):

                antsPR = antModelTx(self.nav, e[i, :], sigsPR,
                                    sat, obs.t, rs[i, :], sig0)
                antsCP = antModelTx(self.nav, e[i, :], sigsCP,
                                    sat, obs.t, rs[i, :], sig0)

            else:

                antsPR = np.zeros(len(sigsPR))
                antsCP = np.zeros(len(sigsCP))

            # Check for invalid values
            #
            if antrPR is None or antrCP is None or \
               antsPR is None or antsCP is None:
                continue

            # Range correction
            # (only the nsf valid slots; high slots stay zero for mixed nf)
            #
            prc[i, :nsf] = trop + antrPR + antsPR + iono - cbias
            cpc[i, :nsf] = trop + antrCP + antsCP - iono - pbias + phw

            r += relatv - _c*dts[i]

            for f in range(nsf):
                y[i, f] = obs.L[i, f]*lam[f]-(r+cpc[i, f])
                y[i, f+nf] = obs.P[i, f]-(r+prc[i, f])

        return y, e, el

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

        mode = 1 if len(y) == ns else 0  # 0:DD,1:SD

        # v / H / Ri / Rj / nb are all allocated and filled by _sdres_core
        # below, sized from the measurement plan rather than the ns*nf*2
        # upper bound the scalar loop used to need.

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
            R = self.ddcov(np.zeros(0, dtype=np.int64), 0,
                           np.zeros(0), np.zeros(0), 0)
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

        R = self.ddcov(nb, len(nb), Ri, Rj, meas_count)

        return v, H, R

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

    def varerr(self, nav, el, f):
        """ variation of measurement """
        s_el = max(np.sin(el), 0.1*rCST.D2R)
        fact = nav.eratio[f-nav.nf] if f >= nav.nf else 1
        a = fact*nav.err[1]
        b = fact*nav.err[2]
        return (a**2+(b/s_el)**2)

    def ddcov(self, nb, n, Ri, Rj, nv):
        """ DD measurement error covariance

        ``n`` is kept for call compatibility; only the first ``n`` block
        sizes of ``nb`` are used, as before.
        """
        nb_arr = np.ascontiguousarray(nb[:n], dtype=np.int64)
        return _ddcov_numpy(nb_arr,
                            np.ascontiguousarray(Ri, dtype=np.float64),
                            np.ascontiguousarray(Rj, dtype=np.float64),
                            int(nv))
