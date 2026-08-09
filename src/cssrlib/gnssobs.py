"""
module for standard PPP positioning
"""

import numpy as np
from numba import njit

from cssrlib.ephemeris import satposs
from cssrlib.gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS, rCST
from cssrlib.gnss import SAT_SYS_ARR
from cssrlib.gnss import uTropoModel, ecef2pos, tropmodel, geodist, satazel
from cssrlib.gnss import time2str, timediff, gpst2utc, tropmapf, uIonoModel
from cssrlib.gnss import time2doy
from cssrlib.atmosphere import tropmapf_niell
from cssrlib.ppp import tidedisp, tidedispIERS2010, uTideModel
from cssrlib.ppp import shapiro, windupcorr
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.cssrlib import sCType
from cssrlib.cssrlib import sCSSRTYPE as sc
from cssrlib.mlambda import mlambda
from cssrlib.state import StateLayout

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

class gnssobs():
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
        # One object owns where every unknown sits; IB/II/IT below read it
        # rather than re-deriving the arithmetic.
        self.layout = StateLayout(
            pmode=self.nav.pmode,
            nf=self.nav.nf,
            ntrop=(1 if self.nav.trop_opt == 1 else 0),
            niono=(uGNSS.MAXSAT if self.nav.iono_opt == 1 else 0))
        self.layout.apply_to(self.nav)

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

    def _prepare_sat_states(self, obs, cs=None, orb=None, pos_pred=None,
                            rs=None, vs=None, dts=None, svh=None):
        """Shared GTSAM front-end helper: satellite states + min-sat count +
        linearisation position.

        Common to the double-difference (rtkpos) and PPP-RTK (ppprtkpos)
        ``prepare_*_measurements`` front-ends. Runs ``satposs`` (applying SSR
        corrections when ``cs``/``orb`` are given) unless pre-computed states
        are passed, and defaults ``pos_pred`` to the current estimate.

        Returns ``(rs, vs, dts, svh, nsat, pos_pred)``.
        """
        if rs is None or vs is None or dts is None or svh is None:
            rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        else:
            nsat = int(np.count_nonzero(~np.isnan(dts)))
        if pos_pred is None:
            pos_pred = self.nav.x[0:3].copy()
        return rs, vs, dts, svh, nsat, np.asarray(pos_pred, dtype=float)

    def IB(self, s, f, na=3):
        """ return index of phase ambiguity """
        return self.layout.ambiguity(s, f, na)

    def II(self, s, na):
        """ return index of slant ionospheric delay estimate """
        return self.layout.iono(s, na)

    def IT(self, na):
        """ return index of zenith tropospheric delay estimate """
        return self.layout.tropo(na)

    @staticmethod
    def nsig_sys(obs, sys):
        """Number of frequency slots this constellation actually carries.

        May be < ``nf`` under a mixed-nf configuration (e.g. GPS L1/L2/L5
        while ``nf=4`` for Galileo E1/E5a/E5b/E6). The unused high slots are
        zero-padded in the obs arrays and treated as absent observations by
        the residual/state loops, so the constellations need not share the
        same signal count.
        """
        return len(obs.sig[sys][uTYP.L])

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

                if f >= self.nsig_sys(obs, sys_i):  # slot not carried (mixed nf)
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

                if f >= self.nsig_sys(obs, sys[i]):  # slot not carried (mixed nf)
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
        K = PHt@np.linalg.inv(S)
        x += K@v
        # P = P - K@H@P
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
        sat_arr = np.ascontiguousarray(sat, dtype=np.int32)
        ix, fix = _ddidx_core(
            sat_arr, nav.x, nav.vsat, nav.el, SAT_SYS_ARR,
            nav.na, nav.nf,
            int(uGNSS.MAXSAT), int(uGNSS.GNSSMAX), nav.elmaskar)
        nav.fix = fix
        return ix

    def resamb_lambda(self, sat, parmode=1, P0=0.995):
        """ resolve integer ambiguity using LAMBDA method

        parmode selects the LAMBDA search (1: full ILS, 2: partial AR); it is
        nav.parmode, not nav.armode -- the latter switches AR on/off and
        fix-and-hold.
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
        if nfix > 0 and (parmode == 2 or s[0] <= 0.0 or
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

        elif parmode == 2 and nfix == 0:
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

        # Loop over all satellites
        #
        sat = []
        for i in range(ns):

            sat_i = i+1
            sys_i, _ = sat2prn(sat_i)

            if sat_i not in obs.sat:
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

            j = np.where(obs.sat == sat_i)[0][0]

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
            self.nav.el[sat_i - 1] = el  # persist for weighting / AR elev mask
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

            # Loop over signals
            #
            for f in range(self.nav.nf):

                # Slot not carried by this constellation (mixed nf): treat as
                # absent. Do NOT set edt -- the downstream
                # "np.any(edt[sat,:]>0)" check would otherwise drop the whole
                # satellite (the padded slot is never observed).
                if f >= len(sigsCP):
                    continue

                # Cycle  slip check by LLI
                #
                if obs.lli[j, f] == 1:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write("{}  {} - edit {:4s} - LLI\n"
                                            .format(time2str(obs.t),
                                                    sat2id(sat_i),
                                                    sigsCP[f].str()))
                    continue

                # Check for measurement consistency
                #
                if obs.P[j, f] == 0.0:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid PR obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsPR[f].str()))
                    continue

                if obs.L[j, f] == 0.0:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid CP obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCP[f].str()))
                    continue

                # Check C/N0
                #
                cnr_min = self.nav.cnr_min_gpy \
                    if sigsCN[f].isGPS_PY() else self.nav.cnr_min
                if obs.S[j, f] < cnr_min:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - low C/N0 {:4.1f} dB-Hz\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCN[f].str(),
                                    obs.S[j, f]))
                    continue

            # cycle-slip detection by geometry-free combination
            # obs.L is nf wide for every system, so the array width alone
            # does not mean this constellation selected two bands: a
            # single-band system (e.g. GLONASS L1 only in an nf=2 setup)
            # has just one entry in sigsCP and cannot form a GF combination.
            if obs.L.shape[1] > 1 and len(sigsCP) > 1:
                L1R, L2R = obs.L[j, 0:2]
                sys, _ = sat2prn(sat_i)
                sig1, sig2 = sigsCP[0:2]
                if sys == uGNSS.GLO:
                    # FDMA channel may be unknown (no GLO eph decoded);
                    # lam=0 keeps the GF test a no-op instead of KeyError.
                    ch = self.nav.glo_ch.get(sat_i)
                    lam1 = sig1.wavelength(ch) if ch is not None else 0.0
                    lam2 = sig2.wavelength(ch) if ch is not None else 0.0
                else:
                    lam1 = sig1.wavelength()
                    lam2 = sig2.wavelength()
                if L1R != 0.0 and L2R != 0.0:
                    gf1 = (L1R*lam1-L2R*lam2)
                    if rr is None:  # rover
                        gf0 = self.nav.gf[sat_i]
                    else:  # base
                        gf0 = self.nav.gf_r[sat_i]
                    if gf1 != 0.0:
                        if rr is None:  # rover
                            self.nav.gf[sat_i] = gf1
                        else:  # base
                            self.nav.gf_r[sat_i] = gf1
                    if gf0 != 0.0 and gf1 != 0.0 and \
                            abs(gf1-gf0) > self.nav.thresslip:
                        self.nav.edt[i, 0:2] = 1
                        if self.nav.monlevel > 0:
                            self.nav.fout.write(" {}  {} - edit {:4s} - GF slip gf0 {:6.3f} gf1 {:6.3f} gf0-gf1 {:6.3f} \n"
                                                .format(time2str(obs.t),
                                                        sat2id(sat_i),
                                                        sig1.str(), gf0, gf1,
                                                        gf0-gf1))

            # Store satellite which have passed all tests, judged over the
            # bands its SYSTEM actually selected (a constellation offering
            # fewer than nav.nf common bands — e.g. GPS L1+L2 in an nf=3
            # setup — is judged on those bands only, so its satellites are
            # not punished for a slot that was never selected). Within the
            # selected bands the classic strict gate applies: any edited
            # band drops the whole satellite — a missing or degraded band
            # on a satellite whose system does provide it is a tracking /
            # multipath canary (admitting L5-less GPS or B1I-only BeiDou-2
            # measurably poisons the urban float solution).
            nf_sys = min(self.nav.nf, len(sigsCP), len(sigsPR))
            if nf_sys <= 0 or np.any(self.nav.edt[i, :nf_sys] > 0):
                self.nav.edt[i, :] = 1
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)

    def process(self, obs, cs=None, orb=None, bsx=None):
        """
        PPP/PPP-RTK positioning

        RTK is not driven from here. The EKF's rover-minus-base residuals
        were removed with the minimal core, leaving this method's old
        ``obsb`` branch without a base ``zdres`` to difference against; use
        ``rtkpos.prepare_double_difference_measurements`` instead, which is
        what the GTSAM examples do.
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

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        iu = np.where(np.isin(obs.sat, sat_ed))[0]
        obs_ = obs

        # y / e are filled from zdres below.
        ns = len(iu)
        y = np.zeros((ns, self.nav.nf*2))
        e = np.zeros((ns, 3))

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

