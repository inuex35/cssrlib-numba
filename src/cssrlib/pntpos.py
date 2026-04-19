"""
module for standalone positioning
"""
import numpy as np
from numba import njit
from cssrlib.constants import D2R, RE_WGS84
from cssrlib.gnss import rCST, ecef2pos, \
    tropmodel, sat2prn, uGNSS, uTropoModel, uIonoModel, \
    timediff, time2gpst, time2doy, dops, uTYP, Obs, time2str
from cssrlib.cssrlib import sCSSRTYPE
from cssrlib.ephemeris import satposs
from cssrlib.pppssr import pppos
from cssrlib.sbas import ionoSBAS
from cssrlib.dgps import vardgps
from math import sin, cos
from cssrlib.ionosphere import klobuchar_delay
from cssrlib.geometry import geodist, satazel
from cssrlib.atmosphere import tropmapf_niell

TROPO_MODEL_SAAST = int(uTropoModel.SAAST)
TROPO_MODEL_HOPF = int(uTropoModel.HOPF)
from cssrlib.orbit import broadcast_orbit


@njit(cache=True)
def _vardgps_variance(el: float, user_height: float, baseline: float) -> float:
    """DGPS measurement variance model rewritten for Numba."""

    denom = RE_WGS84 + user_height
    if denom <= 0.0:
        denom = RE_WGS84
    ratio = RE_WGS84 * np.cos(el) / denom
    if ratio >= 1.0:
        ratio = 0.999999
    elif ratio <= -1.0:
        ratio = -0.999999
    Fpp = 1.0 / np.sqrt(1.0 - ratio * ratio)
    s_iono = Fpp * 0.004 * (baseline + 14.0)
    s_mp = 0.13 + 0.53 * np.exp(-el / 0.1745)
    v_air = 0.11 * 0.11 + s_mp * s_mp
    v_pr = (0.16 + 0.107 * np.exp(-el / 0.271)) ** 2 + 0.08 ** 2
    return v_pr + v_air + s_iono * s_iono


@njit(cache=True)
def _varerr_jit(
    el: float,
    f_idx: int,
    eratio: np.ndarray,
    err: np.ndarray,
    smode: int,
    user_height: float,
    baseline: float,
) -> float:
    """JIT-compatible measurement variance helper."""

    if smode == 2:
        return _vardgps_variance(el, user_height, baseline)

    s_el = np.sin(el)
    min_sin = 0.1 * D2R
    if s_el < min_sin:
        s_el = min_sin
    fact = eratio[f_idx] if f_idx < eratio.size else eratio[-1]
    a = fact * err[1]
    b = fact * err[2]
    return a * a + (b / s_el) ** 2


@njit(cache=True)
def _assemble_sdres_blocks(
    y: np.ndarray,
    e: np.ndarray,
    el: np.ndarray,
    sat_ids: np.ndarray,
    sat_systems: np.ndarray,
    sys_list: np.ndarray,
    nf: int,
    nav_nx: int,
    icb_idx: int,
    eratio: np.ndarray,
    err: np.ndarray,
    smode: int,
    edt_values: np.ndarray,
    user_height: float,
    baseline: float,
):
    """Build SD residual blocks in a vectorised/JIT friendly manner."""

    n_sats = sat_ids.size
    nv_max = n_sats * nf
    v = np.zeros(nv_max, dtype=np.float64)
    H = np.zeros((nv_max, nav_nx), dtype=np.float64)
    Rj = np.zeros(nv_max, dtype=np.float64)
    nb = np.zeros(sys_list.size * nf, dtype=np.int64)

    use_edt = edt_values.size > 0
    edt_rows = edt_values.shape[0] if edt_values.ndim >= 1 else 0
    edt_cols = edt_values.shape[1] if edt_values.ndim == 2 else 0

    nv = 0
    block_index = 0
    for sys_idx in range(sys_list.size):
        sys_code = sys_list[sys_idx]
        for f_idx in range(nf):
            block_len = 0
            for sat_idx in range(n_sats):
                if sat_systems[sat_idx] != sys_code:
                    continue

                if use_edt:
                    sat_number = sat_ids[sat_idx] - 1
                    if sat_number < 0 or sat_number >= edt_rows:
                        continue
                    invalid = False
                    if edt_cols > 0:
                        for col in range(edt_cols):
                            if edt_values[sat_number, col] > 0.0:
                                invalid = True
                                break
                    else:
                        if edt_values[sat_number] > 0.0:
                            invalid = True
                    if invalid:
                        continue

                v[nv] = y[sat_idx, f_idx]
                H[nv, :3] = -e[sat_idx, :]
                H[nv, icb_idx] = 1.0
                Rj[nv] = _varerr_jit(
                    el[sat_idx],
                    f_idx,
                    eratio,
                    err,
                    smode,
                    user_height,
                    baseline,
                )
                nv += 1
                block_len += 1

            if block_len > 0:
                nb[block_index] = block_len
                block_index += 1

    return v[:nv], H[:nv, :], Rj[:nv], nb[:block_index]


@njit(cache=True)
def _tropmapf_dispatch(doy: float, pos: np.ndarray, el: float, model: int):
    if model == TROPO_MODEL_HOPF:
        mapfh = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 72.0) ** 2))
        mapfw = 1.0 / np.sin(np.sqrt(el * el + (np.pi / 120.0) ** 2))
        return mapfh, mapfw
    elif model == TROPO_MODEL_SAAST:
        return tropmapf_niell(doy, pos, el)
    return 0.0, 0.0


@njit(cache=True)
def compute_geometry_and_delays(
    rs: np.ndarray,
    rr: np.ndarray,
    pos: np.ndarray,
    elmin: float,
    doy: float,
    tow: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    trop_hs: float,
    trop_wet: float,
    use_tropo: int,
    use_iono: int,
    trp_model: int,
):
    """Compute geometry, az/el, tropo and iono delays for all satellites."""

    n = rs.shape[0]
    geom = np.zeros(n, dtype=np.float64)
    az_out = np.zeros(n, dtype=np.float64)
    el_out = np.zeros(n, dtype=np.float64)
    los = np.zeros((n, 3), dtype=np.float64)
    trop = np.zeros(n, dtype=np.float64)
    iono = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        rng, los_vec = geodist(rs[i, :], rr)
        geom[i] = rng
        los[i, :] = los_vec
        az_val, el_val = satazel(pos, los_vec)
        az_out[i] = az_val
        el_out[i] = el_val
        if el_val < elmin:
            continue
        valid[i] = True

        if use_tropo:
            mapfh, mapfw = _tropmapf_dispatch(doy, pos, el_val, trp_model)
            trop[i] = mapfh * trop_hs + mapfw * trop_wet

        if use_iono:
            iono[i] = klobuchar_delay(tow, pos[0], pos[1], az_val, el_val,
                                      alpha, beta)

    return geom, az_out, el_out, los, trop, iono, valid

def ionmodel(t, pos, az, el, nav=None, model=uIonoModel.KLOBUCHAR, cs=None):
    """ ionosphere delay estimation """

    if model == uIonoModel.KLOBUCHAR:
        _, tow = time2gpst(t)
        ion_params = getattr(nav, 'ion', None)
        if ion_params is not None and len(ion_params) >= 2:
            alpha = np.asarray(ion_params[0], dtype=np.float64).reshape(4)
            beta = np.asarray(ion_params[1], dtype=np.float64).reshape(4)
        else:
            alpha = np.zeros(4, dtype=np.float64)
            beta = np.zeros(4, dtype=np.float64)
        pos_arr = np.asarray(pos, dtype=np.float64)
        diono = klobuchar_delay(
            float(tow),
            float(pos_arr[0]),
            float(pos_arr[1]),
            float(az),
            float(el),
            alpha,
            beta,
        )
    elif model == uIonoModel.SBAS:
        if cs is None or cs.iodi < 0:
            _, tow = time2gpst(t)
            ion_params = getattr(nav, 'ion', None)
            if ion_params is not None and len(ion_params) >= 2:
                alpha = np.asarray(ion_params[0], dtype=np.float64).reshape(4)
                beta = np.asarray(ion_params[1], dtype=np.float64).reshape(4)
            else:
                alpha = np.zeros(4, dtype=np.float64)
                beta = np.zeros(4, dtype=np.float64)
            pos_arr = np.asarray(pos, dtype=np.float64)
            return klobuchar_delay(
                float(tow),
                float(pos_arr[0]),
                float(pos_arr[1]),
                float(az),
                float(el),
                alpha,
                beta,
            )
        diono, _ = ionoSBAS(t, pos, az, el, cs)
        if diono == 0.0:
            _, tow = time2gpst(t)
            ion_params = getattr(nav, 'ion', None)
            if ion_params is not None and len(ion_params) >= 2:
                alpha = np.asarray(ion_params[0], dtype=np.float64).reshape(4)
                beta = np.asarray(ion_params[1], dtype=np.float64).reshape(4)
            else:
                alpha = np.zeros(4, dtype=np.float64)
                beta = np.zeros(4, dtype=np.float64)
            pos_arr = np.asarray(pos, dtype=np.float64)
            diono = klobuchar_delay(
                float(tow),
                float(pos_arr[0]),
                float(pos_arr[1]),
                float(az),
                float(el),
                alpha,
                beta,
            )

    return diono  # iono delay at L1 [m]


class stdpos(pppos):

    def ICB(self, s=0):
        """ return index of clock bias (s=0), clock drift (s=1) """
        return 3+s if self.nav.pmode == 0 else 6+s

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, trop_opt=0,
                 iono_opt=0, phw_opt=0, csmooth=False, rmode=0):

        self.nav = nav
        self.monlevel = 0

        self.nav.csmooth = csmooth  # carrier-smoothing is enabled/disabled
        self.nav.rmode = rmode  # PR measurement mode

        self.cs_cnt = {}
        self.Lp_ = {}
        self.Ps_ = {}
        self.cs_t0 = {}

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Select iono model
        #
        self.ionoModel = uIonoModel.KLOBUCHAR

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.nav.trop_opt = trop_opt

        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.nav.iono_opt = iono_opt

        self.nav.na = (4 if self.nav.pmode == 0 else 8)
        self.nav.nq = (4 if self.nav.pmode == 0 else 8)

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]            # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]

        self.nav.sig_cb0 = 100.0  # [m]
        self.nav.sig_cd0 = 1.0    # [m/s]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 1.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]

        self.nav.sig_qcb = 0.1
        self.nav.sig_qcd = 0.01

        self.nav.elmin = np.deg2rad(10.0)

        self.nsat = 0
        self.dop = None

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
            dP[6] = self.nav.sig_cb0**2
            dP[7] = self.nav.sig_cd0**2
        else:
            dP[3] = self.nav.sig_cb0**2
        # dP[self.nav.na+1] = self.nav.sig_cd0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2

        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2
            self.nav.q[6] = self.nav.sig_qcb**2
            self.nav.q[7] = self.nav.sig_qcd**2
        else:
            self.nav.q[3] = self.nav.sig_qcb**2

        # Logging level
        #
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def csmooth(self, obs: Obs, sat, Pm, Lm, ns=100, dt_th=1, cs_th=10):
        """ Hatch filter for carrier smoothing """

        if Pm == 0.0 or Lm == 0.0:
            self.cs_cnt[sat] = 1
            return Pm

        if sat not in self.cs_cnt or timediff(obs.t, self.cs_t0[sat]) > dt_th:
            self.cs_cnt[sat] = 1

        if self.cs_cnt[sat] == 1:
            self.Ps_[sat] = Pm
        else:
            Pp = self.Ps_[sat] + (Lm - self.Lp_[sat])  # predicted pseudorange
            if abs(Pm-Pp) < cs_th:
                alp = 1/self.cs_cnt[sat]
                self.Ps_[sat] = alp*Pm + (1-alp)*Pp  # smoothed pseudorange
            else:
                if self.monlevel > 0:
                    print("cycle slip detected, cs reset.")
                self.cs_cnt[sat] = 1
                self.Ps_[sat] = Pm
        self.cs_cnt[sat] = min(self.cs_cnt[sat]+1, ns)
        self.Lp_[sat] = Lm
        self.cs_t0[sat] = obs.t
        return self.Ps_[sat]

    def varerr(self, nav, el, f):
        """ variation of measurement """
        if nav.smode == 2:  # DGPS
            v_sig = vardgps(el, nav)
        else:
            s_el = max(np.sin(el), 0.1*rCST.D2R)
            fact = nav.eratio[f]
            a = fact*nav.err[1]
            b = fact*nav.err[2]
            v_sig = a**2+(b/s_el)**2
        return v_sig

    def udstate(self, obs):
        """ time propagation of states and initialize """

        tt = timediff(obs.t, self.nav.t)

        sys = []
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel,ztd,ion,amb
        #
        nx = self.nav.nx
        Phi = np.eye(nx)

        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            self.nav.x[6] += self.nav.x[7]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
            Phi[6, 7] = tt

        self.nav.P[0:nx, 0:nx] = Phi@self.nav.P[0:nx, 0:nx]@Phi.T

        # Process noise
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*tt

        return 0

    def zdres(self, obs, cs, bsx, rs, vs, dts, x, rtype=1):
        """ non-differential residual """

        _c = rCST.CLIGHT

        nf = self.nav.nf
        n = len(obs.P)
        rr = x[0:3]
        dtr = x[self.ICB()]
        y = np.zeros((n, nf))
        el = np.zeros(n)
        az = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = np.ascontiguousarray(rr.copy(), dtype=np.float64)

        # Geodetic position
        #
        pos = np.asarray(ecef2pos(rr_), dtype=np.float64)

        if self.nav.trop_opt == 0:  # use tropo model
            trop_hs, trop_wet, _ = tropmodel(obs.t, pos, model=self.nav.trpModel)
        else:
            trop_hs = 0.0
            trop_wet = 0.0

        _, tow = time2gpst(obs.t)
        doy = time2doy(obs.t)
        ion_params = getattr(self.nav, 'ion', None)
        if ion_params is not None and len(ion_params) >= 2:
            alpha = np.asarray(ion_params[0], dtype=np.float64).reshape(4)
            beta = np.asarray(ion_params[1], dtype=np.float64).reshape(4)
        else:
            alpha = np.zeros(4, dtype=np.float64)
            beta = np.zeros(4, dtype=np.float64)

        use_tropo = 1 if self.nav.trop_opt == 0 else 0
        use_iono = 1 if self.nav.iono_opt == 0 else 0

        geom_all, az_all, el_all, los_all, trop_all, iono_all, valid_mask = compute_geometry_and_delays(
            np.ascontiguousarray(np.asarray(rs, dtype=np.float64)),
            rr_,
            pos,
            float(self.nav.elmin),
            float(doy),
            float(tow),
            alpha,
            beta,
            float(trop_hs),
            float(trop_wet),
            int(use_tropo),
            int(use_iono),
            int(self.nav.trpModel),
        )

        for i in range(n):

            sat = obs.sat[i]
            sys, _ = sat2prn(sat)

            # Skip edited observations
            #
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            if not valid_mask[i]:
                continue

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            r = geom_all[i]
            e[i, :] = los_all[i]
            az[i] = az_all[i]
            el[i] = el_all[i]

            if self.nav.trop_opt == 0:  # use model
                trop = trop_all[i]
            else:
                trop = 0.0

            if self.nav.iono_opt == 0:  # use model
                iono = iono_all[i]
            else:
                iono = 0.0

            r += dtr - _c*dts[i]

            sigsCP = obs.sig[sys][uTYP.L]
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat])
                                for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])

            if self.nav.rmode == 0:
                PR = obs.P[i, 0]
                CP = lam[0]*obs.L[i, 0]
            else:  # iono-free combination
                iono = 0.0
                if self.nav.rmode == 1:  # L1/L2 iono free combination
                    gam = (rCST.FREQ_G1/rCST.FREQ_G2)**2
                if self.nav.rmode == 2:  # L1/L5 iono free combination
                    gam = (rCST.FREQ_S1/rCST.FREQ_S5)**2
                PR = (obs.P[i, 1]-gam*obs.P[i, 0])/(1-gam)
                CP = (lam[1]*obs.L[i, 1]-gam*lam[0]*obs.L[i, 0])/(1-gam)

            if self.nav.csmooth:  # carrier smoothing for pseudo-range
                PR = self.csmooth(obs, sat, PR, CP)

            y[i, 0] = PR-(r+trop + iono)

        return y, e, az, el

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

        nf = self.nav.nf if self.nav.rmode == 0 else 1
        icb_idx = self.ICB()

        sat_ids = np.ascontiguousarray(np.asarray(sat, dtype=np.int64))
        sat_systems = np.empty(sat_ids.size, dtype=np.int64)
        for idx, sat_id in enumerate(sat_ids):
            sys_idx, _ = sat2prn(int(sat_id))
            sat_systems[idx] = int(sys_idx)

        sig_map = getattr(obs, 'sig', {})
        if sig_map:
            sys_order = np.array([int(sys) for sys in sig_map.keys()], dtype=np.int64)
        else:
            sys_order = np.unique(sat_systems).astype(np.int64)
        sys_order = np.ascontiguousarray(sys_order)

        edt = getattr(self.nav, 'edt', None)
        if edt is not None:
            edt_values = np.ascontiguousarray(np.asarray(edt, dtype=np.float64))
        else:
            edt_values = np.empty((0,), dtype=np.float64)

        eratio = np.ascontiguousarray(np.asarray(self.nav.eratio, dtype=np.float64))
        err = np.ascontiguousarray(np.asarray(self.nav.err, dtype=np.float64))

        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        e_arr = np.ascontiguousarray(np.asarray(e, dtype=np.float64))
        el_arr = np.ascontiguousarray(np.asarray(el, dtype=np.float64))

        user_height = 0.0
        baseline = 0.0
        if self.nav.smode == 2:
            user_height = float(ecef2pos(self.nav.x[0:3])[2])
            baseline = float(getattr(self.nav, 'baseline', 0.0))

        if sat_ids.size == 0 or sys_order.size == 0:
            v = np.zeros(0, dtype=float)
            H = np.zeros((0, self.nav.nx), dtype=float)
            Rj = np.zeros(0, dtype=float)
            nb = np.zeros(0, dtype=int)
        else:
            v, H, Rj, nb = _assemble_sdres_blocks(
                y_arr,
                e_arr,
                el_arr,
                sat_ids,
                sat_systems,
                sys_order,
                int(nf),
                int(self.nav.nx),
                int(icb_idx),
                eratio,
                err,
                int(self.nav.smode),
                edt_values,
                float(user_height),
                float(baseline),
            )

        R = self.ddcov(nb, nb.size, Rj, v.size)

        return v, H, R

    def ddcov(self, nb, n, Rj, nv):
        """ DD measurement error covariance """
        R = np.zeros((nv, nv))
        k = 0
        for b in range(n):
            for j in range(nb[b]):
                R[k+j, k+j] = Rj[k+j]

            k += nb[b]
        return R

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        standalone positioning
        """
        if len(obs.sat) == 0:
            return

        if cs is not None and cs.cssrmode == sCSSRTYPE.DGPS:
            self.nav.smode = 2  # DGPS
            self.nav.baseline = cs.set_dgps_corr(self.nav.x[0:3])

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        if nsat < 4:
            print(" too few satellites < 4: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        if obsb is None:  # standalone
            # Select satellites having passed quality control
            #
            # index of valid sats in obs.sat
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            ns = len(iu)
            y = np.zeros((ns, self.nav.nf))
            e = np.zeros((ns, 3))

            obs_ = obs
        else:  # DGPS
            y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
            ns = len(iu)

        if ns < 4:
            self.nav.t = obs.t
            self.nsat = ns
            print(" too few satellites < 4: ns={:d}".format(ns))
            return

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs_)

        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, azu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp)

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        az = azu[iu]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 4 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 4:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 1
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        if abs(np.mean(v)) > 100.0:  # clock bias initialize/reset
            ic = self.ICB()
            idx_ = np.where(v != 0.0)[0]
            xp[ic] = np.mean(v[idx_])
            v[idx_] -= xp[ic]
            if self.monlevel > 0:
                print("{:s} clock reset.".format(time2str(obs.t)))

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        self.nav.x = xp
        self.nav.P = Pp

        self.nav.smode = 1 if cs is None else 2  # standalone positioning
        # self.nav.smode = 1  # 4: fixed ambiguities, 5: float ambiguities

        # Store epoch for solution
        #
        self.nav.t = obs.t
        self.dop = dops(az, el)
        self.nsat = len(el)

        return 0
