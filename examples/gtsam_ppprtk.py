"""GTSAM PPP-RTK (QZSS CLAS) using cssrlib as the observation front-end.

Architecture (gtsam-first):
  cssrlib  -> ppprtkpos.prepare_ppp_measurements(obs, cs): CLAS-corrected
              satellite states, undifferenced corrected residuals (zdres),
              tropo mapping / iono coeff / wavelength, and CLAS atmosphere
              a-priori sigmas. No EKF.
  GTSAM    -> UndifferencedPseudorangeFactor / UndifferencedCarrierPhaseFactor
              (C++, from the inuex35/gtsam fork) estimate
              [static position, per-epoch clock, ZTD resid, slant-iono resid,
               float ambiguity]; CLAS STEC/tropo quality -> tight priors.

Static rover, float ambiguities. Needs the cssrlib-data repo + ANTEX
(see examples/ppprtk_clas.py) and the custom gtsam build with the
Undifferenced* factors.
"""
import os
from copy import deepcopy
from binascii import unhexlify
import numpy as np

from cssrlib.cssrlib import cssr
from cssrlib.gnss import (ecef2pos, Nav, time2gpst, time2doy,
                          rSigRnx, epoch2time, sat2prn, uGNSS)
from cssrlib.gnss import geodist as cssr_geodist
from cssrlib.peph import atxdec, searchpcv
from cssrlib.ppprtk import ppprtkpos
from cssrlib.rinex import rnxdec
import gtsam
from gtsam import symbol

from gnss_ar import ARSession

CLIGHT = 299792458.0
DATADIR = os.environ.get(
    'CSSRLIB_DATA',
    os.path.join(os.path.dirname(__file__), '..', 'cssrlib-data', 'data'))
ep = [2025, 8, 21, 7, 0, 0]
xyz_ref = np.array([-3962108.7007, 3381309.5532, 3668678.6648])
pos_ref = ecef2pos(xyz_ref)
SYSS = (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS)
NEP = int(os.environ.get('NEP', '120'))

time = epoch2time(ep)
doy = int(time2doy(time))
let = chr(ord('a') + ep[3])
bdir = f'{DATADIR}/doy{ep[0]:04d}-{doy:03d}/'

sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("EC1C"), rSigRnx("EC5Q"),
        rSigRnx("JC1C"), rSigRnx("JC2L"),
        rSigRnx("GL1C"), rSigRnx("GL2W"), rSigRnx("EL1C"), rSigRnx("EL5Q"),
        rSigRnx("JL1C"), rSigRnx("JL2L"),
        rSigRnx("GS1C"), rSigRnx("GS2W"), rSigRnx("ES1C"), rSigRnx("ES5Q"),
        rSigRnx("JS1C"), rSigRnx("JS2L")]

nav = Nav()
nav = rnxdec().decode_nav(bdir + f'{doy:03d}{let}_rnx.nav', nav)
atx = atxdec()
atx.readpcv(f'{DATADIR}/antex/igs20.atx')
rnx = rnxdec()
rnx.setSignals(sigs)
cs = cssr()
cs.monlevel = 0
cs.week = time2gpst(time)[0]
cs.read_griddef(f'{DATADIR}/clas_grid.def')

assert rnx.decode_obsh(bdir + f'{doy:03d}{let}_rnx.obs') >= 0
rnx.autoSubstituteSignals()
ppp = ppprtkpos(nav, rnx.pos)
nav.rcv_ant = searchpcv(atx.pcvr, rnx.ant, rnx.ts)
nav.sat_ant = atx.pcvs
cs.find_grid_index(ecef2pos(rnx.pos))
nf = nav.nf

v = np.genfromtxt(bdir + f'{doy:03d}{let}_qzsl6.txt',
                  dtype=[('wn', 'int'), ('tow', 'int'), ('prn', 'int'),
                         ('type', 'int'), ('len', 'int'), ('nav', 'S500')])

# ---- collect the gtsam front-end output over the run -----------------------
frames = []
obs = rnx.decode_obs()
while time > obs.t and obs.t.time != 0:
    obs = rnx.decode_obs()
for k in range(NEP):
    week, tow = time2gpst(obs.t)
    vi = v[(v['tow'] == tow) & (v['type'] == 0) & (v['prn'] == 199)]
    if len(vi) > 0:
        cs.decode_l6msg(unhexlify(vi['nav'][0]), 0)
        if cs.fcnt == 5:
            cs.decode_cssr(bytes(cs.buff), 0)
    if k == 0:
        nav.t = deepcopy(obs.t)
        t0 = deepcopy(obs.t)
        t0.time = t0.time // 30 * 30
        cs.time = obs.t
        nav.time_p = t0
    if cs.chk_stat():
        ppm = ppp.prepare_ppp_measurements(obs, cs=cs, pos_pred=rnx.pos)
        if ppm is not None and k >= 20:
            frames.append(ppm)
    obs = rnx.decode_obs()
    if obs.t.time == 0:
        break

print(f"collected {len(frames)} epochs via prepare_ppp_measurements")

# ---- GTSAM float PPP-RTK graph (C++ Undifferenced factors) -----------------
X = symbol('x', 0)
def CK(ei, si): return symbol('c', ei * 4 + si)
def ZT(ei): return symbol('z', ei)   # ZTD residual per epoch (random walk)
def IO(s): return symbol('i', int(s))
def AM(s, f): return symbol('n', int(s) * 4 + f)

# Incremental ISAM2 (QR): one update per epoch (like gtsam_rtk.py and
# tightly-coupled-gnss-imu-fgo) instead of a one-shot batch optimization.
params = gtsam.ISAM2Params()
params.setFactorization('QR')
isam = gtsam.ISAM2(params)
x0 = xyz_ref + np.array([5.0, -4.0, 3.0])
ztd_sigs = [fr.ztd_sig for fr in frames if np.isfinite(fr.ztd_sig)]
ztd_sig = float(np.median(ztd_sigs)) if ztd_sigs else 0.1


def ztd_rw():
    def err(this, values, jac):
        a = values.atDouble(this.keys()[0]); b = values.atDouble(this.keys()[1])
        if jac is not None:
            jac[0] = np.array([[1.0]]); jac[1] = np.array([[-1.0]])
        return np.array([a - b])
    return err


seen_io, seen_am, seen_ck = set(), set(), set()
ar = ARSession(ppp, isam, X, AM, nf, SYSS, pos_ref, xyz_ref, conv_sigma=1.0)

nfac = 0
for ei, fr in enumerate(frames):
    graph = gtsam.NonlinearFactorGraph()
    val = gtsam.Values()
    if ei == 0:
        val.insert(X, gtsam.Point3(*x0))
        graph.add(gtsam.PriorFactorPoint3(X, gtsam.Point3(*x0),
                  gtsam.noiseModel.Isotropic.Sigma(3, 30.0)))
    rr = fr.pos_pred
    val.insert(ZT(ei), 0.0)
    if ei == 0:
        graph.addPriorDouble(ZT(0), 0.0,
                             gtsam.noiseModel.Isotropic.Sigma(1, ztd_sig))
    else:
        graph.add(gtsam.CustomFactor(
            gtsam.noiseModel.Isotropic.Sigma(1, 0.003),
            gtsam.KeyVector([ZT(ei), ZT(ei - 1)]), ztd_rw()))
    for i, s in enumerate(fr.sat):
        s = int(s)
        sys = sat2prn(s)[0]
        if sys not in SYSS or fr.el[i] <= 0:
            continue
        if not np.all(np.isfinite(fr.rs[i])) or np.linalg.norm(fr.rs[i]) < 1e6:
            continue
        # Require both frequencies (phase+code) so the per-epoch slant iono is
        # observable from the dual-frequency data (avoids rank deficiency).
        if not (fr.y[i, 0] != 0 and fr.y[i, 1] != 0
                and fr.y[i, nf] != 0 and fr.y[i, nf + 1] != 0):
            continue
        geom, _ = cssr_geodist(fr.rs[i], rr)
        s_el = 1.0 / max(np.sin(fr.el[i]), 0.1)
        sysi = SYSS.index(sys)
        ck = CK(ei, sysi)
        for f in range(nf):
            lam = fr.lam[i, f]
            mu = fr.mu[i, f]
            if lam <= 0 or mu <= 0 or fr.y[i, f] == 0 or fr.y[i, nf + f] == 0:
                continue
            m_phase = fr.y[i, f] + geom
            m_code = fr.y[i, nf + f] + geom
            if ck not in seen_ck:
                seen_ck.add(ck)
                val.insert(ck, float(m_code - geom))
                graph.addPriorDouble(ck, 0.0,
                                     gtsam.noiseModel.Isotropic.Sigma(1, 1e5))
            if IO(s) not in seen_io:
                seen_io.add(IO(s))
                val.insert(IO(s), 0.0)
                sig_i = min(fr.iono_sig[i] if np.isfinite(fr.iono_sig[i]) else 0.05, 0.01)
                graph.addPriorDouble(IO(s), 0.0,
                                     gtsam.noiseModel.Isotropic.Sigma(1, sig_i))
            graph.add(gtsam.UndifferencedPseudorangeFactor(
                X, ck, ZT(ei), IO(s), m_code, gtsam.Point3(*fr.rs[i]),
                fr.mapfw[i], mu, 0.0,
                gtsam.noiseModel.Isotropic.Sigma(1, 0.6 * s_el)))
            ak = AM(s, f)
            if ak not in seen_am:
                seen_am.add(ak)
                val.insert(ak, float((m_phase - geom - val.atDouble(ck)) / lam))
                graph.addPriorDouble(ak, val.atDouble(ak),
                                     gtsam.noiseModel.Isotropic.Sigma(1, 5.0))
            graph.add(gtsam.UndifferencedCarrierPhaseFactor(
                X, ck, ZT(ei), IO(s), ak, m_phase, gtsam.Point3(*fr.rs[i]),
                fr.mapfw[i], mu, lam, 0.0,
                gtsam.noiseModel.Isotropic.Sigma(1, 0.006 * s_el)))
    nfac += graph.size()
    isam.update(graph, val)
    ar.step(ei, isam.calculateEstimate(), seen_am, fr.sat, fr.el)

print(f"updates: {len(frames)} epochs, {nfac} factors, "
      f"{len(seen_am)} ambiguities")
ar.summary()
