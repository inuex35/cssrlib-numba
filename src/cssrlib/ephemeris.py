"""
module for ephemeris processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, rCST, sat2prn, timediff, timeadd
from cssrlib.gnss import gtime_t, Geph, Eph, Alm, prn2sat, gpst2time, \
    time2gpst, timeget, time2gst, time2bdt, gst2time, bdt2time, epoch2time
from cssrlib.glonass import (
    deq as glonass_deq,
    glorbit as glonass_glorbit,
    propagate_glonass as glonass_propagate,
)
from cssrlib.geometry import ecef2llh
from cssrlib.orbit import broadcast_orbit
from datetime import datetime
import xml.etree.ElementTree as et

MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13

MAXDTOE_t = {uGNSS.GPS: 7201.0, uGNSS.GAL: 14400.0, uGNSS.QZS: 7201.0,
             uGNSS.BDS: 7201.0, uGNSS.IRN: 7201.0, uGNSS.GLO: 1800.0,
             uGNSS.SBS: 360.0}


# Module-level cache for findeph: maps id(nav_list) -> (per_sat_dict, length).
# Keyed by id() because nav lists don't accept setattr. Invalidated on length
# change (caller appends/replaces ephemerides).
_FINDEPH_CACHE = {}


def _findeph_index(nav):
    key = id(nav)
    n = len(nav)
    cached = _FINDEPH_CACHE.get(key)
    if cached is not None and cached[1] == n:
        return cached[0]
    idx = {}
    for eph_ in nav:
        idx.setdefault(eph_.sat, []).append(eph_)
    _FINDEPH_CACHE[key] = (idx, n)
    # Bound cache size to avoid leaks across many ephemeris streams.
    if len(_FINDEPH_CACHE) > 32:
        _FINDEPH_CACHE.pop(next(iter(_FINDEPH_CACHE)))
    return idx


_FINDEPH_TOE_CACHE = {}


def _findeph_toe(nav):
    """{(sat, mode): (toe_seconds_sorted: ndarray, ephs_list, navidx_list)} cached by id(nav).

    toe_seconds_sorted  -- float64 array of toe values, sorted ascending
    ephs_list           -- list of eph objects aligned with toe_seconds_sorted
    navidx_list         -- list of nav insertion indices aligned with toe_seconds_sorted

    nav_index = position in the original nav list (enumerate order), used to
    reproduce the original linear scan's last-wins-on-ties semantics.
    """
    key = id(nav)
    n = len(nav)
    cached = _FINDEPH_TOE_CACHE.get(key)
    if cached is not None and cached[1] == n:
        return cached[0]
    table = {}
    for nav_idx, eph_ in enumerate(nav):
        table.setdefault((eph_.sat, eph_.mode), []).append(
            (eph_.toe.time + eph_.toe.sec, nav_idx, eph_)
        )
    out = {}
    for k, lst in table.items():
        lst.sort(key=lambda x: x[0])  # sort by toe_seconds ascending
        toes = np.array([x[0] for x in lst], dtype=np.float64)
        ephs = [x[2] for x in lst]
        navidx = [x[1] for x in lst]
        out[k] = (toes, ephs, navidx)
    _FINDEPH_TOE_CACHE[key] = (out, n)
    if len(_FINDEPH_TOE_CACHE) > 32:
        _FINDEPH_TOE_CACHE.pop(next(iter(_FINDEPH_TOE_CACHE)))
    return out


def findeph(nav, t, sat, iode=-1, mode=0):
    """ find ephemeris for sat """
    sys, _ = sat2prn(sat)
    tmax = MAXDTOE_t[sys]
    if iode < 0:
        tbl = _findeph_toe(nav).get((sat, mode))
        if tbl is None:
            return None
        toes, ephs, navidx = tbl
        tt = t.time + t.sec
        lo = int(np.searchsorted(toes, tt - tmax, side='left'))
        hi = int(np.searchsorted(toes, tt + tmax, side='right'))
        best = None; best_dt = None; best_nav = -1
        for k in range(lo, hi):
            dt = abs(tt - toes[k])
            if best_dt is None or dt < best_dt or (dt == best_dt and navidx[k] > best_nav):
                best = ephs[k]; best_dt = dt; best_nav = navidx[k]
        return best
    # iode >= 0: original linear scan (unchanged)
    idx = _findeph_index(nav)
    candidates = idx.get(sat, ())
    eph = None
    t_time = t.time
    t_sec = t.sec
    tmin = tmax + 1.0
    for eph_ in candidates:
        if iode != eph_.iode:
            continue
        if eph_.mode != mode:
            continue
        toe = eph_.toe
        dt = (t_time - toe.time) + (t_sec - toe.sec)
        if dt < 0:
            dt = -dt
        if dt > tmax:
            continue
        return eph_
    return eph


def dtadjust(t1, t2, tw=604800):
    """ calculate delta time considering week-rollover """
    dt = timediff(t1, t2)
    if dt > tw:
        dt -= tw
    elif dt < -tw:
        dt += tw
    return dt


deq = glonass_deq
glorbit = glonass_glorbit


def geph2pos(time: gtime_t, geph: Geph, flg_v=False, TSTEP=30.0):
    """ calculate GLONASS satellite position based on ephemeris """
    dt = timediff(time, geph.toe)
    pos, vel, dts = glonass_propagate(
        dt,
        np.asarray(geph.pos, dtype=np.float64),
        np.asarray(geph.vel, dtype=np.float64),
        np.asarray(geph.acc, dtype=np.float64),
        float(geph.taun),
        float(geph.gamn),
        step=float(TSTEP),
    )
    if flg_v:
        return pos, vel, dts
    return pos, dts


def geph2clk(time: gtime_t, geph: Geph):
    """ calculate GLONASS satellite clock offset based on ephemeris """
    ts = timediff(time, geph.toe)
    t = ts
    for _ in range(2):
        t = ts - (-geph.taun+geph.gamn*t)
    return -geph.taun + geph.gamn*t


def geph2rel(rs, vs):
    return - 2.0*(rs@vs)/(rCST.CLIGHT**2)


def eccentricAnomaly(M, e):
    """
    Compute eccentric anomaly based on mean anomaly and eccentricity
    """
    E = M
    for _ in range(10):
        Eold = E
        sE = np.sin(E)
        E = M+e*sE
        if abs(Eold-E) < 1e-12:
            break

    return E, sE


def sys2MuOmega(sys):
    if sys == uGNSS.GAL:
        mu = rCST.MU_GAL
        omge = rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        mu = rCST.MU_BDS
        omge = rCST.OMGE_BDS
    else:  # GPS,QZS
        mu = rCST.MU_GPS
        omge = rCST.OMGE
    return mu, omge


def eph2pos(t: gtime_t, eph: Eph, flg_v=False):
    """ calculate satellite position based on ephemeris """
    sys, prn = sat2prn(eph.sat)
    mu, omge = sys2MuOmega(sys)
    dt = dtadjust(t, eph.toe)
    A = float(eph.A)
    n0 = np.sqrt(mu/A**3)
    dna = float(eph.deln)
    Ak = A
    if eph.mode > 0:
        dna += 0.5*dt*float(getattr(eph, 'delnd', 0.0))
        Ak += dt*float(getattr(eph, 'Adot', 0.0))
    n = n0+dna
    M = float(eph.M0)+n*dt
    dtc = dtadjust(t, eph.toc)
    is_bds_geo = 1 if (sys == uGNSS.BDS and (prn <= 5 or prn >= 59)) else 0
    sqrt_mu_A = np.sqrt(mu*A)
    rs, vs, dts = broadcast_orbit(
        float(dt),
        float(dtc),
        float(n),
        float(Ak),
        float(M),
        float(eph.e),
        float(eph.omg),
        float(eph.cuc),
        float(eph.cus),
        float(eph.crc),
        float(eph.crs),
        float(eph.cic),
        float(eph.cis),
        float(eph.i0),
        float(eph.idot),
        float(eph.OMG0),
        float(eph.OMGd),
        float(omge),
        float(getattr(eph, 'toes', 0.0)),
        is_bds_geo,
        float(sqrt_mu_A),
        float(eph.af0),
        float(eph.af1),
        float(eph.af2),
        1 if flg_v else 0,
    )
    if flg_v:
        return rs, vs, dts
    return rs, dts


def eph2clk(time, eph):
    """ calculate clock offset based on ephemeris """
    t = timediff(time, eph.toc)
    for _ in range(2):
        t -= eph.af0+eph.af1*t+eph.af2*t**2
    dts = eph.af0+eph.af1*t+eph.af2*t**2
    return dts


def eph2rel(time, eph):
    sys, _ = sat2prn(eph.sat)
    mu, _ = sys2MuOmega(sys)
    dt = dtadjust(time, eph.toe)
    n0 = np.sqrt(mu/eph.A**3)
    dna = eph.deln
    Ak = eph.A
    if eph.mode > 0:
        dna += 0.5*dt*eph.delnd
        Ak += dt*eph.Adot
    n = n0+dna
    M = eph.M0+n*dt
    _, sE = eccentricAnomaly(M, eph.e)
    mu, _ = sys2MuOmega(sys)
    return -2.0*np.sqrt(mu*eph.A)*eph.e*sE/rCST.CLIGHT**2


def satposs(obs, nav, cs=None, orb=None):
    """
    Calculate pos/vel/clk for observed satellites (broadcast ephemeris).

    Positions, velocities and clock offsets are computed at the signal
    transmission epoch (time-of-flight from the pseudorange). The clock is
    corrected for relativity and, for single-frequency standalone use, TGD.

    NOTE: SSR-correction (``cs``) and precise-orbit (``orb``) support was
    removed with the minimal core; both arguments are accepted for
    signature compatibility but ignored.

    Returns
    -------
    rs, vs : np.ndarray
        satellite ECEF position [m] / velocity [m/s]
    dts : np.ndarray
        satellite clock offset [s]
    svh : np.ndarray of int
        satellite health code
    nsat : int
        number of valid satellites
    """

    n = obs.sat.shape[0]
    rs = np.zeros((n, 3))
    vs = np.zeros((n, 3))
    dts = np.zeros(n)
    svh = np.zeros(n, dtype=int)
    nsat = 0
    obs_sig_keys = obs.sig.keys()

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = sat2prn(sat)

        # Skip undesired constellations
        if sys not in obs_sig_keys:
            continue

        pr = obs.P[i, 0]
        t = timeadd(obs.t, -pr/rCST.CLIGHT)

        if sys == uGNSS.GLO:
            geph = findeph(nav.geph, t, sat, mode=0)
            if geph is None:
                svh[i] = 1
                continue
            svh[i] = geph.svh
            dt = geph2clk(t, geph)
            if sat not in nav.glo_ch:
                nav.glo_ch[sat] = geph.frq
        else:
            eph = findeph(nav.eph, t, sat, mode=0)
            if eph is None:
                svh[i] = 1
                continue
            svh[i] = eph.svh
            dt = eph2clk(t, eph)

        # Re-evaluate at clock-corrected transmission time
        t = timeadd(t, -dt)

        if sys == uGNSS.GLO:
            rs[i, :], vs[i, :], dts[i] = geph2pos(t, geph, True)
        else:
            rs[i, :], vs[i, :], dts[i] = eph2pos(t, eph, True)
            if nav.smode == 1 and nav.nf == 1:  # single-freq standalone
                dts[i] -= eph.tgd

        nsat += 1

    return rs, vs, dts, svh, nsat



def loadXmlAlmanac(fname, sys=uGNSS.GAL):
    """ load Galileo Almanac in XML format:
      https://www.gsc-europa.eu/gsc-products/almanac
    """
    alm_t = []
    root = et.parse(fname).getroot()

    dstr = root.find("./header/GAL-header/issueDate").text
    d = datetime.fromisoformat(dstr)
    ep = [d.year, d.month, d.day, d.hour, d.minute, d.second]
    tref = epoch2time(ep)
    week_ref, tow_ref = time2gst(tref)
    week_ref = week_ref//4*4

    h = root.find('body').find('Almanacs')
    for sv in h.findall('svAlmanac'):
        prn = int(sv.find('SVID').text)

        sts_fnav = sv.find('svFNavSignalStatus')
        sts_E5a = int(sts_fnav.find('statusE5a').text)

        sts_inav = sv.find('svINavSignalStatus')
        sts_E5b = int(sts_inav.find('statusE5b').text)
        sts_E1B = int(sts_inav.find('statusE1B').text)

        alm_ = sv.find('almanac')
        sat = prn2sat(sys, prn)

        alm = Alm(sat)
        rA = float(alm_.find('aSqRoot').text) + np.sqrt(29600e3)
        alm.A = rA**2
        alm.e = float(alm_.find('ecc').text)
        deltai = float(alm_.find('deltai').text)*rCST.SC2RAD
        alm.i0 = 56.0*rCST.D2R + deltai
        alm.OMG0 = float(alm_.find('omega0').text)*rCST.SC2RAD
        alm.OMGd = float(alm_.find('omegaDot').text)*rCST.SC2RAD
        alm.omg = float(alm_.find('w').text)*rCST.SC2RAD
        alm.M0 = float(alm_.find('m0').text)*rCST.SC2RAD
        alm.af0 = float(alm_.find('af0').text)
        alm.af1 = float(alm_.find('af1').text)
        alm.ioda = float(alm_.find('iod').text)
        alm.toas = float(alm_.find('t0a').text)
        wna = float(alm_.find('wna').text)

        alm.toa = gst2time(week_ref + wna, alm.toas)
        alm.svh = (sts_E5a << 4) | (sts_E5b << 2) | (sts_E1B)

        alm_t.append(alm)

    return alm_t


def loadyuma(fname, sys=uGNSS.GPS):
    """ load Yuma almanac """
    alm_t = []
    if sys == uGNSS.GPS or sys == uGNSS.QZS:
        week_ref, _ = time2gpst(timeget())
    elif sys == uGNSS.GAL:
        week_ref, _ = time2gst(timeget())
    elif sys == uGNSS.BDS:
        week_ref, _ = time2bdt(timeget())
    else:
        return alm_t
    flg = False

    with open(fname, 'rt') as fh:
        for line in fh:

            v = line.split(':')
            if v[0][0] == '*':  # comment
                continue
            elif v[0] == 'ID':
                prn = int(v[1])
                sat = prn2sat(sys, prn)
                alm = Alm(sat)
                flg = True
            elif v[0] == 'Health':
                alm.svh = int(v[1])
            elif v[0] == 'Eccentricity':
                alm.e = float(v[1])
            elif v[0] == 'Time of Applicability(s)':
                alm.toas = float(v[1])
            elif v[0] == 'Orbital Inclination(rad)':
                alm.i0 = float(v[1])
            elif v[0] == 'Rate of Right Ascen(r/s)':
                alm.OMGd = float(v[1])
            elif v[0] == 'SQRT(A)  (m 1/2)':
                sqrtA = float(v[1])
                alm.A = sqrtA**2
            elif v[0] == 'Right Ascen at Week(rad)' or \
                    v[0] == 'Right Ascen at TOA(rad)':
                alm.OMG0 = float(v[1])
            elif v[0] == 'Argument of Perigee(rad)':
                alm.omg = float(v[1])
            elif v[0] == 'Mean Anom(rad)':
                alm.M0 = float(v[1])
            elif v[0] == 'Af0(s)':
                alm.af0 = float(v[1])
            elif v[0] == 'Af1(s/s)':
                alm.af1 = float(v[1])
            elif v[0] == 'week':
                alm.week = int(v[1])
                alm.week += week_ref//1023*1023
                if alm.week > week_ref:
                    alm.week -= 1023

                alm.sattype = 0
                if sys == uGNSS.GPS or sys == uGNSS.QZS:
                    alm.toa = gpst2time(alm.week, alm.toas)
                elif sys == uGNSS.GAL:
                    alm.toa = gst2time(alm.week, alm.toas)
                elif sys == uGNSS.BDS:
                    alm.toa = bdt2time(alm.week, alm.toas)

                if flg:
                    alm_t.append(alm)
                    flg = False

    return alm_t


def findalm(alm_t, t, sat, tmax=np.inf):
    """ find almanac for sat """
    sys, _ = sat2prn(sat)
    alm = None
    tmin = tmax + 1.0
    for alm_ in alm_t:
        if alm_.sat != sat:
            continue
        dt = abs(timediff(t, alm_.toa))
        if dt > tmax:
            continue
        if dt <= tmin:
            alm = alm_
            tmin = dt

    return alm


def alm2pos(t: gtime_t, alm: Alm):
    """ calculate satellite position based on ephemeris """
    sys, prn = sat2prn(alm.sat)
    if sys == uGNSS.GAL:
        mu = rCST.MU_GAL
        omge = rCST.OMGE_GAL
    elif sys == uGNSS.BDS:
        mu = rCST.MU_BDS
        omge = rCST.OMGE_BDS
    else:  # GPS,QZS
        mu = rCST.MU_GPS
        omge = rCST.OMGE
    dt = dtadjust(t, alm.toa)
    n0 = np.sqrt(mu/alm.A**3)
    M = alm.M0+n0*dt
    E = M
    for _ in range(10):
        Eold = E
        sE = np.sin(E)
        E = M+alm.e*sE
        if abs(Eold-E) < 1e-12:
            break
    cE = np.cos(E)
    u = np.arctan2(np.sqrt(1.0-alm.e**2)*sE, cE-alm.e)+alm.omg
    r = alm.A*(1.0-alm.e*cE)
    i = alm.i0
    Omg = alm.OMG0+(alm.OMGd-omge)*dt-omge*alm.toas
    x, y = r*np.cos(u), r*np.sin(u)
    cosO, sinO = np.cos(Omg), np.sin(Omg)
    cosi, sini = np.cos(i), np.sin(i)

    rs = np.array([x*cosO-y*cosi*sinO,
                   x*sinO+y*cosi*cosO,
                   y*sini])
    dts = alm.af0 + alm.af1*dt

    return rs, dts
