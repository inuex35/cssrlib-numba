"""
module for ephemeris processing
"""

from cssrlib.core.ssr_types import sCType
from cssrlib.core.ssr_types import sCSSRTYPE as sc
import numpy as np
from cssrlib.gnss import uGNSS, rCST, sat2prn, timediff, timeadd, vnorm
from cssrlib.gnss import gtime_t, Geph, Eph
from cssrlib.core.orbit import broadcast_orbit
from cssrlib.models.glonass import propagate_glonass

MAXDTOE_t = {uGNSS.GPS: 7201.0, uGNSS.GAL: 14400.0, uGNSS.QZS: 7201.0,
             uGNSS.BDS: 7201.0, uGNSS.IRN: 7201.0, uGNSS.GLO: 1800.0,
             uGNSS.SBS: 360.0}


def findeph(nav, t, sat, iode=-1, mode=0):
    """ find ephemeris for sat """
    sys, _ = sat2prn(sat)
    eph = None
    tmax = MAXDTOE_t[sys]
    tmin = tmax + 1.0
    for eph_ in nav:
        if eph_.sat != sat or (iode >= 0 and iode != eph_.iode):
            continue
        if eph_.mode != mode:
            continue
        dt = abs(timediff(t, eph_.toe))
        if dt > tmax or eph_.mode != mode:
            continue
        if iode >= 0:
            return eph_
        if dt <= tmin:
            eph = eph_
            tmin = dt

    # RINEX-4 fallback: a constellation may be broadcast only under a
    # non-default navigation message (e.g. BeiDou-3 as B-CNAV1/2/3, mode
    # 1/2/3, with no legacy D1/D2 mode-0 records). If nothing matched the
    # requested message type, retry ignoring ``mode`` so the nearest-in-time
    # ephemeris is still selected. Restricted to the plain broadcast lookup
    # (mode == 0 and iode < 0); IODE/mode-keyed SSR lookups keep the strict
    # behaviour, so this can only add results where the old code returned None.
    if eph is None and iode < 0 and mode == 0:
        tmin = tmax + 1.0
        for eph_ in nav:
            if eph_.sat != sat:
                continue
            dt = abs(timediff(t, eph_.toe))
            if dt > tmax:
                continue
            if dt <= tmin:
                eph = eph_
                tmin = dt

    return eph


def dtadjust(t1, t2, tw=604800):
    """ calculate delta time considering week-rollover """
    dt = timediff(t1, t2)
    if dt > tw:
        dt -= tw
    elif dt < -tw:
        dt += tw
    return dt


def geph2pos(time: gtime_t, geph: Geph, flg_v=False, TSTEP=1.0):
    """ calculate GLONASS satellite position based on ephemeris

    The RK4 integration itself lives in :mod:`cssrlib.models.glonass`, which
    compiles it with Numba. This module used to carry a second, pure-Python
    copy of the same integrator and call that instead, so every GLONASS
    satellite of every epoch ran the slow one -- 40x, measured over a 900 s
    propagation -- while the compiled kernel sat unreferenced.
    """
    t = timediff(time, geph.toe)
    rs, vs, dts = propagate_glonass(t, geph.pos, geph.vel, geph.acc,
                                    geph.taun, geph.gamn, step=TSTEP)

    if flg_v:
        return rs, vs, dts
    else:
        return rs, dts


def geph2clk(time: gtime_t, geph: Geph):
    """ calculate GLONASS satellite clock offset based on ephemeris """
    ts = timediff(time, geph.toe)
    t = ts
    for _ in range(2):
        t = ts - (-geph.taun+geph.gamn*t)
    return -geph.taun + geph.gamn*t


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
    """ calculate satellite position based on ephemeris

    The Keplerian propagation is :func:`cssrlib.core.orbit.broadcast_orbit`,
    which Numba compiles; what stays here is unpacking ``Eph`` into the
    scalars it takes. That kernel existed for a long time with no caller
    while this function did the arithmetic in NumPy, so the two drifted: the
    kernel had lost the ``Adot`` and ``delnd`` velocity terms and derived the
    harmonic rate from the wrong basis. There is now one copy.

    ``M = M0 + (n0 + deln) dt + delnd dt^2 / 2``, so ``dM/dt = n + delnd
    dt / 2``, and the semi-major axis drifts at ``Adot``. Both terms exist
    only for CNAV ephemerides (``eph.mode > 0``).
    """
    sys, prn = sat2prn(eph.sat)
    mu, omge = sys2MuOmega(sys)
    dt = dtadjust(t, eph.toe)

    dna, Ak, nd, Akd = eph.deln, eph.A, 0.0, 0.0
    if eph.mode > 0:
        dna += 0.5*dt*eph.delnd
        Ak += dt*eph.Adot
        nd = 0.5*eph.delnd*dt
        Akd = eph.Adot
    n = np.sqrt(mu/eph.A**3)+dna

    is_geo = 1 if (sys == uGNSS.BDS and (prn <= 5 or prn >= 59)) else 0

    rs, vs, dts = broadcast_orbit(
        dt, dtadjust(t, eph.toc), n, nd, Ak, Akd, eph.M0+n*dt, eph.e,
        eph.omg, eph.cuc, eph.cus, eph.crc, eph.crs, eph.cic, eph.cis,
        eph.i0, eph.idot, eph.OMG0, eph.OMGd, omge, eph.toes, is_geo,
        np.sqrt(mu*eph.A), eph.af0, eph.af1, eph.af2, 1 if flg_v else 0)

    return (rs, vs, dts) if flg_v else (rs, dts)


def eph2clk(time, eph):
    """ calculate clock offset based on ephemeris """
    t = timediff(time, eph.toc)
    for _ in range(2):
        t -= eph.af0+eph.af1*t+eph.af2*t**2
    dts = eph.af0+eph.af1*t+eph.af2*t**2
    return dts


def satposs(obs, nav, cs=None, orb=None):
    """
    Calculate pos/vel/clk for observed satellites

    The satellite position, velocity and clock offset are computed at
    transmission epoch. The signal time-of-flight is computed from
    a pseudorange measurement corrected by the satellite clock offset,
    hence the observations are required at this stage. The satellite clock
    is already corrected for the relativistic effects. The satellite health
    indicator is extracted from the broadcast navigation message.

    Parameters
    ----------
    obs : Obs()
        contains GNSS measurements
    nav : Nav()
        contains coarse satellite orbit and clock offset information
    cs  : cssr_has()
        contains precise SSR corrections for satellite orbit and clock offset
    obs : peph()
        contains precise satellite orbit and clock offset information

    Returns
    -------
    rs  : np.array() of float
        satellite position in ECEF [m]
    vs  : np.array() of float
        satellite velocities in ECEF [m/s]
    dts : np.array() of float
        satellite clock offsets [s]
    svh : np.array() of int
        satellite health code [-]
    nsat : int
        number of effective satellite
    """

    n = obs.sat.shape[0]
    rs = np.zeros((n, 3))
    vs = np.zeros((n, 3))
    dts = np.zeros(n)
    svh = np.zeros(n, dtype=int)
    iode = -1
    nsat = 0

    for i in range(n):

        sat = obs.sat[i]
        sys, _ = sat2prn(sat)

        # Skip undesired constellations
        #
        if sys not in obs.sig.keys():
            continue

        pr = obs.P[i, 0]  # TODO: catch invalid observation!
        t = timeadd(obs.t, -pr/rCST.CLIGHT)

        if nav.ephopt == 4:

            rs_, dts_, _ = orb.peph2pos(t, sat, nav)
            if rs_ is None or dts_ is None or np.isnan(dts_[0]):
                continue
            dt = dts_[0]

            if sys == uGNSS.GLO and len(nav.geph) > 0:
                geph = findeph(nav.geph, t, sat)
                if geph is None:
                    svh[i] = 1
                    continue
                svh[i] = geph.svh

                if sat not in nav.glo_ch:
                    nav.glo_ch[sat] = geph.frq

            elif len(nav.eph) > 0:
                eph = findeph(nav.eph, t, sat)
                if eph is None:
                    svh[i] = 1
                    continue
                svh[i] = eph.svh

            else:
                svh[i] = 0

        else:

            if cs is not None:

                if cs.iodssr >= 0 and cs.iodssr_c[sCType.ORBIT] == cs.iodssr:
                    if sat not in cs.sat_n:
                        continue
                elif cs.iodssr_p >= 0 and \
                        cs.iodssr_c[sCType.ORBIT] == cs.iodssr_p:
                    if sat not in cs.sat_n_p:
                        continue
                else:
                    continue

                if sat not in cs.lc[0].iode.keys():
                    continue

                iode = cs.lc[0].iode[sat]
                dorb = cs.lc[0].dorb[sat]  # radial,along-track,cross-track

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                    dorb += cs.lc[0].dvel[sat] * \
                        (timediff(obs.t, cs.lc[0].t0[sat][sCType.ORBIT]))

                if cs.cssrmode == sc.BDS_PPP:  # consistency check for IOD corr

                    if cs.lc[0].iodc[sat] == cs.lc[0].iodc_c[sat]:
                        dclk = cs.lc[0].dclk[sat]
                    else:
                        if cs.lc[0].iodc[sat] == cs.lc[0].iodc_c_p[sat]:
                            dclk = cs.lc[0].dclk_p[sat]
                        else:
                            continue

                else:

                    if cs.cssrmode == sc.GAL_HAS_SIS:  # HAS only
                        if cs.mask_id != cs.mask_id_clk:  # mask has changed
                            if sat not in cs.sat_n_p:
                                continue
                    else:
                        if cs.iodssr_c[sCType.CLOCK] == cs.iodssr:
                            if sat not in cs.sat_n:
                                continue
                        else:
                            if cs.iodssr_c[sCType.CLOCK] == cs.iodssr_p:
                                if sat not in cs.sat_n_p:
                                    continue
                            else:
                                continue

                    if sat in cs.lc[0].dclk:
                        dclk = cs.lc[0].dclk[sat]
                    else:
                        continue

                    if cs.lc[0].cstat & (1 << sCType.HCLOCK) and \
                            sat in cs.lc[0].hclk.keys() and \
                            not np.isnan(cs.lc[0].hclk[sat]):
                        dclk += cs.lc[0].hclk[sat]

                    if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                        dclk += cs.lc[0].ddft[sat] * \
                            (timediff(obs.t, cs.lc[0].t0[sat][sCType.CLOCK]))

                if np.isnan(dclk) or np.isnan(dorb@dorb):
                    continue

                mode = cs.nav_mode[sys]

            else:

                mode = 0

            if sys == uGNSS.GLO:
                geph = findeph(nav.geph, t, sat, iode, mode=mode)
                if geph is None:
                    svh[i] = 1
                    continue

                svh[i] = geph.svh
                dt = geph2clk(t, geph)

                if sat not in nav.glo_ch:
                    nav.glo_ch[sat] = geph.frq

            else:
                eph = findeph(nav.eph, t, sat, iode, mode=mode)
                if eph is None:
                    svh[i] = 1
                    continue

                svh[i] = eph.svh
                dt = eph2clk(t, eph)

        t = timeadd(t, -dt)

        if nav.ephopt == 4:  # precise ephemeris

            rs_, dts_, _ = orb.peph2pos(t, sat, nav)
            rs[i, :] = rs_[0: 3]
            vs[i, :] = rs_[3: 6]
            dts[i] = dts_[0]
            nsat += 1

        else:

            if sys == uGNSS.GLO:
                rs[i, :], vs[i, :], dts[i] = geph2pos(t, geph, True)
            else:
                rs[i, :], vs[i, :], dts[i] = eph2pos(t, eph, True)

            # Apply SSR correction
            #
            if cs is not None:

                if cs.cssrmode == sc.BDS_PPP:
                    er = vnorm(rs[i, :])
                    rc = np.cross(rs[i, :], vs[i, :])
                    ec = vnorm(rc)
                    ea = np.cross(ec, er)
                    A = np.array([er, ea, ec])
                else:
                    ea = vnorm(vs[i, :])
                    rc = np.cross(rs[i, :], vs[i, :])
                    ec = vnorm(rc)
                    er = np.cross(ea, ec)
                    A = np.array([er, ea, ec])

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5):
                    dorb_e = dorb
                else:
                    dorb_e = dorb@A

                rs[i, :] -= dorb_e
                dts[i] += dclk/rCST.CLIGHT

                if cs.cssrmode in (sc.PVS_PPP, sc.SBAS_L1, sc.SBAS_L5,
                                   sc.DGPS) and sys == uGNSS.GPS:
                    dts[i] -= eph.tgd

                ers = vnorm(rs[i, :]-nav.x[0: 3])
                dorb_ = -ers@dorb_e
                sis = dclk-dorb_
                if cs.lc[0].t0[sat][sCType.ORBIT].time % 30 == 0 and \
                        timediff(cs.lc[0].t0[sat][sCType.ORBIT], nav.time_p) > 0:
                    if abs(nav.sis[sat]) > 0:
                        nav.dsis[sat] = sis - nav.sis[sat]
                    nav.sis[sat] = sis

                nav.dorb[sat] = dorb_
                nav.dclk[sat] = dclk

            elif nav.smode == 1 and nav.nf == 1:  # stand-alone positioning
                dts[i] -= eph.tgd

            nsat += 1

    if cs is not None:
        if sat in cs.lc[0].t0 and sCType.ORBIT in cs.lc[0].t0[sat]:
            nav.time_p = cs.lc[0].t0[sat][sCType.ORBIT]

    return rs, vs, dts, svh, nsat
