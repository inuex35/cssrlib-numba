"""Site displacement and phase effects: solid Earth tides, wind-up, Shapiro.

The Earth-orientation machinery these need -- rotation matrices, IAU1980
nutation, ECI<->ECEF, Sun and Moon -- lives in :mod:`cssrlib.models.frames`.
"""
import cssrlib.gnss as gn
from cssrlib.gnss import gpst2utc, time2epoch
from cssrlib.models.frames import sunmoonpos
from enum import IntEnum
from math import sin, cos, atan2, asin
import numpy as np
try:
    from pysolid.solid import solid_grid
except ImportError:
    solid_grid = None

_warned_no_pysolid = False


def shapiro(rsat, rrcv):
    """ relativistic shapiro effect """
    rs = np.linalg.norm(rsat)
    rr = np.linalg.norm(rrcv)
    rrs = np.linalg.norm(rsat-rrcv)
    corr = (2*gn.rCST.GME/gn.rCST.CLIGHT**2)*np.log((rs+rr+rrs)/(rs+rr-rrs))
    return corr


def windupcorr(time, rs, vs, rr, phw, full=False):
    """ calculate windup correction """
    ek = gn.vnorm(rr-rs)
    if full:
        # Satellite antenna frame unit vectors assuming standard yaw attitude law
        #
        rsun, _, _ = sunmoonpos(gpst2utc(time))
        r = -rs
        ezs = r/np.linalg.norm(r)
        r = rsun-rs
        ess = r/np.linalg.norm(r)
        r = np.cross(ezs, ess)
        eys = r/np.linalg.norm(r)
        exs = np.cross(eys, ezs)
    else:
        we = np.array([0, 0, gn.rCST.OMGE])
        ek = gn.vnorm(rr-rs)
        ezs = gn.vnorm(-rs)
        ess = gn.vnorm(vs+np.cross(we, rs))
        eys = gn.vnorm(np.cross(ezs, ess))
        exs = np.cross(eys, ezs)
    pos = gn.ecef2pos(rr)
    E = gn.xyz2enu(pos)
    exr = E[0, :]
    eyr = E[1, :]
    eks = np.cross(ek, eys)
    ekr = np.cross(ek, eyr)
    ds = exs-ek*(ek@exs)-eks
    dr = exr-ek*(ek@exr)+ekr
    c_p = (ds@dr)/(np.linalg.norm(ds)*np.linalg.norm(dr))
    c_p = max(-1.0, min(1.0, c_p))
    ph = np.arccos(c_p)/(2.0*np.pi)
    drs = np.cross(ds, dr)
    if ek@drs < 0.0:
        ph = -ph
    phw = ph+np.floor(phw-ph+0.5)  # [cycle]
    return phw


class uTideModel(IntEnum):
    """
    Enumeration for Earth tide model selection
    """

    NONE = -1
    SIMPLE = 0
    IERS2010 = 1


def tide_pl(eu, rp, GMp, pos):
    """ pole tide correction """
    H3 = 0.293
    L3 = 0.0156
    r = np.linalg.norm(rp)
    ep = rp/r
    K2 = GMp/gn.rCST.GME*gn.rCST.RE_WGS84**4/r**3
    K3 = K2*gn.rCST.RE_WGS84/r
    latp = asin(ep[2])
    lonp = atan2(ep[1], ep[0])
    c_p = cos(latp)
    c_l = cos(pos[0])
    s_l = sin(pos[0])

    p = (3.0*s_l**2-1.0)/2.0
    H2 = 0.6078-0.0006*p
    L2 = 0.0847+0.0002*p
    a = ep@eu
    a2 = a**2
    dp = K2*3.0*L2*a
    du = K2*(H2*(1.5*a2-0.5)-3.0*L2*a2)

    dp += K3*L3*(7.5*a2-1.5)
    du += K3*a*(H3*(2.5*a2-1.5)-L3*(7.5*a2-1.5))
    dlon = pos[1]-lonp
    du += 3.0/4.0*0.0025*K2*sin(2.0*latp)*sin(2.0*pos[0])*sin(dlon)
    du += 3.0/4.0*0.0022*K2*(c_p*c_l)**2*sin(2.0*dlon)

    dr = dp*ep+du*eu
    return dr


def solid_tide(rsun, rmoon, pos, E, gmst, flag=True):
    """ solid earth tide correction """
    # time domain
    eu = E[2, :]
    dr1 = tide_pl(eu, rsun, gn.rCST.GMS, pos)
    dr2 = tide_pl(eu, rmoon, gn.rCST.GMM, pos)
    # frequency domain
    s_2l = sin(2.0*pos[0])
    du = -0.012*s_2l*sin(gmst+pos[1])

    dr = dr1+dr2+du*eu

    # eliminate permanent tide
    if flag:
        s_l = sin(pos[0])
        du = 0.1196*(1.5*s_l**2-0.5)
        dn = 0.0247*s_2l
        dr += du*E[2, :]+dn*E[1, :]

    return dr


def tidedisp(tutc, pos, erpv=None):
    """ displacement by tide """
    if erpv is None:
        erpv = np.zeros(5)
    rs, rm, gmst = sunmoonpos(tutc, erpv)
    E = gn.xyz2enu(pos)
    dr = solid_tide(rs, rm, pos, E, gmst)
    return dr


def tidedispIERS2010(tutc, pos, erpv=None):
    """
    Wrapper for solid_grid() method of PySolid module to compute Earth tide
    displacement corrections according to the IERS2010 conventions
    """
    if solid_grid is None:  # PySolid not installed
        global _warned_no_pysolid
        if not _warned_no_pysolid:
            import warnings
            warnings.warn(
                "uTideModel.IERS2010 requested but pysolid is not "
                "installed — falling back to the simplified solid-tide "
                "model. pip install pysolid (or cssrlib[tides]).",
                RuntimeWarning)
            _warned_no_pysolid = True
        return tidedisp(tutc, pos, erpv)

    e = time2epoch(tutc)
    disp_e, disp_n, disp_u = solid_grid(e[0], e[1], e[2], e[3], e[4],
                                        int(e[5]),
                                        np.rad2deg(pos[0]), 0, 1,
                                        np.rad2deg(pos[1]), 0, 1)
    E = gn.enu2xyz(pos)
    return E@np.array([disp_e[0, 0], disp_n[0, 0], disp_u[0, 0]])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cssrlib.models.ephemeris import findeph, eph2pos
    from cssrlib.rinex import rnxdec

    tgps_ = gn.epoch2time([2021, 3, 19, 0, 0, 0])
    pos_ = np.array([0.61678759,  2.43512138, 64.94054687])
    erpv_ = np.array([2.1079217879069683e-06, 4.8733853217911866e-07,
                     -0.044509672541668682, -0.0007141, 0])

    flg_tide = False
    flg_pwup = True

    if flg_tide:
        n = 86400//300
        t = np.zeros(n)
        dr_ = np.zeros((n, 3))
        for k in range(n):
            tn = gn.timeadd(tgps_, k*300)
            t[k] = gn.timediff(tn, tgps_)
            dn_ = tidedisp(gn.gpst2utc(tn), pos_, erpv_)
            dr_[k, :] = gn.ecef2enu(pos_, dn_)

        plt.figure()
        plt.plot(t/3600, dr_)
        plt.xlabel('time [h]')
        plt.ylabel('displacement [m]')
        plt.grid()
        plt.axis([0, 24, -0.2, 0.2])
        plt.legend(('east', 'north', 'up'))

    if flg_pwup:
        bdir = '../data/'
        navfile = bdir+'30340780.21q'
        nav = gn.Nav()
        dec = rnxdec()
        nav = dec.decode_nav(navfile, nav)
        rr_ = gn.pos2ecef(pos_)
        sat = gn.prn2sat(gn.uGNSS.QZS, 194)

        n = 86400//300
        t = np.zeros(n)
        ph_ = np.zeros(n)
        d = np.zeros(n)
        phw_ = 0
        for k in range(n):
            tn = gn.timeadd(tgps_, k*300)
            eph = findeph(nav.eph, tn, sat)
            rs_, vs_, dts = eph2pos(tn, eph, True)
            phw_ = windupcorr(tn, rs_, vs_, rr_, phw_)
            t[k] = gn.timediff(tn, tgps_)
            ph_[k] = phw_
            d[k] = shapiro(rs_, rr_)

        plt.figure()
        plt.plot(t/3600, ph_, label='phase windup')
        plt.plot(t/3600, d, label='shapiro')
        plt.xlabel('time [h]')
        plt.ylabel('delta range [m]')
        plt.grid()
        plt.axis([0, 24, -0.2, 0.2])
        plt.legend()
