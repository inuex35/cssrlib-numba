"""Precise orbit and clock products (SP3 / CLK)."""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.gnss import timeadd, timediff, gtime_t
from cssrlib.gnss import rCST, uGNSS

import numpy as np
from math import sin, cos


NMAX = 10
MAXDTE = 900.0
EXTERR_CLK = 1e-3
EXTERR_EPH = 5e-7


class peph_t:
    def __init__(self, time=None):
        if time is not None:
            self.time = time
        else:
            self.time = gtime_t()
        self.pos = np.ones((uGNSS.MAXSAT, 4))*np.nan
        self.vel = np.ones((uGNSS.MAXSAT, 4))*np.nan
        self.std = np.zeros((uGNSS.MAXSAT, 4))
        self.vst = np.zeros((uGNSS.MAXSAT, 4))


class peph:
    nsat = 0
    nep = 0
    t0 = None
    week0 = -1
    svid = None
    acc = None
    svpos = None
    svclk = None
    svvel = None
    status = 0
    scl = [0.0, 0.0]

    def __init__(self):
        self.t = None
        self.nmax = 24*12

    def interppol(self, x, y, n):
        for j in range(1, n):
            for i in range(n-j):
                y[i] = (x[i+j]*y[i]-x[i]*y[i+1])/(x[i+j]-x[i])
        return y[0]

    def pephpos(self, time, sat, nav, vare=False, varc=False):
        rs = np.zeros(3)
        dts = np.zeros(2)
        t = np.zeros(NMAX+1)
        p = np.zeros((3, NMAX+1))

        if nav.ne < NMAX+1 or \
                timediff(time, nav.peph[0].time) < -MAXDTE or \
                timediff(time, nav.peph[nav.ne-1].time) > MAXDTE:
            return None, None, False, False
        i, j = 0, nav.ne-1
        while i < j:
            k = (i+j)//2
            if timediff(nav.peph[k].time, time) < 0.0:
                i = k+1
            else:
                j = k
        index = 0 if i <= 0 else i-1

        i = index-(NMAX+1)//2
        if i < 0:
            i = 0
        elif i+NMAX >= nav.ne:
            i = nav.ne-NMAX-1

        for j in range(NMAX+1):
            t[j] = timediff(nav.peph[i+j].time, time)
            if np.linalg.norm(nav.peph[i+j].pos[sat-1, :]) <= 0.0:
                return None, None, False, False

        for j in range(NMAX+1):
            pos = nav.peph[i+j].pos[sat-1, :]
            sinl = sin(rCST.OMGE*t[j])
            cosl = cos(rCST.OMGE*t[j])
            p[0, j] = cosl*pos[0]-sinl*pos[1]
            p[1, j] = sinl*pos[0]+cosl*pos[1]
            p[2, j] = pos[2]

        for i in range(3):
            rs[i] = self.interppol(t, p[i, :], NMAX+1)

        p_ = nav.peph[index:index+2]

        if vare:
            s = np.zeros(3)
            for i in range(3):
                s[i] = p_[0].std[sat-1, i]
            std = np.linalg.norm(s)
            if t[0] > 0.0:
                std += EXTERR_EPH*(t[0]**2)/2.0
            elif t[NMAX] < 0.0:
                std += EXTERR_EPH*(t[NMAX]**2)/2.0
            vare = std**2

        t[0] = timediff(time, p_[0].time)
        t[1] = timediff(time, p_[1].time)

        c = [p_[0].pos[sat-1, 3], p_[1].pos[sat-1, 3]]

        if t[0] <= 0.0:
            dts[0] = c[0]
            if dts[0] != 0.0:
                std = p_[0].std[sat-1, 3]*rCST.CLIGHT-EXTERR_CLK*t[0]
        elif t[1] >= 0.0:
            dts[0] = c[1]
            if dts[0] != 0.0:
                std = p_[1].std[sat-1, 3]*rCST.CLIGHT-EXTERR_CLK*t[1]
        elif c[0] != np.nan and c[1] != np.nan:
            dts[0] = (c[1]*t[0]-c[0]*t[1])/(t[0]-t[1])
            i = 0 if t[0] < -t[1] else 1
            std = p_[i].std[sat-1, 3]+EXTERR_CLK*abs(t[i])
        else:
            dts[0] = np.nan

        if varc:
            varc = std**2

        return rs, dts, vare, varc

    def pephclk(self, time, sat, nav, varc=False):
        dts = np.zeros(2)
        t = np.zeros(NMAX+1)

        if nav.nc < 2 or \
                timediff(time, nav.pclk[0].time) < -MAXDTE or \
                timediff(time, nav.pclk[nav.nc-1].time) > MAXDTE:
            return None, False
        i, j = 0, nav.nc-1
        while i < j:
            k = (i+j)//2
            if timediff(nav.pclk[k].time, time) < 0.0:
                i = k+1
            else:
                j = k
        index = 0 if i <= 0 else i-1
        p_ = nav.pclk[index:index+2]

        t[0] = timediff(time, p_[0].time)
        t[1] = timediff(time, p_[1].time)

        c = [p_[0].clk[sat-1], p_[1].clk[sat-1]]

        if t[0] <= 0.0:
            dts[0] = c[0]
            if dts[0] == 0.0:
                return None, False
            std = p_[0].std[sat-1]*rCST.CLIGHT-EXTERR_CLK*t[0]
        elif t[1] >= 0.0:
            dts[0] = c[1]
            if dts[0] == 0.0:
                return None, False
            std = p_[1].std[sat-1]*rCST.CLIGHT-EXTERR_CLK*t[1]
        elif c[0] != 0.0 and c[1] != 0.0:
            dts[0] = (c[1]*t[0]-c[0]*t[1])/(t[0]-t[1])
            i = 0 if t[0] < -t[1] else 1
            std = p_[i].std[sat-1]+EXTERR_CLK*abs(t[i])
        else:
            return None, False

        if varc:
            varc = std**2

        return dts, varc

    def pephrel(self, rs):
        """ Relativistic correction based on satellite position and velocity """
        return - 2.0*(rs[0:3]@rs[3:6])/(rCST.CLIGHT**2)

    def peph2pos(self, time, sat, nav, var=False):
        """ Satellite position, velocity and clock offset """

        tt = 1e-3

        # Satellite position based on SP3 at epoch
        #
        rss, dtss, vare, varc = self.pephpos(time, sat, nav, var, var)
        if rss is None:
            return None, None, False

        # Satellite clock based on Clock-RINEX
        #
        if nav.nc >= 2:
            dtss, varc = self.pephclk(time, sat, nav, var)
            if dtss is None:
                return None, None, False

        # Satellite position based on SP3 at epoch plus delta t
        #
        time_tt = timeadd(time, tt)
        rst, dtst, _, _ = self.pephpos(time_tt, sat, nav)
        if rss is None:
            return None, None, False

        # Get clock offset from Clock-RINEX
        #
        if nav.nc >= 2:
            dtst, _ = self.pephclk(time_tt, sat, nav)
            if dtst is None:
                return None, None, False

        # Satellite position and velocity (from differentiation)
        #
        rs = np.zeros(6)
        dts = np.zeros(2)

        rs[0:3] = rss
        rs[3:6] = (rst-rss)/tt

        # Apply relativistic correction to clock offset, compute clock rate from
        # differentiation
        #
        if dtss[0] != 0.0:
            dt_rel = self.pephrel(rs)
            dts[0] = dtss[0] + dt_rel
            dts[1] = (dtst[0]-dtss[0])/tt
        else:
            dts = dtss

        if var:
            var = vare+varc

        return rs, dts, var
