"""Precise orbit and clock products (SP3 / CLK)."""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.domain.gnss import id2sat, sat2id
from cssrlib.domain.gnss import time2epoch, timeadd, timediff, gtime_t
from cssrlib.domain.gnss import str2time, time2doy
from cssrlib.domain.gnss import time2gpst
from cssrlib.domain.gnss import rCST, uGNSS

import numpy as np
from math import pow, sin, cos


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

    def parse_satlist(self, line):
        n = len(line[9:60].strip())//3
        for k in range(n):
            svid = line[9+3*k:12+3*k]
            if int(svid[1:]) > 0:
                self.sat[self.cnt] = id2sat(svid)
                self.cnt += 1

    def parse_acclist(self, line):
        n = len(line[9:60].strip())//3
        for k in range(n):
            acc = int(line[9+3*k:12+3*k])
            if self.cnt < self.nsat:
                self.acc[self.cnt] = acc
            self.cnt += 1

    def parse_sp3(self, fname, nav, opt=0):
        ver_t = ['c', 'd']
        self.status = 0
        v = False
        with open(fname, "r") as fh:
            for line in fh:
                if line[0:3] == 'EOF':  # end of record
                    break
                if line[0:2] == '/*':  # skip comment
                    continue
                if line[0:2] == '* ':  # header of body part
                    self.status = 10

                if self.status == 0:
                    if line[0] != '#':
                        break
                    self.ver = line[1]
                    if self.ver not in ver_t:
                        print("invalid version: {:s}".format(self.ver))
                        break
                    self.flag = line[2]
                    if self.flag not in ('P', 'V'):
                        print("invalid P/V flag: {:s}".format(self.flag))
                        break
                    self.t0 = str2time(line, 3, 27)
                    self.status = 1
                elif self.status == 1:
                    if line[0:2] != '##':
                        break
                    self.week0 = int(line[3:7])
                    # print("week={:4d}".format(self.week0))
                    self.status = 2
                elif self.status == 2:
                    if line[0:2] == '+ ':
                        self.cnt = 0
                        self.nsat = int(line[3:6])
                        self.sat = np.zeros(self.nsat, dtype=int)
                        self.acc = np.zeros(self.nsat, dtype=int)
                        self.parse_satlist(line)
                        self.status = 3
                        continue
                elif self.status == 3:
                    if line[0:2] == '++':
                        self.cnt = 0
                        self.status = 4
                        self.parse_acclist(line)
                        continue
                    self.parse_satlist(line)
                elif self.status == 4:
                    if line[0:2] == '%c':
                        # two %c, two %f, two %i records
                        # Columns 10-12 in the first %c record define
                        # what time system is used for the date/times
                        # in the ephemeris.
                        # c4-5 File type
                        # c10-12 Time System
                        line = fh.readline()  # %c
                        line = fh.readline()  # %f
                        if line[0:2] != '%f':
                            break
                        # Base for Pos/Vel  (mm or 10**-4 mm/sec)
                        # Base for Clk/Rate (psec or 10**-4 psec/sec)
                        self.scl[0] = float(line[3:13])
                        self.scl[1] = float(line[14:26])
                        self.status = 10
                        for _ in range(3):
                            line = fh.readline()
                        continue
                    self.parse_acclist(line)

                if self.status == 10:  # body
                    if line[0:2] == '* ':  # epoch header
                        v = False
                        nav.ne += 1
                        self.cnt = 0
                        peph = peph_t(str2time(line, 3, 27))
                        # ep = time2epoch(peph.time)
                        # print("{:4.0f}/{:02.0f}/{:02.0f} {:2.0f}:{:2.0f}:{:5.2f}"
                        #       .format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5]))

                        nline = self.nsat if self.flag == 'P' else self.nsat*2
                        for _ in range(nline):
                            line = fh.readline()
                            if line[0] != 'P' and line[0] != 'V':
                                continue

                            svid = line[1:4]
                            sat_ = id2sat(svid)

                            # clk_ev   = line[74] # clock event flag
                            # clk_pred = line[75] # clock pred. flag
                            # mnv_flag = line[78] # maneuver flag
                            # orb_pred = line[79] # orbit pred. flag
                            pred_c = len(line) >= 76 and line[75] == 'P'
                            pred_o = len(line) >= 80 and line[79] == 'P'

                            # x,y,z[km],clock[usec]
                            for j in range(4):
                                if j < 3 and (opt & 1) and pred_o:
                                    continue
                                if j < 3 and (opt & 2) and not pred_o:
                                    continue
                                if j == 3 and (opt & 1) and pred_c:
                                    continue
                                if j == 3 and (opt & 2) and not pred_c:
                                    continue

                                val = float(line[4+j*14:18+j*14])
                                if abs(val-999999.999999) >= 1e-6:
                                    scl = 1e3 if j < 3 else 1e-6
                                    if line[0] == 'P':
                                        v = True
                                        peph.pos[sat_-1, j] = val*scl
                                    elif v:
                                        peph.vel[sat_-1, j] = val*scl*1e-4

                            if len(line) >= 74:
                                for j in range(4):
                                    if j < 3:
                                        slen, scl, ofst = 2, 1e-3, 0
                                    else:
                                        slen, scl, ofst = 3, 1e-12, 1
                                    s = line[61+j*3:61+j*3+slen]
                                    std = int(s) if s[-1] != ' ' else 0
                                    if self.scl[ofst] > 0.0 and std > 0.0:
                                        v = pow(self.scl[ofst], std)*scl
                                        if line[0] == 'P':
                                            peph.std[sat_-1, j] = v
                                        else:
                                            peph.vst[sat_-1, j] = v*1e-4
                    if v:
                        nav.peph.append(peph)

        return nav

    def write_sp3(self, fname, nav, sats=None):
        """
        Write data to SP3 file
        """

        # Update accumulated satellite list
        #
        if sats is not None:
            self.nsat = len(sats)
            self.sat = sorted(list(sats))

        with open(fname, "w") as fh:

            # Write header section
            #

            t0 = nav.peph[0].time
            e = time2epoch(t0)
            ne = len(nav.peph)

            # Epoch lines
            #

            fh.write("#dP{:04d} {:02d} {:02d} {:02d} {:02d} {:011.8f} {:7d} d+D {:16s}\n"
                     .format(e[0], e[1], e[2], e[3], e[4], e[5], ne, ' '))

            week, secs = time2gpst(t0)
            tstep = timediff(nav.peph[1].time, t0)
            mjd = 44244 + 7*week + int(secs/86400.0)
            fod = time2doy(t0) % 1

            fh.write("## {:04d} {:15.8f} {:14.8f} {:5n} {:15.13f}\n"
                     .format(week, secs, tstep, mjd, fod))

            # Satellite list and accuracy indicators
            #
            for i in range(int(np.ceil(self.nsat / 17))):

                nsat = "{:4n}".format(self.nsat) if i == 0 else "    "
                prns = [sat2id(s) for s in self.sat[i*17:i*17+17]]
                fh.write('+ {}   {:51s}\n'.format(nsat, ''.join(prns)))

            for i in range(int(self.nsat / 17+1)):

                accs = ['  0' for s in self.sat[i*17:i*17+17]]
                fh.write('++{}   {:51s}\n'.format('    ', ''.join(accs)))

            fh.write(
                '%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n')
            fh.write(
                '%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n')

            fh.write(
                '%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n')
            fh.write(
                '%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n')
            fh.write(
                '%i    0    0    0    0      0      0      0      0         0\n')
            fh.write(
                '%i    0    0    0    0      0      0      0      0         0\n')

            # Comment section
            #
            fh.write('/* \n')

            # Write data section
            #
            for peph in nav.peph:

                e = time2epoch(peph.time)

                fh.write("*  {:04d} {:02d} {:02d} {:02d} {:02d} {:011.8f}\n"
                         .format(e[0], e[1], e[2], e[3], e[4], e[5]))

                for sat in self.sat:

                    pos = [0, 0, 0] \
                        if np.isnan(peph.pos[sat-1][0:3]).any() \
                        else peph.pos[sat-1][0:3]

                    clk = 0.999999999999 \
                        if np.isnan(peph.pos[sat-1][3]) \
                        else peph.pos[sat-1][3]

                    fh.write("P{:3s} {:13.6f} {:13.6f} {:13.6f} {:13.6f}\n"
                             .format(sat2id(sat),
                                     pos[0]*1e-3,
                                     pos[1]*1e-3,
                                     pos[2]*1e-3,
                                     clk*1e+6))

            # Terminate file
            #
            fh.write('EOF\n')

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
