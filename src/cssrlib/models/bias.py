"""Bias-SINEX code and phase bias decoding."""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.gnss import id2sat, char2sys
from cssrlib.gnss import timediff, gtime_t
from cssrlib.gnss import rSigRnx, uGNSS, uTYP

import numpy as np



class bias_t():
    def __init__(self, sat: int, tst: gtime_t, ted: gtime_t, sig1: rSigRnx,
                 sig2=rSigRnx(), bias=0.0, std=0.0):
        self.sat = sat
        self.tst = tst
        self.ted = ted
        self.sig1 = sig1
        self.sig2 = sig2
        self.bias = bias
        self.std = std


class biasdec():
    def __init__(self):
        self.dcb = []
        self.osb = []

    def doy2time(self, ep):
        """ calculate time from doy """
        year = int(ep[0])
        doy = int(ep[1])
        sec = int(ep[2])
        if year == 0 and doy == 0 and sec == 0:  # undef
            year = 3000
        days = (year-1970)*365+(year-1969)//4+doy-1
        return gtime_t(days*86400+sec)

    def getosb(self, sat, time, sig):
        """ retrieve OSB value based on satellite, epoch and signal code """

        bias = np.nan

        for osb in self.osb:

            if osb.sat == sat and \
                    osb.sig1 == sig and \
                    timediff(time, osb.tst) >= 0.0 and \
                    timediff(time, osb.ted) < 0.0:
                bias = osb.bias
                break

        return bias

    def parse(self, fname, siteID=None):
        with open(fname, "r", encoding='latin-1') as fh:
            status = False
            for line in fh:
                if line[0] == '*':
                    continue
                if '+BIAS/SOLUTION' in line:
                    status = True
                elif '-BIAS/SOLUTION' in line:
                    status = False
                if status and line[0:5] == ' DSB ':

                    # Differential Signal Bias

                    sys = char2sys(line[6])
                    prn = line[11:14]
                    site = line[15:24].strip()

                    # Skip station DSBs
                    #
                    if site and not prn[1:].strip():
                        continue

                    # Skip GLONASS biases if site ID does not match
                    #
                    if sys == 'R' and not site == siteID:
                        continue

                    sig1 = rSigRnx(sys, line[25:28])
                    sig2 = rSigRnx(sys, line[30:33])

                    # year:doy:sec
                    ep1 = [int(line[35:39]), int(
                        line[40:43]), int(line[44:49])]
                    ep2 = [int(line[50:54]), int(
                        line[55:58]), int(line[59:64])]
                    tst = self.doy2time(ep1)
                    ted = self.doy2time(ep2)
                    unit = line[65:69]

                    if sig1.typ != sig2.typ:
                        print("format error: different type of sig1 and sig2")
                        return -1
                    if (sig1.typ == uTYP.C and unit[0:2] != 'ns') or \
                       (sig1.typ == uTYP.L and unit[0:3] != 'cyc'):
                        print("format error: inconsistent dimension")
                        return -1
                    if (sig1 == rSigRnx() or sig2 == rSigRnx()):
                        print("ERROR: invalid signal code!")
                        return -1
                    bias = float(line[70:91])
                    std = float(line[92:103])
                    """
                    if len(line) >= 137:
                        slope = float(line[104:125])
                        std_s = float(line[126:137])
                    """
                    dcb = bias_t(id2sat(prn), tst, ted, sig1, sig2, bias, std)
                    self.dcb.append(dcb)

                elif status and line[0:5] == ' OSB ':

                    # Observable-specific Signal Bias

                    sys = char2sys(line[6])
                    prn = line[11:14]
                    site = line[15:24].strip()

                    # Skip station OSBs
                    #
                    if site and not prn[1:].strip():
                        continue

                    # Skip GLONASS biases if site ID does not match
                    #
                    if sys == uGNSS.GLO and not site == siteID:
                        continue

                    sig1 = rSigRnx(sys, line[25:28])
                    sig2 = rSigRnx()

                    # year:doy:sec
                    ep1 = [int(line[35:39]), int(
                        line[40:43]), int(line[44:49])]
                    ep2 = [int(line[50:54]), int(
                        line[55:58]), int(line[59:64])]
                    tst = self.doy2time(ep1)
                    ted = self.doy2time(ep2)
                    unit = line[65:69]
                    # if (type1 == 0 and unit[0:2] != 'ns') or (type1 == 1 and unit[0:3] != 'cyc'):
                    if (sig1.typ == uTYP.L and unit[0:2] != 'ns'):
                        print("format error: inconsistent dimension {} in {}"
                              .format(sig1.str(), unit))
                        return -1

                    bias = float(line[70:91])
                    std = float(line[92:103])
                    """
                    if len(line) >= 137:
                        slope = float(line[104:125])
                        std_s = float(line[126:137])
                    """
                    osb = bias_t(id2sat(prn), tst, ted, sig1, sig2, bias, std)
                    self.osb.append(osb)
