"""RINEX decoding: observation, navigation and clock files.

Split out of rinex.py, which held the decoder and the encoder in one
1,685-line module although neither uses the other. cssrlib.rinex
re-exports everything, so existing imports keep working.
"""

import numpy as np
from pathlib import Path
from cssrlib.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.gnss import bdt2gpst, time2bdt
from cssrlib.gnss import gpst2time, bdt2time, epoch2time, timediff, gtime_t
from cssrlib.gnss import prn2sat, char2sys, utc2gpst
from cssrlib.gnss import Eph, Geph, Obs, Nav, time2gpst
from cssrlib.gnss import timeadd, id2sat, Seph, STOParam, EOPParam
from cssrlib.gnss import IONParam


class pclk_t:
    """ class for precise clock data """

    def __init__(self, time=None):
        if time is not None:
            self.time = time
        else:
            self.time = gtime_t()
        self.clk = np.zeros(uGNSS.MAXSAT)
        self.std = np.zeros(uGNSS.MAXSAT)


class rnxdec:
    """ class for RINEX decoder """

    def __init__(self):

        self.ver = -1.0
        self.fobs = None

        # signal code mapping from RINEX header to columns in data section
        self.sig_map = {}
        # signal selection for internal data structure
        self.sig_tab = {}
        self.sig_index = {}
        self.nsig = {uTYP.C: 0, uTYP.L: 0, uTYP.D: 0, uTYP.S: 0}

        self.pos = np.array([0, 0, 0])
        self.ecc = np.array([0, 0, 0])
        self.rcv = None
        self.ant = None
        self.ts = None
        self.te = None
        # 0:LNAV,INAV,D1/D2, 1:CNAV/CNAV1/FNAV, 2: CNAV2, 3: CNAV3,
        # 4:FDMA, 5:SBAS
        self.mode_nav = 0
        self.glo_ch = {}

        self.ofst_src = {'GP': uGNSS.GPS, 'GL': uGNSS.GLO,
                         'GA': uGNSS.GAL, 'BD': uGNSS.BDS,
                         'QZ': uGNSS.QZS, 'IR': uGNSS.IRN,
                         'SB': uGNSS.SBS, 'UT': uGNSS.NONE}
        self.itype_t = {'LNAV': 0, 'FDMA': 1, 'IFNV': 2, 'D1D2': 3,
                        'SBAS': 4, 'CNVX': 5, 'L1NV': 6, 'LXOC': 7}

    def setSignals(self, sigList):
        """ define the signal list for each constellation """

        for sig in sigList:
            if sig.sys not in self.sig_tab:
                self.sig_tab.update({sig.sys: {}})
            if sig.typ not in self.sig_tab[sig.sys]:
                self.sig_tab[sig.sys].update({sig.typ: []})
            if sig not in self.sig_tab[sig.sys][sig.typ]:
                self.sig_tab[sig.sys][sig.typ].append(sig)
            else:
                raise ValueError("duplicate signal {} specified!".format(sig))

        for _, sigs in self.sig_tab.items():
            for typ, sig in sigs.items():
                self.nsig[typ] = max((self.nsig[typ], len(sig)))

        self._rebuild_signal_index()

    def _rebuild_signal_index(self):
        self.sig_index = {}
        for sys, sigs_by_type in self.sig_tab.items():
            sys_idx = {}
            for typ, sigs in sigs_by_type.items():
                sys_idx[typ] = {sig.str(): idx for idx, sig in enumerate(sigs)}
            self.sig_index[sys] = sys_idx

    def getSignals(self, sys, typ):
        """ retrieve signal list for constellation and obs type """
        if sys in self.sig_tab.keys() and typ in self.sig_tab[sys].keys():
            return self.sig_tab[sys][typ]
        else:
            return []

    def autoSignals(self, decb=None, max_freq=2, **kwargs):
        """Detect signals from the decoded header and apply them.

        Convenience wrapper around :func:`auto_detect_signals`: builds the
        signal list from this decoder's ``sig_map`` (call ``decode_obsh``
        first) and runs ``setSignals``. When a base decoder ``decb`` is
        given, both decoders are configured with matching signals and the
        ``(sigs, sigsb)`` lists are returned.

        Returns the rover signal list (and base list when ``decb`` is given).
        """
        sig_map_base = decb.sig_map if decb is not None else None
        sigs, sigsb = auto_detect_signals(
            self.sig_map, sig_map_base, max_freq=max_freq, **kwargs)
        self.setSignals(sigs)
        if decb is not None:
            decb.setSignals(sigsb)
            return sigs, sigsb
        return sigs

    def autoSubstituteSignals(self):
        """
        Automatically substitute signal tracking attribute based on
        available signals
        """
        for sys, tmp in self.sig_tab.items():
            for typ, sigs in tmp.items():
                for i, sig in enumerate(sigs):

                    # Skip unavailable systems or available signals
                    #
                    if sys not in self.sig_map.keys():
                        continue
                    if sig in self.sig_map[sys].values():
                        continue

                    # Not found try to replace
                    #
                    if sys == uGNSS.GPS and sig.str()[1] in '12':
                        atts = 'CW' if sig.str()[2] in 'CW' else 'SLX'
                    elif sys == uGNSS.GPS and sig.str()[1] in '5':
                        atts = 'IQX'
                    elif sys == uGNSS.GAL and sig.str()[1] in '578':
                        atts = 'IQX'
                    elif sys == uGNSS.GAL and sig.str()[1] in '16':
                        atts = 'BCX'
                    elif sys == uGNSS.QZS and sig.str()[1] in '126':
                        atts = 'SLX'
                    elif sys == uGNSS.QZS and sig.str()[1] in '5':
                        atts = 'IQX'
                    elif sys == uGNSS.BDS and sig.str()[1] in '157':
                        atts = 'PX'
                    else:
                        atts = []

                    for a in atts:
                        if sig.toAtt(a) in self.sig_map[sys].values():
                            self.sig_tab[sys][typ][i] = sig.toAtt(a)
        self._rebuild_signal_index()

    def flt(self, u, c=-1):
        """ convert string to float """
        if c >= 0:
            u = u[19*c+4:19*(c+1)+4]
        if u.isspace():
            return 0.0
        return float(u.replace("D", "E"))

    def adjday(self, t: gtime_t, t0: gtime_t):
        """ adjust time to within 1 day of t0 """
        tt = timediff(t, t0)
        if tt < -43200.0:
            return timeadd(t, 86400.0)
        if tt > 43200.0:
            return timeadd(t, -86400.0)
        return t

    def decode_time(self, s, ofst=0, slen=2):
        """ decode time from string """
        year = int(s[ofst+0:ofst+4])
        month = int(s[ofst+5:ofst+7])
        day = int(s[ofst+8:ofst+10])
        hour = int(s[ofst+11:ofst+13])
        minute = int(s[ofst+14:ofst+16])
        sec = float(s[ofst+17:ofst+slen+17])
        t = epoch2time([year, month, day, hour, minute, sec])
        return t

    def decode_nav(self, navfile: str, nav: Nav, append: bool = False) -> Nav:
        """Wrapper for decode_nav with Path support"""

        navfile: Path = Path(navfile)
        if navfile.suffix.lower() in ['.gz', '.z']:
            import gzip
            with gzip.open(navfile, 'rt') as fnav:
                return self._decode_nav(fnav, nav, append)
        else:
            with open(navfile, 'rt') as fnav:
                return self._decode_nav(fnav, nav, append)

    def _decode_nav(self, fnav, nav, append=False):
        """
        Decode RINEX Navigation message from file

        NOTE: system time epochs are converted into GPST on reading!

        """

        if not append:
            nav.eph = []
            nav.geph = []
            nav.seph = []

        for line in fnav:
            if line[60:73] == 'END OF HEADER':
                break
            elif line[60:80] == 'RINEX VERSION / TYPE':
                self.ver = float(line[4:10])
                if self.ver < 3.02:
                    return -1
            elif line[60:76] == 'IONOSPHERIC CORR':
                if line[0:4] == 'GPSA' or line[0:4] == 'QZSA':
                    for k in range(4):
                        nav.ion[0, k] = self.flt(line[5+k*12:5+(k+1)*12])
                if line[0:4] == 'GPSB' or line[0:4] == 'QZSB':
                    for k in range(4):
                        nav.ion[1, k] = self.flt(line[5+k*12:5+(k+1)*12])
            elif line[60:72] == 'LEAP SECONDS':
                nav.leaps = int(line[:6])

        for line in fnav:

            if self.ver >= 4.0:

                if line[0:5] == '> STO':  # system time offset (TBD)

                    sys = char2sys(line[6])
                    itype = line[10:14]

                    if sys not in nav.sto_prm:
                        nav.sto_prm[sys] = {}

                    if itype not in self.itype_t:
                        fnav.readline()
                        fnav.readline()
                        continue

                    im = self.itype_t[itype]

                    if im not in nav.sto_prm[sys]:
                        nav.sto_prm[sys][im] = STOParam()

                    line = fnav.readline()
                    nav.sto_prm[sys][im].t_ot = self.decode_time(line, 4)
                    mode = line[24:28]
                    if mode[0:2] in self.ofst_src and \
                            mode[2:4] in self.ofst_src:
                        nav.sto_prm[sys][im].prm[0] = \
                            self.ofst_src[mode[0:2]]
                        nav.sto_prm[sys][im].prm[1] = \
                            self.ofst_src[mode[2:4]]

                    line = fnav.readline()
                    nav.sto_prm[sys][im].t_t = self.flt(line, 0)
                    for k in range(3):
                        nav.sto_prm[sys][im].a[k] = self.flt(line, k+1)
                    continue

                elif line[0:5] == '> EOP':  # earth orientation param
                    sys = char2sys(line[6])
                    itype = line[10:14]

                    if sys not in nav.eop_prm:
                        nav.eop_prm[sys] = {}

                    if itype not in self.itype_t:
                        fnav.readline()
                        fnav.readline()
                        fnav.readline()
                        continue

                    im = self.itype_t[itype]

                    if im not in nav.eop_prm[sys]:
                        nav.eop_prm[sys][im] = EOPParam()

                    line = fnav.readline()
                    nav.eop_prm[sys][im].t_eop = self.decode_time(line, 4)
                    for k in range(3):
                        nav.eop_prm[sys][im].prm[k] = self.flt(line, k+1)
                    line = fnav.readline()
                    for k in range(3):
                        nav.eop_prm[sys][im].prm[k+3] = self.flt(line, k+1)
                    line = fnav.readline()
                    nav.eop_prm[sys][im].t_t = self.flt(line, 0)
                    for k in range(3):
                        nav.eop_prm[sys][im].prm[k+6] = self.flt(line, k+1)
                    continue

                elif line[0:5] == '> ION':  # iono (TBD)
                    sys = char2sys(line[6])
                    itype = line[10:14]
                    stype = '' if len(line) < 20 else line[15:19]

                    if sys not in nav.ion_prm:
                        nav.ion_prm[sys] = {}

                    im = self.itype_t[itype]
                    nav.ion_prm[sys][im] = IONParam()

                    line = fnav.readline()
                    nav.ion_prm[sys][im].t_tm = self.decode_time(line, 4)
                    if sys == uGNSS.GAL and itype == 'IFNV':  # Nequick-G
                        for k in range(3):  # ai0, ai1, ai2
                            nav.ion_prm[sys][im].prm[k] = \
                                self.flt(line, k+1)
                        line = fnav.readline()
                        # disturbance flags
                        nav.ion_prm[sys][im].prm[3] = \
                            int(self.flt(line, 0))
                    elif sys == uGNSS.BDS and itype == 'CNVX':  # BDGIM
                        for k in range(3):
                            nav.ion_prm[sys][im].prm[k] = \
                                self.flt(line, k+1)
                        line = fnav.readline()
                        for k in range(4):
                            nav.ion_prm[sys][im].prm[k+3] = \
                                self.flt(line, k)
                        line = fnav.readline()
                        for k in range(2):
                            nav.ion_prm[sys][im].prm[k+7] = \
                                self.flt(line, k)
                    elif sys == uGNSS.IRN and itype == 'L1NV':  # L1NAV
                        if stype == 'KLOB':  #
                            iodk = self.flt(line, 1)
                            line = fnav.readline()
                            for k in range(4):
                                nav.ion_prm[sys][im].prm[k] = \
                                    self.flt(line, k)
                            line = fnav.readline()
                            for k in range(4):
                                nav.ion_prm[sys][im].prm[k+4] = \
                                    self.flt(line, k)
                            nav.ion_prm[sys][im].iod = iodk
                            line = fnav.readline()
                            nav.ion_prm[sys][im].region = np.zeros(4)
                            for k in range(4):
                                nav.ion_prm[sys][im].region[k] = \
                                    self.flt(line, k)
                        elif stype == 'NEQN':
                            nav.ion_prm[sys][im].iod = self.flt(line, 1)
                            prm = np.zeros((3, 8))
                            for j in range(3):
                                line = fnav.readline()
                                for k in range(4):  # a0, a1, a2, idf
                                    prm[j, k] = self.flt(line, k)
                                line = fnav.readline()
                                # lon_min, lon_max, mopid_min, mopid_max
                                for k in range(4):
                                    prm[j, k+4] = self.flt(line, k)
                            nav.ion_prm[sys][im].prm = prm

                    elif sys == uGNSS.GLO and itype == 'LXOC':
                        c_A = self.flt(line, 1)
                        c_F10_7 = self.flt(line, 2)
                        c_Ap = self.flt(line, 3)
                        nav.ion_prm[sys][im].prm[0:3] = \
                            [c_A, c_F10_7, c_Ap]

                    else:  # Klobuchar (LNAV, D1D2, CNVX)
                        nav.ion_prm[sys][im].prm = np.zeros(9)

                        for k in range(3):
                            nav.ion_prm[sys][im].prm[k] = \
                                self.flt(line, k+1)
                        line = fnav.readline()
                        for k in range(4):
                            nav.ion_prm[sys][im].prm[k+3] = \
                                self.flt(line, k)
                        line = fnav.readline()
                        nav.ion_prm[sys][im].prm[7] = self.flt(line, 0)
                        if len(line) >= 42:
                            nav.ion_prm[sys][im].prm[8] = \
                                int(self.flt(line, 1))
                    continue

                elif line[0:5] == '> EPH':
                    sys = char2sys(line[6])
                    self.mode_nav = 0  # LNAV, D1/D2, INAV
                    m = line[10:14]
                    if m == 'CNAV' or m == 'CNV1' or m == 'FNAV':
                        self.mode_nav = 1
                    elif m == 'CNV2' or m == 'L1NV':
                        self.mode_nav = 2
                    elif m == 'CNV3':
                        self.mode_nav = 3
                    elif m == 'FDMA':
                        self.mode_nav = 0
                    elif m == 'L1OC':
                        self.mode_nav = 1
                    elif m == 'L3OC':
                        self.mode_nav = 3
                    elif m == 'SBAS':
                        self.mode_nav = 0
                    line = fnav.readline()

            elif self.ver >= 3.0:  # RINEX 3.0.x
                self.mode_nav = 0

            # Process ephemeris information
            #
            sys = char2sys(line[0])

            # Skip undesired constellations
            #
            if sys == uGNSS.GLO:
                prn = int(line[1:3])
                sat = prn2sat(sys, prn)
                geph = Geph(sat)

                pos = np.zeros(3)
                vel = np.zeros(3)
                acc = np.zeros(3)

                geph.mode = self.mode_nav
                toc = self.decode_time(line, 4)
                week, tocs = time2gpst(toc)
                toc = gpst2time(week,
                                np.floor((tocs+450.0)/900.0)*900.0)
                dow = int(tocs//86400.0)

                geph.taun = -self.flt(line, 1)
                geph.gamn = self.flt(line, 2)
                if self.mode_nav == 0:  # FDMA
                    t0 = self.flt(line, 3)
                else:  # L1OC, L3OC
                    bet_ = self.flt(line, 3)  # clock drift rate

                line = fnav.readline()  # line #1
                pos[0] = self.flt(line, 0)*1e3
                vel[0] = self.flt(line, 1)*1e3
                acc[0] = self.flt(line, 2)*1e3
                geph.svh = int(self.flt(line, 3))

                line = fnav.readline()  # line #2
                pos[1] = self.flt(line, 0)*1e3
                vel[1] = self.flt(line, 1)*1e3
                acc[1] = self.flt(line, 2)*1e3

                if self.mode_nav == 0:  # FDMA
                    geph.frq = int(self.flt(line, 3))

                    if geph.frq > 128:
                        geph.frq -= 256
                else:  # L1OC
                    dvalid = int(self.flt(line, 3))

                line = fnav.readline()  # line #3
                pos[2] = self.flt(line, 0)*1e3
                vel[2] = self.flt(line, 1)*1e3
                acc[2] = self.flt(line, 2)*1e3

                geph.pos = pos
                geph.vel = vel
                geph.acc = acc

                if self.mode_nav == 0:  # FDMA
                    geph.age = int(self.flt(line, 3))
                elif self.mode_nav == 1:  # L1OC
                    tgd_L2OCp = self.flt(line, 3)  # tgd_L2OCp
                elif self.mode_nav == 3:  # L3OC
                    isc_L3OCp = self.flt(line, 3)  # isc_L3OCp

                # Use GLONASS line #4 only from RINEX v3.05 onwards
                #
                if self.ver >= 3.05:

                    line = fnav.readline()  # line #4

                    if self.mode_nav == 0:  # FDMA
                        # b7-8: M, b6: P4, b5: P3, b4: P2, b2-3: P1, b0-1: P
                        geph.status = int(self.flt(line, 0))
                        geph.dtaun = -self.flt(line, 1)
                        geph.urai[0] = int(self.flt(line, 2))
                        if len(line) >= 80:
                            geph.svh = int(self.flt(line, 3))
                    else:  # L1OC,L3OC
                        sattype = int(self.flt(line, 0))
                        src = int(self.flt(line, 1))
                        geph.aode = int(self.flt(line, 2))
                        geph.aodc = int(self.flt(line, 3))

                        line = fnav.readline()  # line #5
                        P2 = int(self.flt(line, 0))  # attitude flag
                        geph.tin = self.flt(line, 1)  # sec of day, UTC(SU)
                        geph.tau1 = self.flt(line, 2)
                        geph.tau2 = self.flt(line, 3)

                        line = fnav.readline()  # line #6
                        geph.yaw = self.flt(line, 0)
                        geph.sn = int(self.flt(line, 1))
                        geph.win = self.flt(line, 2)
                        geph.dw = self.flt(line, 3)

                        line = fnav.readline()  # line #7
                        geph.wmax = self.flt(line, 0)
                        geph.dpoc[0] = self.flt(line, 1)
                        geph.dpoc[1] = self.flt(line, 2)
                        geph.dpoc[2] = self.flt(line, 3)

                        line = fnav.readline()  # line #8
                        geph.urai[0] = int(self.flt(line, 0))
                        geph.urai[1] = int(self.flt(line, 1))
                        tot = self.flt(line, 2)

                tod = t0 % 86400.0
                tof = gpst2time(week, tod + dow*86400.0)
                tof = self.adjday(tof, toc)

                geph.toe = utc2gpst(toc)
                geph.tof = utc2gpst(tof)

                # iode = Tb(7bit)
                geph.iode = int(((tocs+10800.0) % 86400)/900.0+0.5)

                nav.geph.append(geph)
                continue

            elif sys == uGNSS.SBS:
                prn = int(line[1:3])+100
                sat = prn2sat(sys, prn)
                seph = Seph(sat)

                pos = np.zeros(3)
                vel = np.zeros(3)
                acc = np.zeros(3)

                seph.toc = self.decode_time(line, 4)
                seph.af0 = self.flt(line, 1)
                seph.af1 = self.flt(line, 2)
                seph.tot = self.flt(line, 3)

                line = fnav.readline()  # line #1
                pos[0] = self.flt(line, 0)*1e3
                vel[0] = self.flt(line, 1)*1e3
                acc[0] = self.flt(line, 2)*1e3
                seph.svh = int(self.flt(line, 3))

                line = fnav.readline()  # line #2
                pos[1] = self.flt(line, 0)*1e3
                vel[1] = self.flt(line, 1)*1e3
                acc[1] = self.flt(line, 2)*1e3
                seph.sva = self.flt(line, 3)

                line = fnav.readline()  # line #3
                pos[2] = self.flt(line, 0)*1e3
                vel[2] = self.flt(line, 1)*1e3
                acc[2] = self.flt(line, 2)*1e3
                seph.iodn = int(self.flt(line, 3))

                seph.pos = pos
                seph.vel = vel
                seph.acc = acc

                nav.seph.append(seph)
                continue

            elif sys not in (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS, uGNSS.BDS,
                             uGNSS.IRN):
                continue

            prn = int(line[1:3])
            if sys == uGNSS.QZS:
                prn += 192
            sat = prn2sat(sys, prn)
            eph = Eph(sat)

            eph.urai = np.zeros(4, dtype=int)
            eph.sisai = np.zeros(4, dtype=int)
            eph.isc = np.zeros(6)

            eph.mode = self.mode_nav

            eph.toc = self.decode_time(line, 4)
            eph.af0 = self.flt(line, 1)
            eph.af1 = self.flt(line, 2)
            eph.af2 = self.flt(line, 3)

            line = fnav.readline()  # line #1

            if sys == uGNSS.GAL or \
                    (sys == uGNSS.IRN and self.mode_nav == 0):
                eph.iode = int(self.flt(line, 0))
                eph.iodc = eph.iode
            else:
                if self.mode_nav > 0:
                    eph.Adot = self.flt(line, 0)
                else:
                    eph.iode = int(self.flt(line, 0))

            eph.crs = self.flt(line, 1)
            eph.deln = self.flt(line, 2)
            eph.M0 = self.flt(line, 3)

            line = fnav.readline()  # line #2
            eph.cuc = self.flt(line, 0)
            eph.e = self.flt(line, 1)
            eph.cus = self.flt(line, 2)
            sqrtA = self.flt(line, 3)
            eph.A = sqrtA**2

            line = fnav.readline()  # line #3
            if sys == uGNSS.IRN and self.mode_nav == 2:
                eph.iode = int(self.flt(line, 0))
                eph.iode = eph.iodc
            else:
                if (sys == uGNSS.GPS or sys == uGNSS.QZS) and \
                        self.mode_nav > 0:  # CNAV, CNAV/2
                    eph.tops = self.flt(line, 0)
                    eph.week, eph.toes = time2gpst(eph.toc)
                else:
                    eph.toes = self.flt(line, 0)
            eph.cic = self.flt(line, 1)
            eph.OMG0 = self.flt(line, 2)
            eph.cis = self.flt(line, 3)

            line = fnav.readline()  # line #4
            eph.i0 = self.flt(line, 0)
            eph.crc = self.flt(line, 1)
            eph.omg = self.flt(line, 2)
            eph.OMGd = self.flt(line, 3)

            line = fnav.readline()  # line #5
            eph.idot = self.flt(line, 0)

            if sys == uGNSS.GAL or self.mode_nav == 0:
                eph.code = int(self.flt(line, 1))  # source for GAL
                eph.week = int(self.flt(line, 2))

                if sys == uGNSS.GAL and self.ver < 4.0:
                    eph.mode = 1 if eph.code & 0x2 else 0

            elif sys == uGNSS.IRN and self.mode_nav == 0:
                eph.week = int(self.flt(line, 2))

            else:
                eph.delnd = self.flt(line, 1)
                if sys == uGNSS.BDS:
                    eph.sattype = int(self.flt(line, 2))
                    eph.tops = int(self.flt(line, 3))
                elif sys == uGNSS.IRN and self.mode_nav == 2:
                    eph.integ = int(self.flt(line, 3))  # rsf
                else:  # CNAV, CNAV/2
                    eph.urai = [0, 0, 0, 0]
                    eph.urai[0] = int(self.flt(line, 2))
                    eph.urai[1] = int(self.flt(line, 3))

            line = fnav.readline()  # line #6
            if sys == uGNSS.BDS and self.mode_nav > 0:
                eph.sisai[0] = int(self.flt(line, 0))  # oe
                eph.sisai[1] = int(self.flt(line, 1))  # ocb
                eph.sisai[2] = int(self.flt(line, 2))  # oc1
                eph.sisai[3] = int(self.flt(line, 3))  # oc2
            elif sys == uGNSS.IRN:
                if self.mode_nav == 0:
                    eph.sva = self.flt(line, 0)
                else:  # L1NV
                    eph.urai = self.flt(line, 0)
                eph.svh = int(self.flt(line, 1))
                if self.mode_nav == 2 and eph.integ == 1:
                    eph.tgd = int(self.flt(line, 3))
                else:
                    eph.tgd = int(self.flt(line, 2))
            else:
                eph.sva = int(self.flt(line, 0))
                eph.svh = int(self.flt(line, 1))
                eph.tgd = float(self.flt(line, 2))
                if sys == uGNSS.GPS or sys == uGNSS.QZS:
                    if self.mode_nav == 0:
                        eph.iodc = int(self.flt(line, 3))
                    else:
                        eph.urai[2] = int(self.flt(line, 3))  # URAI_NED2
                        eph.urai[3] = eph.sva  # URAI_ED
                elif sys == uGNSS.GAL:
                    tgd_b = float(self.flt(line, 3))
                    if (eph.code >> 9) & 1:  # E5b,E1
                        eph.tgd_b = eph.tgd
                        eph.tgd = tgd_b
                    else:  # E5a,E1
                        eph.tgd_b = tgd_b
                elif sys == uGNSS.BDS:
                    eph.tgd_b = float(self.flt(line, 3))  # tgd2 B2/B3

                if sys == uGNSS.QZS:
                    eph.code = eph.svh & 0x11  # L1C/A:0x01 or L1C/B:0x10
                    eph.svh = eph.svh & 0xEE   # mask L1C/A, L1C/B health

            if self.mode_nav < 3:
                line = fnav.readline()  # line #7
                if sys == uGNSS.BDS:
                    if self.mode_nav == 0:  # D1/D2
                        tot = self.flt(line, 0)
                        eph.iodc = int(self.flt(line, 1))
                    else:  # CNAV-1,2,3
                        if self.mode_nav == 1:  # CNAV-1
                            eph.isc[0] = float(self.flt(line, 0))  # B1Cd
                        elif self.mode_nav == 2:  # CNAV-2
                            eph.isc[1] = float(self.flt(line, 1))  # B2ad

                        eph.tgd = float(self.flt(line, 2))    # tgd_B1Cp
                        eph.tgd_b = float(self.flt(line, 3))  # tgd_B2ap

                elif sys == uGNSS.IRN:
                    if self.mode_nav > 0:
                        if eph.integ == 0:  # rsf
                            eph.isc[5] = float(self.flt(line, 0))  # S
                            eph.isc[4] = float(self.flt(line, 1))  # L1D
                        else:
                            eph.isc[5] = float(self.flt(line, 2))  # L1P
                            eph.isc[4] = float(self.flt(line, 3))  # L1D

                        line = fnav.readline()  # line #8

                    tot = self.flt(line, 0)

                elif sys == uGNSS.GAL:
                    tot = self.flt(line, 0)

                elif sys in (uGNSS.GPS, uGNSS.QZS):
                    if self.mode_nav > 0:  # CNAV, CNAV/2
                        eph.isc[0] = self.flt(line, 0)  # ISC_L1CA
                        eph.isc[1] = self.flt(line, 1)  # ISC_L2C
                        eph.isc[2] = self.flt(line, 2)  # ISC_L5I5
                        eph.isc[3] = self.flt(line, 3)  # ISC_L5Q5
                    else:  # LNAV
                        tot = self.flt(line, 0)
                        if len(line) >= 42:
                            eph.fit = int(self.flt(line, 1))

            if sys in (uGNSS.GPS, uGNSS.QZS):
                if self.mode_nav > 0:  # CNAV, CNAV/2
                    line = fnav.readline()  # line #8
                    if self.mode_nav == 2:  # CNAV/2
                        eph.isc[4] = self.flt(line, 0)  # ISC_L1Cd
                        eph.isc[5] = self.flt(line, 1)  # ISC_L1Cp

                        line = fnav.readline()  # line #9

                    tot = int(self.flt(line, 0))
                    eph.wn_op = int(self.flt(line, 1))
                    if len(line) >= 61:  # optional
                        eph.integ = int(self.flt(line, 2))

            elif sys == uGNSS.BDS and self.mode_nav > 0:  # CNAV-1,2,3
                line = fnav.readline()  # line #8
                eph.sismai = int(self.flt(line, 0))
                eph.svh = int(self.flt(line, 1))
                eph.integ = int(self.flt(line, 2))
                if self.mode_nav < 3:  # CNAV-1,2
                    eph.iodc = int(self.flt(line, 3))
                else:  # CNAV-3
                    eph.tgd_b = float(self.flt(line, 3))  # tgd_B2bI

                line = fnav.readline()  # line #9
                tot = self.flt(line, 0)
                if self.mode_nav < 3:  # CNAV-1,2
                    eph.iode = int(self.flt(line, 3))

            if sys == uGNSS.BDS:
                if self.mode_nav > 0:
                    eph.week, _ = time2bdt(eph.toc)
                eph.toc = bdt2gpst(eph.toc)
                eph.toe = bdt2gpst(bdt2time(eph.week, eph.toes))
                eph.tot = bdt2gpst(bdt2time(eph.week, tot))
            else:
                eph.toe = gpst2time(eph.week, eph.toes)
                eph.tot = gpst2time(eph.week, tot)

            nav.eph.append(eph)

        return nav

    def decode_clk(self, clkfile, nav):
        """decode Clock-RINEX data from file """

        # Offset for Clock-RINEX v3.x data section
        #
        offs = None

        nav.pclk = []
        fnav = open(clkfile, 'rt')

        # Read header section
        #
        for line in fnav:

            if 'RINEX VERSION / TYPE' in line:
                ver = float(line[0:20])
                offs = 0 if ver < 3.04 else 5

            if 'END OF HEADER' in line:
                break

        # Read data section
        #
        for line in fnav:

            if line[0:2] != 'AS':
                continue

            sys = char2sys(line[3])
            prn = int(line[4:7])
            if sys == uGNSS.QZS:
                prn += 192
            sat = prn2sat(sys, prn)

            t = self.decode_time(line, offs+8, 9)

            if nav.nc <= 0 or abs(timediff(nav.pclk[-1].time, t)) > 1e-9:
                nav.nc += 1
                pclk = pclk_t()
                pclk.time = t
                nav.pclk.append(pclk)

            nrec = int(line[offs+35:offs+37])
            clk = float(line[offs+40:offs+59])
            std = float(line[offs+61:offs+80]) if nrec >= 2 else 0.0

            nav.pclk[nav.nc-1].clk[sat-1] = clk
            nav.pclk[nav.nc-1].std[sat-1] = std

        return nav

    def decode_obsh(self, obsfile: str) -> int:
        """Wrapper of decode RINEX Observation header from file"""

        obsfile: Path = Path(obsfile)
        if obsfile.suffix.lower() in ['.gz', '.z']:
            import gzip
            self.fobs = gzip.open(
                obsfile, 'rt', encoding='utf-8', errors='ignore')
        else:
            self.fobs = open(obsfile, 'rt')
        return self._decode_obsh()

    # TODO: decode GLONASS FCN lines
    def _decode_obsh(self):
        """ decode RINEX Observation header from file """
        for line in self.fobs:
            if line[60:73] == 'END OF HEADER':
                break
            if line[60:80] == 'RINEX VERSION / TYPE':
                self.ver = float(line[4:10])
                if self.ver < 3.02:
                    return -1
            elif 'REC # / TYPE / VERS' in line:
                self.rcv = line[20:40].upper()
            elif 'ANT # / TYPE' in line:
                self.ant = line[20:40].upper()
            elif line[60:79] == 'APPROX POSITION XYZ':
                self.pos = np.array([float(line[0:14]),
                                     float(line[14:28]),
                                     float(line[28:42])])
            elif 'ANTENNA: DELTA H/E/N' in line[60:]:
                self.ecc = np.array([float(line[14:28]),  # East
                                     float(line[28:42]),  # North
                                     float(line[0:14])])  # Up
            elif line[60:79] == 'SYS / # / OBS TYPES':

                gns = char2sys(line[0])
                nsig = int(line[3:6])

                # Extract string list of signal codes
                #
                sigs = line[7:60].split()
                while len(sigs) < nsig:
                    line2 = self.fobs.readline()
                    sigs += line2[7:60].split()

                # Convert to RINEX signal code and store in map
                #
                for i, sig in enumerate(sigs):
                    rnxSig = rSigRnx(gns, sig)
                    if gns not in self.sig_map:
                        self.sig_map.update({gns: {}})
                    self.sig_map[gns].update({i: rnxSig})
            elif 'TIME OF FIRST OBS' in line[60:]:
                self.ts = epoch2time([float(v) for v in line[0:44].split()])
            elif 'TIME OF LAST OBS' in line[60:]:
                self.te = epoch2time([float(v) for v in line[0:44].split()])
            elif 'GLONASS SLOT / FRQ #' in line[60:]:
                nsat = int(line[0:3])
                for i in range(nsat):
                    if i > 0 and i % 8 == 0:
                        line = self.fobs.readline()
                    j = i % 8
                    sat = id2sat(line[4+7*j:7+7*j])
                    ch = int(line[8+7*j: 10+7*j])
                    self.glo_ch[sat] = ch

        return 0

    def decode_obs(self):
        """ decode RINEX Observation message from file """

        obs = Obs()

        for line in self.fobs:

            if line[0] != '>':
                continue

            nsat = int(line[32:35])

            year = int(line[2:6])
            month = int(line[7:9])
            day = int(line[10:12])
            hour = int(line[13:15])
            minute = int(line[16:18])
            sec = float(line[19:29])
            obs.t = epoch2time([year, month, day, hour, minute, sec])

            # Initialize data structures
            #
            pr_rows = []
            cp_rows = []
            dp_rows = []
            cn_rows = []
            lli_rows = []
            sats = []
            obs.sig = self.sig_tab

            for _ in range(nsat):

                line = self.fobs.readline()
                sys = char2sys(line[0])

                # Skip constellation not contained in RINEX header
                #
                if sys not in self.sig_map.keys():
                    continue

                # Skip undesired constellations
                #
                if sys not in self.sig_tab:
                    continue

                sig_index = self.sig_index.get(sys, {})

                # Convert to satellite ID
                #
                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                elif sys == uGNSS.SBS:
                    prn += 100
                sat = prn2sat(sys, prn)

                pr = np.zeros(self.nsig[uTYP.C], dtype=np.float64)
                cp = np.zeros(self.nsig[uTYP.L], dtype=np.float64)
                ll = np.zeros(self.nsig[uTYP.L], dtype=np.int32)
                dp = np.zeros(self.nsig[uTYP.D], dtype=np.float64)
                cn = np.zeros(self.nsig[uTYP.S], dtype=np.float64)

                for i, sig in self.sig_map[sys].items():

                    # Skip undesired signals
                    #
                    if sig.typ not in self.sig_tab[sys] or \
                            sig not in self.sig_tab[sys][sig.typ]:
                        continue

                    # Get string representation of measurement value
                    #
                    sval = line[16*i+3:16*i+17].strip()
                    slli = line[16*i+17] if len(line) > 16*i+17 else ''

                    # Convert from string to numerical value
                    #
                    val = 0.0 if not sval else float(sval)
                    lli = 1 if slli == '1' else 0

                    # Signal index in data structure
                    #
                    j = sig_index[sig.typ][sig.str()]

                    if sig.typ == uTYP.C:
                        pr[j] = val
                    elif sig.typ == uTYP.L:
                        cp[j] = val
                        ll[j] = lli
                    elif sig.typ == uTYP.D:
                        dp[j] = val
                    elif sig.typ == uTYP.S:
                        cn[j] = val
                    else:
                        continue

                # Store prn and data
                #
                pr_rows.append(pr)
                cp_rows.append(cp)
                dp_rows.append(dp)
                cn_rows.append(cn)
                lli_rows.append(ll)
                sats.append(sat)

            nobs = len(sats)
            obs.P = np.asarray(pr_rows, dtype=np.float64).reshape(
                nobs, self.nsig[uTYP.C]
            )
            obs.L = np.asarray(cp_rows, dtype=np.float64).reshape(
                nobs, self.nsig[uTYP.L]
            )
            obs.D = np.asarray(dp_rows, dtype=np.float64).reshape(
                nobs, self.nsig[uTYP.D]
            )
            obs.S = np.asarray(cn_rows, dtype=np.float64).reshape(
                nobs, self.nsig[uTYP.S]
            )
            obs.lli = np.asarray(lli_rows, dtype=np.int32).reshape(
                nobs, self.nsig[uTYP.L]
            )
            obs.sat = np.asarray(sats, dtype=np.int32)

            break

        return obs


def sync_obs(dec, decb, dt_th=0.1):
    """ sync observation between rover and base """
    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    while True:
        dt = timediff(obs.t, obsb.t)
        if np.abs(dt) <= dt_th:
            break
        if dt > dt_th:
            obsb = decb.decode_obs()
        elif dt < dt_th:
            obs = dec.decode_obs()
    return obs, obsb


def _obs_is_eof(obs):
    """ EOF check for rnxdec.decode_obs (returns default Obs() at EOF) """
    return obs.t.time == 0 and obs.t.sec == 0.0


def sync_obs_hold(dec, decb, maxage=30.0):
    """
    Rover-driven sync generator with base-station hold (RTKLIB maxtdiff-style).

    Yields `(obs_rover, obs_base, dt)` for every rover epoch:
      - obs_base: nearest base observation with |t_rover - t_base| <= maxage,
        reused across rover epochs until a newer base arrives. ``None`` when
        no base is within ``maxage`` (e.g. base stream ended or not yet
        started). ``dt`` is set even when base is out of range so the caller
        can log it.
      - dt: ``timediff(t_rover, t_base)`` (NaN when no base decoded yet).

    Works for arbitrary rate combinations, e.g. 5 Hz rover + 1 Hz base: each
    1 Hz base record is reused for ~5 rover epochs until the next base record
    is closer.

    Parameters
    ----------
    dec, decb : rnxdec
        Rover / base decoders positioned after ``decode_obsh``.
    maxage : float
        Maximum |t_rover - t_base| (seconds) for which the base obs is still
        considered usable. Mirrors RTKLIB ``prcopt.maxtdiff`` (default 30 s).
    """
    obsb_curr = decb.decode_obs()
    if _obs_is_eof(obsb_curr):
        obsb_curr = None
        obsb_next = None
    else:
        obsb_next = decb.decode_obs()
        if _obs_is_eof(obsb_next):
            obsb_next = None

    while True:
        obs = dec.decode_obs()
        if _obs_is_eof(obs):
            return

        # Advance base while the next base record is strictly closer to the
        # current rover epoch than the held one (nearest-neighbor hold).
        while obsb_next is not None:
            if obsb_curr is None:
                obsb_curr = obsb_next
                nxt = decb.decode_obs()
                obsb_next = None if _obs_is_eof(nxt) else nxt
                continue
            dt_curr = abs(timediff(obs.t, obsb_curr.t))
            dt_next = abs(timediff(obs.t, obsb_next.t))
            if dt_next < dt_curr:
                obsb_curr = obsb_next
                nxt = decb.decode_obs()
                obsb_next = None if _obs_is_eof(nxt) else nxt
            else:
                break

        if obsb_curr is None:
            yield obs, None, float('nan')
            continue

        dt = timediff(obs.t, obsb_curr.t)
        if abs(dt) <= maxage:
            yield obs, obsb_curr, dt
        else:
            yield obs, None, dt


# Band priority — pick lowest-numbered bands first when more than max_freq
# are common (L1 preferred, then L2, then L5/L7/L6/L8/...).
_BAND_PRIORITY = (1, 2, 5, 7, 6, 8, 3, 4, 9)


def _group_by_band(sigs, typ):
    """Group the rSigRnx values of one sig_map system by frequency band.

    Returns ``{band: rSigRnx}`` for the requested observation type, keeping
    the first signal seen per band (sig_map preserves RINEX header order).
    """
    out = {}
    for s in sigs.values():
        if s.typ != typ:
            continue
        band = int(s.sig) // 100
        out.setdefault(band, s)
    return out


def auto_detect_signals(sig_map_rov, sig_map_base=None, max_freq=2,
                        required=(uTYP.C, uTYP.L, uTYP.S),
                        systems=None, strict_freq=False):
    """Build signal list(s) directly from RINEX header signal maps.

    Mimics RTKLIB's "use whatever the file declares" behaviour, so the caller
    need not hand-craft per-system signal lists. With a base ``sig_map`` it
    returns matching rover/base lists covering the same (sys, typ, band).

    Typical usage::

        dec = rnxdec();  dec.decode_obsh(rover_obs)
        decb = rnxdec(); decb.decode_obsh(base_obs)
        sigs, sigsb = auto_detect_signals(dec.sig_map, decb.sig_map, max_freq=2)
        dec.setSignals(sigs); decb.setSignals(sigsb)

    Parameters
    ----------
    sig_map_rov : dict
        ``rnxdec.sig_map`` of the rover, populated by ``decode_obsh``.
    sig_map_base : dict, optional
        ``rnxdec.sig_map`` of the base. When omitted, only the rover list is
        built and the second return value is an empty list.
    max_freq : int
        Number of frequency bands to keep per system (RTKLIB ``nf``).
    required : tuple of uTYP
        Observation types each band must provide (default C+L+S =
        pseudorange, carrier phase, SNR).
    systems : iterable of uGNSS, optional
        Constellations to consider; default = all common to both receivers.
    strict_freq : bool, default False
        Drop systems that cannot supply ``max_freq`` common bands. No longer
        required for safety (qcedit tolerates short-band systems), so it
        defaults to False to keep single-frequency systems usable.

    Returns
    -------
    (sigs, sigsb) : tuple of list of rSigRnx
        Ready to pass to ``rnxdec.setSignals``. ``sigsb`` is empty when no
        base sig_map is given.
    """
    rov_systems = set(sig_map_rov.keys())
    have_base = sig_map_base is not None
    base_systems = set(sig_map_base.keys()) if have_base else rov_systems
    if systems is None:
        systems = rov_systems & base_systems
    else:
        systems = set(systems) & rov_systems & base_systems

    sigs, sigsb = [], []
    for sys in systems:
        rov_by_typ = {t: _group_by_band(sig_map_rov[sys], t) for t in required}
        if have_base:
            base_by_typ = {t: _group_by_band(sig_map_base[sys], t)
                           for t in required}

        # Bands fully covered (every required type) on both sides.
        common_bands = set(rov_by_typ[required[0]].keys())
        for t in required[1:]:
            common_bands &= rov_by_typ[t].keys()
        if have_base:
            for t in required:
                common_bands &= base_by_typ[t].keys()
        if not common_bands:
            continue
        if strict_freq and len(common_bands) < max_freq:
            continue

        # Canonical band order (L1, then L2, then L5 ...).
        ordered = [b for b in _BAND_PRIORITY if b in common_bands]
        ordered += sorted(b for b in common_bands if b not in ordered)
        for band in ordered[:max_freq]:
            for t in required:
                sigs.append(rov_by_typ[t][band])
                if have_base:
                    sigsb.append(base_by_typ[t][band])
    return sigs, sigsb
