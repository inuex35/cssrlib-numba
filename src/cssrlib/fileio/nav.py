"""RINEX navigation files.

The header, the RINEX-4 record types (system time offset, Earth
orientation, ionosphere) and the three ephemeris families: GLONASS,
SBAS and the Keplerian constellations."""

import numpy as np
from pathlib import Path
from cssrlib.gnss import uGNSS
from cssrlib.gnss import bdt2gpst, time2bdt
from cssrlib.gnss import gpst2time, bdt2time
from cssrlib.gnss import prn2sat, char2sys, utc2gpst
from cssrlib.gnss import Eph, Geph, Nav, time2gpst
from cssrlib.gnss import Seph, STOParam, EOPParam
from cssrlib.gnss import IONParam


class NavFileMixin:
    """Mixed into :class:`~cssrlib.fileio.reader.rnxdec`."""

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

