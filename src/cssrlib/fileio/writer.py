"""RINEX encoding: observation and navigation file output."""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP
from cssrlib.gnss import time2bdt
from cssrlib.gnss import timediff, gtime_t
from cssrlib.gnss import timeget, utc2gpst, time2epoch
from cssrlib.gnss import sat2id, sat2prn, gpst2bdt, time2gpst
from cssrlib.gnss import gpst2utc


class rnxenc:
    """ class for RINEX encoder """

    def __init__(self, sig_tab=None):
        self.ver = -1.0
        self.fobs = None
        self.fnav = None
        self.rnx_obs_header_sent = False
        self.sig_tab = sig_tab

        self.prog = "cssrlib"
        self.runby = "Unknown"
        self.agency = "Unknown"
        self.observer = "Unknown"
        self.rec = "Unknown"
        self.rectype = "Unkown"
        self.recver = ""
        self.ant = "Unknown"
        self.anttype = "Unknown"
        self.pos = np.zeros(3)
        self.dant = np.zeros(3)
        self.glo_bias = None

        self.rec_eph = {}

    def rnx_nav_header(self, fh=None, ver=4.02):
        """ write RINEX navigation header to file """
        tutc = timeget()
        tgps = utc2gpst(tutc)
        leaps = timediff(tgps, tutc)

        ep = time2epoch(tutc)
        s = "{:4d}{:02d}{:02d} {:02d}{:02d}{:02d} {:3s}". \
            format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], "UTC")

        fh.write("{:9.2f}           {:19s} {:19s} {:20s}\n".
                 format(ver, "NAVIGATION DATA", "M", "RINEX VERSION / TYPE"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".
                 format(self.prog, self.runby, s, "PGM / RUN BY / DATE"))
        fh.write("{:6d}{:6s}{:6s}{:6s}{:3s}{:33s}{:20s}\n".
                 format(leaps, "", "", "", "", "", "LEAP SECONDS"))
        fh.write("{:60s}{:20s}\n".
                 format("", "END OF HEADER"))

    def rnx_obs_header(self, ts: gtime_t, fh=None, ver=4.02):
        """ write RINEX observation header to file """

        if self.rnx_obs_header_sent:
            return
        self.rnx_obs_header_sent = True

        sys_t = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E',
                 uGNSS.QZS: 'J', uGNSS.BDS: 'C', uGNSS.IRN: 'I',
                 uGNSS.SBS: 'S'}

        tutc = timeget()
        tgps = utc2gpst(tutc)
        leaps = timediff(tgps, tutc)

        ep = time2epoch(tutc)
        s = "{:4d}{:02d}{:02d} {:02d}{:02d}{:02d} {:3s}". \
            format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], "UTC")

        fh.write("{:9.2f}           {:19s} {:19s} {:20s}\n".
                 format(ver, "OBSERVATION DATA", "M", "RINEX VERSION / TYPE"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".
                 format(self.prog, self.runby, s, "PGM / RUN BY / DATE"))

        fh.write("{:60s}{:20s}\n".format("Unknown", "MARKER NAME"))
        fh.write("{:20s}{:40s}{:20s}\n".format(
            self.observer, self.agency, "OBSERVER / AGENCY"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".format(
            self.rec, self.rectype, self.recver[:20], "REC # / TYPE / VERS"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".format(
            self.ant, self.anttype, "", "ANT # / TYPE"))
        fh.write("{:14.4f}{:14.4f}{:14.4f}{:18s}{:20s}\n".format(
            self.pos[0], self.pos[1], self.pos[2], "", "APPROX POSITION XYZ"))
        fh.write("{:14.4f}{:14.4f}{:14.4f}{:18s}{:20s}\n".format(
            self.dant[0], self.dant[1], self.dant[2], "",
            "ANTENNA: DELTA H/E/N"))

        for sys in self.sig_tab:
            pr = self.sig_tab[sys][uTYP.C]
            cp = self.sig_tab[sys][uTYP.L]
            nsig = len(pr)+len(cp)

            if uTYP.D in self.sig_tab[sys]:
                dp = self.sig_tab[sys][uTYP.D]
                nsig += len(dp)

            if uTYP.S in self.sig_tab[sys]:
                cn = self.sig_tab[sys][uTYP.S]
                nsig += len(cn)

            if nsig == 0:
                continue

            fh.write("{:1s}  {:3d}".format(sys_t[sys], nsig))

            n = 0
            for k, _ in enumerate(pr):
                fh.write(" {:3s}".format(pr[k].str()))
                n += 1
                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))

                fh.write(" {:3s}".format(cp[k].str()))
                n += 1
                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))

                if uTYP.D in self.sig_tab[sys]:
                    fh.write(" {:3s}".format(dp[k].str()))
                    n += 1
                    if n == 13:
                        fh.write("  {:20s}\n{:6s}".format(
                            "SYS / # / OBS TYPES", ""))

                if uTYP.S in self.sig_tab[sys]:
                    fh.write(" {:3s}".format(cn[k].str()))
                    n += 1

                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))
                elif n >= nsig-1:
                    fh.write("  {:s}".format("    "*(13-(nsig % 13))))
                    fh.write("{:20s}".format("SYS / # / OBS TYPES \n"))

        # TBD
        ep = time2epoch(ts)
        fh.write(" {:5d} {:5d} {:5d} {:5d} {:5d}{:13.7f}".
                 format(int(ep[0]), int(ep[1]), int(ep[2]), int(ep[3]),
                        int(ep[4]), ep[5]))
        fh.write("{:5s}{:3s}{:9s}{:20s}\n".
                 format("", "GPS", "", "TIME OF FIRST OBS"))

        fh.write("{:6d}{:6s}{:6s}{:6s}{:3s}{:33s}{:20s}\n".
                 format(leaps, "", "", "", "", "", "LEAP SECONDS"))
        fh.write("{:60s}{:20s}\n".
                 format("", "END OF HEADER"))
        fh.flush()

    def sval(self, v: float):
        if v == 0.0:
            s = "{:14s}".format("")
        else:
            s = "{:14.3f}".format(v)
        return s

    def rnx_obs_body(self, obs=None, fh=None):
        """ write RINEX observation message to file """

        ep = time2epoch(obs.time)
        nsat = len(obs.sat)
        nsig = obs.P.shape[1]
        fh.write("> {:4d} {:02d} {:02d} {:02d} {:02d} {:010.7f}".
                 format(int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), ep[5]))
        fh.write("  {:1d}{:3d}\n".format(0, nsat))

        for k in range(nsat):
            sys, prn = sat2prn(obs.sat[k])
            fh.write("{:3s}".format(sat2id(obs.sat[k])))
            for i in range(nsig):
                ssi = min(max(int(obs.S[k][i]/6), 1), 9)
                lli = obs.lli[k][i]
                fh.write("{:14s}{:2s}".format(
                    self.sval(obs.P[k][i]), ""))

                fh.write("{:14s}".format(self.sval(obs.L[k][i])))
                if obs.L[k][i] == 0.0:
                    fh.write("{:2s}".format(""))
                else:
                    fh.write("{:1d}{:1d}".format(lli, ssi))

                if uTYP.D in self.sig_tab[sys]:
                    fh.write("{:14s}{:2s}".format(self.sval(obs.D[k][i]), ""))

                if uTYP.S in self.sig_tab[sys]:
                    fh.write("{:14s}{:2s}".format(self.sval(obs.S[k][i]), ""))

            fh.write("\n")

    def rnx_nav_body(self, eph=None, fh=None):
        """ write RINEX navigation message to file """
        if eph.sat in self.rec_eph.keys():
            if eph.mode in self.rec_eph[eph.sat].keys() and \
                    self.rec_eph[eph.sat][eph.mode][0] == eph.iode:
                return
        else:
            self.rec_eph[eph.sat] = {}
        self.rec_eph[eph.sat][eph.mode] = [eph.iode, eph.toes]

        id_ = sat2id(eph.sat)
        sys, prn = sat2prn(eph.sat)

        if sys == uGNSS.BDS:
            ep = time2epoch(gpst2bdt(eph.toc))
            week, tot_ = time2bdt(eph.tot)
        else:
            ep = time2epoch(eph.toc)
            week, tot_ = time2gpst(eph.tot)

        if sys == uGNSS.BDS:
            if eph.mode == 0:  # D1/D2
                lbl = "D1" if (prn > 5 and prn < 59) else "D2"
                v1 = float(eph.iode)
                v2 = eph.toes
            else:
                if eph.mode == 1:  # B-CNAV1
                    lbl = "CNV1"
                elif eph.mode == 2:  # B-CNAV2
                    lbl = "CNV2"
                elif eph.mode == 3:  # B-CNAV3
                    lbl = "CNV3"
                else:
                    return
                v1 = eph.Adot
                v2 = eph.toes

        elif (sys == uGNSS.GPS or sys == uGNSS.QZS):
            if eph.mode == 0:  # LNAV
                lbl = "LNAV"
                v1 = float(eph.iode)
                v2 = eph.toes
            else:
                lbl = "CNAV" if eph.mode == 1 else "CNV2"
                v1 = float(eph.Adot)
                v2 = eph.tops
        elif sys == uGNSS.GAL:
            lbl = "INAV" if eph.mode == 0 else "FNAV"
            v1 = float(eph.iode)
            v2 = eph.toes
        elif sys == uGNSS.IRN:
            if eph.mode == 0:
                lbl = "LNAV"
                v1 = float(eph.iode)
                v2 = eph.toes
            else:
                lbl = "L1NV"
                v1 = eph.Adot
                v2 = eph.iode
        else:
            return

        fh.write("> {:2s} {:3s} {:2s}\n".format("EPH", id_, lbl))
        fh.write("{:3s} {:4d} {:02d} {:02d} {:02d} {:02d} {:02d}".
                 format(id_, int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), int(ep[5])))
        fh.write("{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.af0, eph.af1, eph.af2))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(v1, eph.crs, eph.deln, eph.M0))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.cuc, eph.e, eph.cus, np.sqrt(eph.A)))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(v2, eph.cic, eph.OMG0, eph.cis))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.i0, eph.crc, eph.omg, eph.OMGd))

        if sys == uGNSS.BDS:
            if eph.mode == 0:  # D1/D2
                fh.write("    {:19.12E}{:19s}{:19.12E}{:19s}\n".
                         format(eph.idot, "", eph.week, ""))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(float(eph.sva), float(eph.svh),
                                eph.tgd, eph.tgd_b))
                fh.write("    {:19.12E}{:19.12E}{:19s}{:19s}\n".
                         format(tot_, float(eph.iodc), "", ""))

            else:  # B-CNAV1,2,3
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(eph.idot, eph.delnd, eph.sattype, eph.tops))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(eph.sisai[0], eph.sisai[1], eph.sisai[2],
                                eph.sisai[3]))
                if eph.mode == 1:
                    fh.write("    {:19.12E}{:19s}{:19.12E}{:19.12E}\n".
                             format(eph.isc[0], "", eph.tgd, eph.tgd_b))
                elif eph.mode == 2:
                    fh.write("    {:19s}{:19.12E}{:19.12E}{:19.12E}\n".
                             format("", eph.isc[1], eph.tgd, eph.tgd_b))

                if eph.mode <= 2:
                    fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                             format(float(eph.sismai), float(eph.svh),
                                    float(eph.integ), float(eph.iodc)))
                    fh.write("    {:19.12E}{:19s}{:19s}{:19.12E}\n".
                             format(tot_, "", "", float(eph.iode)))
                else:  # B-CNAV3
                    fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                             format(float(eph.sismai), float(eph.svh),
                                    float(eph.integ), eph.tgd))
                    fh.write("    {:19.12E}{:19s}{:19s}{:19s}\n".
                             format(tot_, "", "", ""))

        if (sys == uGNSS.GPS or sys == uGNSS.QZS):
            if eph.mode == 0:  # LNAV
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(eph.idot, float(eph.code), float(eph.week),
                                float(eph.l2p)))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(float(eph.sva), float(eph.svh), eph.tgd,
                                float(eph.iodc)))
                fh.write("    {:19.12E}{:19.12E}{:19s}{:19s}\n".
                         format(tot_, float(eph.fit), "", ""))
            else:  # CNAV/CNAV2
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(eph.idot, float(eph.delnd), float(eph.urai[0]),
                                float(eph.urai[1])))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(float(eph.urai[3]), float(eph.svh), eph.tgd,
                                float(eph.urai[2])))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                         format(float(eph.isc[0]), float(eph.isc[1]),
                                float(eph.isc[2]), float(eph.isc[3])))
                if eph.mode == 2:  # CNAV2
                    fh.write("    {:19.12E}{:19.12E}{:19s}{:19s}\n".
                             format(float(eph.isc[4]), float(eph.isc[5]),
                                    "", ""))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                         format(tot_, float(eph.wn_op), float(eph.integ), ""))

        if sys == uGNSS.GAL:  # I/NAV, F/NAV
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                     format(eph.idot, float(eph.code), float(eph.week), ""))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(eph.sva), float(eph.svh), eph.tgd,
                            float(eph.tgd_b)))
            fh.write("    {:19.12E}{:19s}{:19s}{:19s}\n".
                     format(tot_, "", "", ""))

        if sys == uGNSS.IRN:
            if eph.mode == 0:  # LNAV
                fh.write("    {:19.12E}{:19s}{:19.12E}{:19s}\n".
                         format(eph.idot, "", float(eph.week),
                                ""))
                fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                         format(float(eph.sva), float(eph.svh), eph.tgd,
                                ""))
                fh.write("    {:19.12E}{:19s}{:19s}{:19s}\n".
                         format(tot_, "", "", ""))
            elif eph.mode == 2:  # L1NV
                rsf = eph.integ
                fh.write("    {:19.12E}{:19.12E}{:19s}{:19.12E}\n".
                         format(eph.idot, eph.delnd, "", rsf))
                if rsf == 0:
                    fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                             format(float(eph.urai), float(eph.svh), eph.tgd,
                                    ""))
                    fh.write("    {:19.12E}{:19.12E}{:19s}{:19s}\n".
                             format(float(eph.isc[5]), float(eph.isc[4]), "",
                                    ""))
                else:  # rsf = 1
                    fh.write("    {:19.12E}{:19.12E}{:19s}{:19.12E}\n".
                             format(float(eph.urai), float(eph.svh), "",
                                    eph.tgd))
                    fh.write("    {:19s}{:19s}{:19.12E}{:19.12E}\n".
                             format("", "", float(eph.isc[5]),
                                    float(eph.isc[4])))

                fh.write("    {:19.12E}{:19s}{:19s}{:19s}\n".
                         format(tot_, "", "", ""))

    def rnx_gnav_body(self, geph=None, fh=None):
        """ write RINEX navigation message for GLONASS to file """
        if geph.sat in self.rec_eph.keys():
            if geph.mode in self.rec_eph[geph.sat].keys() and \
                    self.rec_eph[geph.sat][geph.mode][0] == geph.iode:
                return
        else:
            self.rec_eph[geph.sat] = {}
        self.rec_eph[geph.sat][geph.mode] = [geph.iode, geph.toes]

        id_ = sat2id(geph.sat)
        sys, prn = sat2prn(geph.sat)

        if sys != uGNSS.GLO:
            return

        ep = time2epoch(gpst2utc(geph.toe))
        week, tot_ = time2gpst(geph.tof)

        if geph.mode == 0:
            lbl = "FDMA"
            v1 = tot_
            v2 = float(geph.frq)
            v3 = float(geph.age)
        elif geph.mode == 1:
            lbl = "L1OC"
            v1 = geph.beta
            v2 = float(geph.status)
            v3 = geph.isc[1]  # tgd_L2OCp
        else:
            lbl = "L3OC"
            v1 = geph.beta
            v2 = float(geph.status)
            v3 = geph.isc[2]  # ISC_L3OC

        fh.write("> {:2s} {:3s} {:2s}\n".format("EPH", id_, lbl))
        fh.write("{:3s} {:4d} {:02d} {:02d} {:02d} {:02d} {:02d}".
                 format(id_, int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), int(ep[5])))
        fh.write("{:19.12E}{:19.12E}{:19.12E}\n".
                 format(-geph.taun, geph.gamn, v1))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(geph.pos[0]*1e-3, geph.vel[0]*1e-3,
                        geph.acc[0]*1e-3, float(geph.svh)))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(geph.pos[1]*1e-3, geph.vel[1]*1e-3,
                        geph.acc[1]*1e-3, v2))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(geph.pos[2]*1e-3, geph.vel[2]*1e-3,
                        geph.acc[2]*1e-3, v3))

        if geph.mode == 0:  # FDMA
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                     format(float(geph.flag), float(geph.dtaun),
                            float(geph.sva), ""))
        else:  # L1OC, L3OC
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(geph.sattype), float(geph.src),
                            geph.aode, geph.aodc))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(geph.flag), geph.tin,
                            geph.tau1, geph.tau2))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(geph.psi, float(geph.sn),
                            geph.win, geph.dw))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(geph.wmax, float(geph.dpos[0]),
                            geph.dpos[1], geph.dpos[2]))
            fh.write("    {:19.12E}{:19.12E}{:19s}{:19.12E}\n".
                     format(float(geph.urai[0]), float(geph.urai[1]), "",
                            tot_))

    def rnx_snav_body(self, seph=None, fh=None):
        """ write RINEX navigation message for SBAS to file """
        if seph.sat in self.rec_eph.keys():
            if seph.mode in self.rec_eph[seph.sat].keys() and \
                    self.rec_eph[seph.sat][seph.mode][0] == seph.iodn:
                return
        else:
            self.rec_eph[seph.sat] = {}
        self.rec_eph[seph.sat][seph.mode] = [seph.iodn]

        id_ = sat2id(seph.sat)
        sys, prn = sat2prn(seph.sat)

        if sys != uGNSS.SBS:
            return

        ep = time2epoch(seph.t0)
        week, tot_ = time2gpst(seph.tof)

        fh.write("> {:2s} {:3s} {:2s}\n".format("EPH", id_, "SBAS"))
        fh.write("{:3s} {:4d} {:02d} {:02d} {:02d} {:02d} {:02d}".
                 format(id_, int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), int(ep[5])))
        fh.write("{:19.12E}{:19.12E}{:19.12E}\n".
                 format(seph.af0, seph.af1, tot_))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(seph.pos[0]*1e-3, seph.vel[0]*1e-3,
                        seph.acc[0]*1e-3, float(seph.svh)))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(seph.pos[1]*1e-3, seph.vel[1]*1e-3,
                        seph.acc[1]*1e-3, seph.sva))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(seph.pos[2]*1e-3, seph.vel[2]*1e-3,
                        seph.acc[2]*1e-3, float(seph.iodn)))
