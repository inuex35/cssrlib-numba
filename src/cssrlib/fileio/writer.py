"""RINEX encoding: observation and navigation file output."""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP
from cssrlib.gnss import timediff, gtime_t
from cssrlib.gnss import timeget, utc2gpst, time2epoch
from cssrlib.gnss import sat2id, sat2prn


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
