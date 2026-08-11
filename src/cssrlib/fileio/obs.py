"""RINEX observation files."""

import numpy as np
from pathlib import Path
from cssrlib.domain.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.domain.gnss import epoch2time
from cssrlib.domain.gnss import prn2sat, char2sys
from cssrlib.domain.gnss import Obs
from cssrlib.domain.gnss import id2sat


class ObsFileMixin:
    """Mixed into :class:`~cssrlib.fileio.reader.rnxdec`."""

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

