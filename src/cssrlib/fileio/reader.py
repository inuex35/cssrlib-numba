"""RINEX decoding.

rnxdec is the composition point; the per-format decoding lives beside it,
one module per thing being read:

    cssrlib.fileio.nav   navigation files -- header, RINEX-4 records,
                         GLONASS / SBAS / Keplerian ephemerides
    cssrlib.fileio.obs   observation files
    cssrlib.fileio.clk   clock files
    cssrlib.fileio.sync  pairing two receivers, choosing signals

What stays here is what all of them share: the signal table, the numeric
and time field parsers, and the file handles.
"""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP
from cssrlib.gnss import epoch2time, timediff, gtime_t
from cssrlib.gnss import timeadd

from cssrlib.fileio.nav import NavFileMixin
from cssrlib.fileio.obs import ObsFileMixin
from cssrlib.fileio.clk import ClockFileMixin, pclk_t  # noqa: F401
from cssrlib.fileio.sync import (sync_obs, sync_obs_hold,  # noqa: F401
                                 auto_detect_signals)      # noqa: F401


class rnxdec(NavFileMixin, ObsFileMixin, ClockFileMixin):
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
        if not u.strip():
            # a short line slices to "" past its end; "".isspace() is
            # False, so this used to fall through into float("")
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


    # TODO: decode GLONASS FCN lines


# Band priority — pick lowest-numbered bands first when more than max_freq
# are common (L1 preferred, then L2, then L5/L7/L6/L8/...).
