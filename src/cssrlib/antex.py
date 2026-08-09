"""Antenna phase-centre offsets and variations (ANTEX)."""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.gnss import id2sat, char2sys, sat2prn
from cssrlib.gnss import timediff, gtime_t
from cssrlib.gnss import str2time
from cssrlib.gnss import ecef2enu
from cssrlib.gnss import rSigRnx, uGNSS, uTYP, uSIG

import numpy as np

from cssrlib.frames import orb2ecef


NMAX = 10
MAXDTE = 900.0
EXTERR_CLK = 1e-3
EXTERR_EPH = 5e-7


class pcv_t():
    def __init__(self):
        self.sat = 0
        self.code = ''
        self.type = ''
        self.ts = gtime_t()
        self.te = gtime_t()
        self.off = {}
        self.var = {}
        self.zen = [0, 0, 0]
        self.nv = 0


class atxdec():
    """ decoder for ANTEX files """

    def __init__(self):
        self.pcvs = []
        self.pcvr = []

    def readpcv(self, fname, onlyReceiver=False):
        """ read ANTEX file """

        state = False
        sys = uGNSS.NONE
        sig = rSigRnx()
        pcv = pcv_t()

        with open(fname, "r") as fh:
            for line in fh:
                if len(line) < 60 or "COMMENT" in line[60:]:
                    continue
                if "START OF ANTENNA" in line[60:]:
                    pcv = pcv_t()
                    state = True
                elif "END OF ANTENNA" in line[60:]:
                    if pcv.sat is None:
                        self.pcvr.append(pcv)
                    elif not onlyReceiver:
                        self.pcvs.append(pcv)
                    state = False
                if not state:
                    continue
                if "TYPE / SERIAL NO" in line[60:]:
                    pcv.type = line[0:20]
                    pcv.code = line[20:40].strip()
                    if not pcv.code:
                        pcv.sat = None
                    else:
                        pcv.sat = id2sat(pcv.code)
                elif "VALID FROM" in line[60:]:
                    pcv.ts = str2time(line, 2, 40)
                elif "VALID UNTIL" in line[60:]:
                    pcv.te = str2time(line, 2, 40)
                elif "START OF FREQUENCY" in line[60:]:
                    sys = char2sys(line[3])
                    sig = rSigRnx(sys, 'L'+line[5])
                elif "END OF FREQUENCY" in line[60:]:
                    sys = uGNSS.NONE
                    sig = rSigRnx()
                elif "NORTH / EAST / UP" in line[60:]:  # unit [mm]
                    neu = [float(x) for x in line[3:30].split()]
                    pcv.off.update({sig: np.zeros(3)})
                    # For satellite use XYZ, for receiver use ENU
                    pcv.off[sig][0] = neu[0] if pcv.sat is not None else neu[1]
                    pcv.off[sig][1] = neu[1] if pcv.sat is not None else neu[0]
                    pcv.off[sig][2] = neu[2]
                elif "ZEN1 / ZEN2 / DZEN" in line[60:]:
                    pcv.zen = [float(x) for x in line[3:20].split()]
                    pcv.nv = int((pcv.zen[1]-pcv.zen[0])/pcv.zen[2])+1
                elif "NOAZI" in line[3:8]:  # unit [mm]
                    var = [float(x) for x in line[8:].split()]
                    if len(var) > pcv.nv:
                        """
                        print("WARNING: fix length of NOAZI for {} {} {}"
                              .format(pcv.type.strip(), pcv.code, sig))
                        """
                        var = var[0:pcv.nv]
                    pcv.var.update({sig: np.array(var)})

    def readngspcv(self, fname, pcvs=None):
        """ read NGS antenna parameter file """

        state = False
        sys = uGNSS.NONE
        n = 0

        sig1 = rSigRnx('GL1')
        sig2 = rSigRnx('GL2')

        with open(fname, "r") as fh:
            for line in fh:
                if len(line) >= 62 and line[61] == '|':
                    continue
                if line[0] != ' ':
                    n = 0
                n += 1
                if n == 1:
                    pcv = pcv_t()
                    pcv.sat = None
                    pcv.code = 0
                    pcv.type = line[:20]
                    pcv.zen = [0.0, 90.0, 5.0]
                    pcv.nv = int((pcv.zen[1]-pcv.zen[0])/pcv.zen[2])+1
                elif n == 2:
                    neu = np.array([float(x) for x in line[3:30].split()])
                    pcv.off[sig1] = neu[[1, 0, 2]]
                elif n == 3:
                    var = [float(x) for x in line.split()]
                elif n == 4:
                    var += [float(x) for x in line.split()]
                    pcv.var.update({sig1: np.array(var)})
                elif n == 5:
                    neu = np.array([float(x) for x in line[3:30].split()])
                    pcv.off[sig2] = neu[[1, 0, 2]]
                elif n == 6:
                    var = [float(x) for x in line.split()]
                elif n == 7:
                    var += [float(x) for x in line.split()]
                    pcv.var.update({sig2: np.array(var)})
                    self.pcvr.append(pcv)


def searchpcv(pcvs, name, time):
    """ get satellite or receiver antenna pcv """

    if isinstance(name, str):

        for pcv in pcvs:
            if pcv.type != name:
                continue
            if pcv.ts.time != 0 and timediff(pcv.ts, time) > 0.0:
                continue
            if pcv.te.time != 0 and timediff(pcv.te, time) < 0.0:
                continue
            return pcv

    else:

        for pcv in pcvs:
            if pcv.sat != name:
                continue
            if pcv.ts.time != 0 and timediff(pcv.ts, time) > 0.0:
                continue
            if pcv.te.time != 0 and timediff(pcv.te, time) < 0.0:
                continue
            return pcv

    return None


def substSigTx(pcv, sig):
    """
    Substitute frequency band for PCO/PCV selection of transmitting antenna.

    This function converts a RINEX observation code to a phase observation code
    without tracking attribute. If the signal is not available in the list of
    PCOs, a substitution based on system and frequency band is done.

    Parameters
    ----------
    pcv : pcv_t
        Receiver antenna PCV element
    sig : rRnxSig
        RINEX signal code

    Returns
    -------
    sig : rRnxSig
        Substituted RINEX signal code
    """

    # Convert to phase observation without tracking attribute
    #
    sig = sig.toTyp(uTYP.L).toAtt()

    # Use directly if an corresponding offset exists
    #
    if sig in pcv.off:
        return sig

    # Substitute if signal does not exist
    #
    if sig.sys == uGNSS.GPS:
        if sig.sig == uSIG.L5:
            sig = rSigRnx(sig.sys, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.GLO:
        if sig.sig == uSIG.L3:
            sig = rSigRnx(sig.sys, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.BDS:
        if sig.sig == uSIG.L8:  # BDS-3 B2a+b
            sig = rSigRnx(sig.sys, sig.typ, uSIG.L5)

    return sig


def substSigRx(pcv, sig):
    """
    Substitute frequency band for PCO/PCV selection of receiving antenna.

    This function converts a RINEX observation code to a phase observation code
    without tracking attribute. If the signal is not available in the list of
    PCOs, a substitution based on system and frequency band is done.

    Parameters
    ----------
    pcv : pcv_t
        Receiver antenna PCV element
    sig : rRnxSig
        RINEX signal code

    Returns
    -------
    sig : rRnxSig
        Substituted RINEX signal code
    """

    # Convert to phase observation without tracking attribute
    #
    sig = sig.toTyp(uTYP.L).toAtt()

    # Use directly if an corresponding offset exists
    #
    if sig in pcv.off:
        return sig

    # Substitute if signal does not exist
    #
    if sig.sys == uGNSS.GPS:
        if sig.sig == uSIG.L5:
            sig = rSigRnx(sig.sys, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.GLO:
        if sig.sig == uSIG.L1:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L1)
        elif sig.sig == uSIG.L2 or sig.sig == uSIG.L3:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.GAL:
        if sig.sig == uSIG.L1:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L1)
        elif sig.sig == uSIG.L5 or sig.sig == uSIG.L6 or \
                sig.sig == uSIG.L7 or sig.sig == uSIG.L8:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.BDS:
        if sig.sig == uSIG.L1 or sig.sig == uSIG.L2:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L1)
        elif sig.sig == uSIG.L5 or sig.sig == uSIG.L6 or \
                sig.sig == uSIG.L7 or sig.sig == uSIG.L8:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.QZS:
        if sig.sig == uSIG.L1:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L1)
        elif sig.sig == uSIG.L2 or sig.sig == uSIG.L5 or sig.sig == uSIG.L6:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.IRN:
        if sig.sig == uSIG.L5:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)
    elif sig.sys == uGNSS.SBS:
        if sig.sig == uSIG.L1:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L1)
        elif sig.sig == uSIG.L5:
            sig = rSigRnx(uGNSS.GPS, sig.typ, uSIG.L2)

    return sig


def antModelTx(nav, e, sigs, sat, time, rs, sig0=None):
    """
    Range correction for transmitting antenna

    This function computes the range correction for the transmitting antenna
    from the PCO correction projected on line-of-sight vector as well as the
    interpolated phase variation correction depending on the zenith angle.

    Parameters
    ----------
    nav : Nav()
        contains the PCO/PCV corrections for rover and base antenna
    pos : np.array
        receiver position in ECEF
    e : np.array
        line-of-sight vector in ECEF from receiver to satellite
    sigs : list of rRnxSig
        RINEX signal codes
    sat : int
        satellite number
    time : gtime_t
        epoch
    rs : np.array() of float
        satellite position in ECEF
    sig0: list of rRnxSig
        RINEX signal codes for APC reference (empty list for CoM)

    Returns
    -------
    dant : np.array of float values
        range correction for each specified signal
    """

    sys, _ = sat2prn(sat)
    # Select satellite antenna
    #
    ant = searchpcv(nav.sat_ant, sat, time)
    if ant is None:
        return None

    # Rotation matrix from satellite antenna frame to ECEF frame [ex, ey, ez]
    #
    A = orb2ecef(time, rs)
    ez = A[2, :]

    # Zenith angle and zenith angle grid
    #
    za = np.rad2deg(np.arccos(np.dot(ez, -e)))
    za_t = np.arange(ant.zen[0], ant.zen[1]+ant.zen[2], ant.zen[2])

    # CoM offset of reference signals
    #
    off0 = np.zeros(3)
    if sig0 is not None:
        if sys == uGNSS.GLO:
            freq = [s.frequency(nav.glo_ch[sat]) for s in sig0]
        else:
            freq = [s.frequency() for s in sig0]
        fac0 = [1.0 for s in sig0]

        if len(freq) == 2:
            fac0 = (+freq[0]**2/(freq[0]**2-freq[1]**2),
                    -freq[1]**2/(freq[0]**2-freq[1]**2),)

        for fac0_, sig0_ in zip(fac0, sig0):

            # Substitute signal if not available
            #
            sig = substSigTx(ant, sig0_)

            # Satellite PCO in local antenna frame
            #
            off0 += fac0_*ant.off[sig]

    # Interpolate PCV and map PCO on line-of-sight vector
    #
    dant = np.zeros(len(sigs))
    for i, sig_ in enumerate(sigs):

        # Substitute signal if not available
        #
        sig = substSigTx(ant, sig_)

        # Satellite PCO in local antenna frame
        #
        off = ant.off[sig] - off0

        # Convert satellite PCO from antenna frame into ECEF frame
        #
        pco_v = off@A

        # Interpolate PCV and map PCO on line-of-sight vector
        #
        if sig not in ant.off or sig not in ant.var:
            dant[i] = None
        else:
            pcv = np.interp(za, za_t, ant.var[sig])
            pco = -np.dot(pco_v, -e)
            dant[i] = (pco+pcv)*1e-3

    return dant


def antModelRx(nav, pos, e, sigs, rtype=1):
    """
    Range correction for receiving antenna

    This function computes the range correction for the receiving antenna
    from the PCO correction projected on line-of-sight vector as well as the
    interpolated phase variation correction depending on the zenith angle.

    Parameters
    ----------
    nav : Nav()
        contains the PCO/PCV corrections for rover and base antenna
    pos : np.array
        Receiver position in ECEF
    e : np.array of float
        Line-of-sight vector in ECEF from receiver to satellite
    sigs : list of rRnxSig
        RINEX signal codes
    rtype : int
        flag 1 for rover, else for base PCO/PCV

    Returns
    -------
    dant : np.array of float
        Range correction for each specified signal
    """

    # Convert LOS vector to local antenna frame
    #
    e = ecef2enu(pos, e)

    # Select rover or base antenna
    #
    if rtype == 1:  # for rover
        ant = nav.rcv_ant
    else:  # for base
        ant = nav.rcv_ant_b

    # Elevation angle, zenith angle and zenith angle grid
    #
    za = np.rad2deg(np.arccos(e[2]))
    za_t = np.arange(ant.zen[0], ant.zen[1]+ant.zen[2], ant.zen[2])

    # Loop over signals
    #
    dant = np.zeros(len(sigs))
    for i, sig_ in enumerate(sigs):

        # Substitute signal if not available
        #
        sig = substSigRx(ant, sig_)

        # Interpolate PCV and map PCO on line-of-sight vector
        #
        if sig not in ant.off or sig not in ant.var:
            dant[i] = None
        else:
            pcv = np.interp(za, za_t, ant.var[sig])
            pco = -np.dot(ant.off[sig], e)
            dant[i] = (pco+pcv)*1e-3

    return dant


def apc2com(nav, sat, time, rs, sigs, k=None):
    """
    Satellite position vector correction in ECEF from APC to CoM
    using ANTEX PCO corrections
    """

    # Select satellite antenna
    #
    ant = searchpcv(nav.sat_ant, sat, time)
    if ant is None:
        return None

    # Rotation matrix from satellite antenna frame to ECEF frame [ex, ey, ez]
    #
    A = orb2ecef(time, rs)

    freq = [s.frequency(k) for s in sigs]
    if len(sigs) == 1:
        facs = (1.0,)
    elif len(sigs) == 2:
        f12 = (freq[0]**2-freq[1]**2)
        facs = (+freq[0]**2/f12, -freq[1]**2/f12)
    else:
        return None

    # Interpolate PCV and map PCO on line-of-sight vector
    #
    dr = np.zeros(3)
    for fac_, sig_ in zip(facs, sigs):

        # Substitute signal if not available
        #
        sig = substSigTx(ant, sig_)

        # Satellite PCO in local antenna frame
        #
        off = fac_*ant.off[sig]*1e-3  # [m]

        # Convert satellite PCO from antenna frame into ECEF frame
        #
        dr -= off@A

    return dr
