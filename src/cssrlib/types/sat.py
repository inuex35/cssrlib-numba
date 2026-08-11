"""Satellite numbering.

Conversion between (system, PRN), the internal satellite number and the
RINEX identifier, with the lookup tables that make it O(1)."""

import numpy as np

from cssrlib.types.enums import *  # noqa: F401,F403


def prn2sat(sys, prn):
    """ convert sys+prn to sat """
    if sys == uGNSS.GPS:
        sat = prn
    elif sys == uGNSS.GAL:
        sat = prn+uGNSS.GALMIN
    elif sys == uGNSS.QZS:
        sat = prn-192+uGNSS.QZSMIN
    elif sys == uGNSS.GLO:
        sat = prn+uGNSS.GLOMIN
    elif sys == uGNSS.BDS:
        sat = prn+uGNSS.BDSMIN
    elif sys == uGNSS.SBS:
        sat = prn-119+uGNSS.SBSMIN
    elif sys == uGNSS.IRN:
        sat = prn+uGNSS.IRNMIN
    else:
        sat = 0
    return sat


def _build_sat_lookup():
    """Pre-compute sat → sys, sat → prn arrays for O(1) tight-loop access."""
    n = int(uGNSS.MAXSAT) + 1
    sys_arr = np.full(n, int(uGNSS.NONE), dtype=np.int32)
    prn_arr = np.zeros(n, dtype=np.int32)
    cache = {}
    for sat in range(1, n):
        if sat > uGNSS.MAXSAT:
            prn = 0
            sys = uGNSS.NONE
        elif sat > uGNSS.IRNMIN:
            prn = sat-uGNSS.IRNMIN; sys = uGNSS.IRN
        elif sat > uGNSS.SBSMIN:
            prn = sat+119-uGNSS.SBSMIN; sys = uGNSS.SBS
        elif sat > uGNSS.GLOMIN:
            prn = sat-uGNSS.GLOMIN; sys = uGNSS.GLO
        elif sat > uGNSS.BDSMIN:
            prn = sat-uGNSS.BDSMIN; sys = uGNSS.BDS
        elif sat > uGNSS.QZSMIN:
            prn = sat+192-uGNSS.QZSMIN; sys = uGNSS.QZS
        elif sat > uGNSS.GALMIN:
            prn = sat-uGNSS.GALMIN; sys = uGNSS.GAL
        else:
            prn = sat; sys = uGNSS.GPS
        sys_arr[sat] = int(sys)
        prn_arr[sat] = int(prn)
        cache[sat] = (sys, prn)
    return sys_arr, prn_arr, cache


SAT_SYS_ARR, SAT_PRN_ARR, _SAT2PRN_CACHE = _build_sat_lookup()


def sat2prn(sat):
    """ convert sat to sys+prn (cached) """
    cached = _SAT2PRN_CACHE.get(sat)
    if cached is not None:
        return cached
    # Out-of-range fallback (extremely rare).
    if sat > uGNSS.MAXSAT:
        out = (uGNSS.NONE, 0)
    else:
        out = (uGNSS.NONE, 0)
    _SAT2PRN_CACHE[sat] = out
    return out


def sat2id(sat):
    """ convert satellite number to id """
    sys, prn = sat2prn(sat)
    gnss_tbl = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E', uGNSS.BDS: 'C',
                uGNSS.QZS: 'J', uGNSS.SBS: 'S', uGNSS.IRN: 'I'}
    if sys not in gnss_tbl:
        print(f"{sat} {sys} {prn}")
        return -1
    if sys == uGNSS.QZS:
        prn -= 192
    elif sys == uGNSS.SBS:
        prn -= 100
    return '%s%02d' % (gnss_tbl[sys], prn)


def id2sat(id_):
    """ convert id to satellite number """
    sys = char2sys(id_[0])
    if sys == uGNSS.NONE:
        return -1

    prn = int(id_[1:3])
    if sys == uGNSS.QZS:
        prn += 192
    elif sys == uGNSS.SBS:
        prn += 100
    sat = prn2sat(sys, prn)
    return sat


def char2sys(c):
    """ convert character to GNSS """
    gnss_tbl = {'G': uGNSS.GPS, 'R': uGNSS.GLO, 'E': uGNSS.GAL, 'C': uGNSS.BDS,
                'J': uGNSS.QZS, 'S': uGNSS.SBS, 'I': uGNSS.IRN}

    if c not in gnss_tbl:
        return uGNSS.NONE
    else:
        return gnss_tbl[c]


def sys2char(sys):
    """ convert gnss to character """
    gnss_tbl = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E', uGNSS.BDS: 'C',
                uGNSS.QZS: 'J', uGNSS.SBS: 'S', uGNSS.IRN: 'I'}

    if sys not in gnss_tbl:
        return "?"
    else:
        return gnss_tbl[sys]


def sys2str(sys):
    """ convert gnss to string """
    gnss_tbl = {uGNSS.GPS: 'GPS', uGNSS.GLO: 'GLONASS',
                uGNSS.GAL: 'GALILEO', uGNSS.BDS: 'BEIDOU',
                uGNSS.QZS: 'QZSS', uGNSS.SBS: 'SBAS', uGNSS.IRN: 'IRNSS'}

    if sys not in gnss_tbl:
        return "???"
    else:
        return gnss_tbl[sys]
