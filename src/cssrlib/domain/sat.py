"""Satellite numbering.

Conversion between (system, PRN), the internal satellite number and the
RINEX identifier, with the lookup tables that make it O(1)."""

import numpy as np

from cssrlib.domain.enums import *  # noqa: F401,F403


# Single source of truth for constellation numbering:
#   (sys, rinex_char, sat_min, prn_offset, rinex_id_offset)
# Ordered from highest sat_min to lowest so range lookup matches the first hit.
_SAT_SYSTEMS = (
    (uGNSS.IRN, 'I', uGNSS.IRNMIN, 0, 0),
    (uGNSS.SBS, 'S', uGNSS.SBSMIN, 119, 100),
    (uGNSS.GLO, 'R', uGNSS.GLOMIN, 0, 0),
    (uGNSS.BDS, 'C', uGNSS.BDSMIN, 0, 0),
    (uGNSS.QZS, 'J', uGNSS.QZSMIN, 192, 192),
    (uGNSS.GAL, 'E', uGNSS.GALMIN, 0, 0),
    (uGNSS.GPS, 'G', 0, 0, 0),
)
_SYS_INFO = {sys: (ch, sat_min, prn_off, id_off)
             for sys, ch, sat_min, prn_off, id_off in _SAT_SYSTEMS}
_CHAR_TO_SYS = {ch: sys for sys, ch, _, _, _ in _SAT_SYSTEMS}


def prn2sat(sys, prn):
    """ convert sys+prn to sat """
    info = _SYS_INFO.get(sys)
    if info is None:
        return 0
    _, sat_min, prn_off, _ = info
    return prn - prn_off + sat_min


def _build_sat_lookup():
    """Pre-compute sat → sys, sat → prn arrays for O(1) tight-loop access."""
    n = int(uGNSS.MAXSAT) + 1
    sys_arr = np.full(n, int(uGNSS.NONE), dtype=np.int32)
    prn_arr = np.zeros(n, dtype=np.int32)
    cache = {}
    for sat in range(1, n):
        sys, prn = uGNSS.NONE, 0
        for s_id, _, sat_min, prn_off, _ in _SAT_SYSTEMS:
            if sat > sat_min:
                sys = s_id
                prn = sat + prn_off - sat_min
                break
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
    out = (uGNSS.NONE, 0)
    _SAT2PRN_CACHE[sat] = out
    return out


def sat2id(sat):
    """ convert satellite number to id """
    sys, prn = sat2prn(sat)
    info = _SYS_INFO.get(sys)
    if info is None:
        print(f"{sat} {sys} {prn}")
        return -1
    ch, _, _, id_off = info
    return '%s%02d' % (ch, prn - id_off)


def id2sat(id_):
    """ convert id to satellite number """
    sys = char2sys(id_[0])
    if sys == uGNSS.NONE:
        return -1
    prn = int(id_[1:3]) + _SYS_INFO[sys][3]
    return prn2sat(sys, prn)


def char2sys(c):
    """ convert character to GNSS """
    return _CHAR_TO_SYS.get(c, uGNSS.NONE)


def sys2char(sys):
    """ convert gnss to character """
    info = _SYS_INFO.get(sys)
    return info[0] if info is not None else "?"
