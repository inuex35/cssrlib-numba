"""Vocabulary for State-Space Representation corrections.

Pure enumerations: which service a correction stream came from, and which
kind of correction a message carries. No decoding logic, no dependencies.

They live here rather than in :mod:`cssrlib.ssr.base` because the consumers
are spread across the stack. ``ephemeris.satposs`` needs to ask "is this an
orbit correction?" while applying one, and importing that question from the
1,300-line Compact SSR decoder inverted the dependency: broadcast-ephemeris
RTK, which decodes no SSR at all, still pulled the decoder (and bitstruct)
in through ``ephemeris``.

:mod:`cssrlib.ssr.base` re-exports both names, so
``from cssrlib.ssr.base import sCType`` keeps working.
"""

from enum import IntEnum


class sCSSRTYPE(IntEnum):
    """ class to define the SSR service a correction stream comes from """
    QZS_CLAS = 0     # QZS CLAS PPP-RTK
    QZS_MADOCA = 1   # MADOCA-PPP
    GAL_HAS_SIS = 2  # Galileo HAS Signal-In-Space
    GAL_HAS_IDD = 3  # Galileo HAS Internet Data Distribution
    BDS_PPP = 4      # BDS PPP
    IGS_SSR = 5
    RTCM3_SSR = 6
    PVS_PPP = 7      # PPP via SouthPAN
    SBAS_L1 = 8      # L1 SBAS
    SBAS_L5 = 9      # L5 SBAS (DFMC)
    DGPS = 10        # DGPS (QZSS SLAS)
    STDPOS = 11


class sCType(IntEnum):
    """ class to define correction message types """
    MASK = 0
    ORBIT = 1
    CLOCK = 2
    CBIAS = 3
    PBIAS = 4
    STEC = 5
    TROP = 6
    URA = 7
    AUTH = 8
    HCLOCK = 9
    VTEC = 10
    MAX = 11
