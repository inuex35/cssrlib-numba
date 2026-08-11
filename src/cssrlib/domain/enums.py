"""GNSS constants and enumerations.

Constellations, signal bands and types, and the model selectors. No logic;
imported by everything else."""

from enum import IntEnum
import numpy as np

from cssrlib.core import constants as _c


gpst0 = [1980, 1, 6, 0, 0, 0]  # GPS system time reference
gst0 = [1999, 8, 22, 0, 0, 0]  # Galileo system time reference
bdt0 = [2006, 1, 1, 0, 0, 0]  # BeiDou system time reference


class rCST():
    """ class for constants kept for backwards compatibility """
    pass


_R_CONSTANTS = (
    "CLIGHT", "MU_GPS", "MU_GLO", "MU_GAL", "MU_BDS", "J2_GLO", "GME", "GMS",
    "GMM", "OMGE", "OMGE_GLO", "OMGE_GAL", "OMGE_BDS", "RE_WGS84", "FE_WGS84",
    "RE_GLO", "AU", "D2R", "R2D", "AS2R", "DAY_SEC", "WEEK_SEC",
    "HALFWEEK_SEC", "CENTURY_SEC", "PI", "HALFPI", "FREQ_G1", "FREQ_G2",
    "FREQ_G5", "FREQ_R1", "FREQ_R1k", "FREQ_R2", "FREQ_R2k", "FREQ_R1a",
    "FREQ_R2a", "FREQ_R3", "FREQ_E1", "FREQ_E5a", "FREQ_E5b", "FREQ_E5",
    "FREQ_E6", "FREQ_C1", "FREQ_C12", "FREQ_C2a", "FREQ_C2b", "FREQ_C2",
    "FREQ_C3", "FREQ_J1", "FREQ_J2", "FREQ_J5", "FREQ_J6", "FREQ_S1",
    "FREQ_S5", "FREQ_I1", "FREQ_I5", "FREQ_IS", "COS_5", "SIN_5",
    "P2_5", "P2_6", "P2_8", "P2_9", "P2_10", "P2_11", "P2_12", "P2_13",
    "P2_14", "P2_15", "P2_16", "P2_17", "P2_19", "P2_20", "P2_21", "P2_24",
    "P2_26", "P2_27", "P2_28", "P2_29", "P2_30", "P2_31", "P2_32", "P2_33",
    "P2_34", "P2_35", "P2_37", "P2_38", "P2_39", "P2_40", "P2_41", "P2_43",
    "P2_44", "P2_46", "P2_48", "P2_49", "P2_50", "P2_51", "P2_55", "P2_57",
    "P2_59", "P2_60", "P2_66", "P2_68", "SC2RAD",
)

for _name in _R_CONSTANTS:
    setattr(rCST, _name, getattr(_c, _name))


def _ensure_vec(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape == (3,):
        return arr
    return arr.reshape(3)


class uGNSS(IntEnum):
    """ class for GNS systems """

    NONE = -1

    GPS = 0
    GAL = 1
    QZS = 2
    BDS = 3
    GLO = 4
    SBS = 5
    IRN = 6

    GNSSMAX = 7

    GPSMAX = 32
    GALMAX = 36
    QZSMAX = 17
    BDSMAX = 63
    GLOMAX = 32
    SBSMAX = 39
    IRNMAX = 14

    GPSMIN = 0
    GALMIN = GPSMIN+GPSMAX
    QZSMIN = GALMIN+GALMAX
    BDSMIN = QZSMIN+QZSMAX
    GLOMIN = BDSMIN+BDSMAX
    SBSMIN = GLOMIN+GLOMAX
    IRNMIN = SBSMIN+SBSMAX

    MAXSAT = GPSMAX+GALMAX+QZSMAX+BDSMAX+GLOMAX+SBSMAX+IRNMAX


class uTYP(IntEnum):
    """ class for observation types"""

    NONE = -1

    C = 1
    L = 2
    D = 3
    S = 4


class uSIG(IntEnum):
    """ class for signal band and attribute """

    NONE = -1

    # GPS   L1  1575.42 MHz
    # GLO   G1  1602+k*9/16 MHz
    # GAL   E1  1575.42 MHz
    # SBAS  L1  1575.42 MHz
    # QZSS  L1  1575.42 MHz
    # BDS-3 B1  1575.42 MHz
    L1 = 100
    L1A = 101
    L1B = 102
    L1C = 103
    L1D = 104
    L1E = 105
    L1L = 112
    L1M = 113
    L1N = 114
    L1P = 116
    L1S = 119
    L1W = 123
    L1X = 124
    L1Y = 125
    L1Z = 126

    # GPS   L2  1227.60  MHz
    # GLO   G2  1246+k*7/16 MHz
    # QZS   L2  1227.60  MHz
    # BDS   B1  1561.098 MHz
    L2 = 200
    L2C = 203
    L2D = 204
    L2I = 209
    L2L = 212
    L2M = 213
    L2N = 214
    L2P = 216
    L2Q = 217
    L2S = 219
    L2W = 223
    L2X = 224
    L2Y = 225

    # GLO G3 1202.025 MHz
    L3 = 300
    L3I = 309
    L3Q = 317
    L3X = 324

    # GLO G1a 1600.995 MHz
    L4 = 400
    L4A = 401
    L4B = 402
    L4X = 424

    # GPS   L5  1176.45 MHz
    # GAL   E5  1176.45 MHz
    # SBS   L5  1176.45 MHz
    # QZS   L5  1176.45 MHz
    # BDS-3 B2a 1176.45 MHz
    # IRN   L5  1176.45 MHz
    L5 = 500
    L5A = 501
    L5B = 502
    L5C = 503
    L5D = 504
    L5I = 509
    L5P = 516
    L5Q = 517
    L5X = 524
    L5Z = 526

    # GLO   G2a 1248.06 MHz
    # GAL   E6  1278.75 MHz
    # QZS   L6  1278.75 MHz
    # BDS   B3  1278.75 MHz
    L6 = 600
    L6A = 601
    L6B = 602
    L6C = 603
    L6D = 604
    L6E = 605
    L6I = 609
    L6L = 612
    L6P = 616
    L6Q = 617
    L6S = 619
    L6X = 624
    L6Z = 626

    # GAL   E5b 1207.14 MHz
    # BDS-2 B2  1207.14 MHz
    # BDS-3 B2b 1207.14 MHz
    L7 = 700
    L7D = 704
    L7I = 709
    L7P = 716
    L7Q = 717
    L7X = 724
    L7Z = 726

    # GAL  E5a+b 1191.795 MHz
    # BDS  B2a+b 1191.795 MHz
    L8 = 800
    L8D = 804
    L8I = 809
    L8P = 816
    L8Q = 817
    L8X = 824

    # IRN  S    2492.028 MHz
    L9 = 900
    L9A = 901
    L9B = 902
    L9C = 903
    L9X = 924


class uTropoModel(IntEnum):
    """
    Enumeration for tropo model selection
    """

    NONE = -1
    SAAST = 0
    HOPF = 1


class uIonoModel(IntEnum):
    """
    Enumeration for iono model selection
    """

    NONE = -1
    KLOBUCHAR = 0
    NEQUICK_G = 1
    GIM = 2
    SBAS = 3


class uTideModel(IntEnum):
    """
    Enumeration for Earth tide model selection
    """

    NONE = -1
    SIMPLE = 0
    IERS2010 = 1
