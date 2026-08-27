"""Observation, ephemeris and navigation data structures.

Includes the four containers Nav delegates to -- NavData, ProcConfig,
ReceiverState and FilterState."""

import numpy as np

from cssrlib.domain.enums import *  # noqa: F401,F403
from cssrlib.domain.enums import uTideModel
from cssrlib.domain.timescale import *  # noqa: F401,F403


# The three RINEX-4 parameter records below hold their arrays per instance.
# They used to be class attributes, which meant one array shared by every
# instance ever made: the decoder writes some of them in place (NeQuick-G,
# BDGIM, EOP, STO) and only rebinds others (Klobuchar), so a Galileo
# ionosphere record and a BeiDou one were the same nine numbers, whichever
# was read last, and decoding two files in one process carried the first
# file's parameters into the second.


class STOParam():
    """ System Time and UTC Office """

    def __init__(self):
        self.sbas = 0  # SBAS ID
        self.prm = [0, 0]  # System time offset parameter
        self.t_ot = None  # reference epoch
        self.t_t = 0.0  # transmission time of message (Time of week [sec])
        self.a = np.zeros(3)  # a0, a1, a2


class EOPParam():
    """ Earth Orientation Parameter """

    def __init__(self):
        # EOP parameters (xp,dxp,ddxp,yp,dyp,ddyp,dut1,ddut1,dddut1)
        self.prm = np.zeros(9)
        self.t_ot = None  # reference epoch
        self.t_t = 0.0  # transmission time of message (Time of week [sec])


class IONParam():
    """ Ionospheric delay model Parameter """

    def __init__(self):
        self.iod = 0
        self.prm = np.zeros(9)  # ION parameters
        self.t_tm = None  # transmission time
        self.region = None


class Obs():
    """ class to define the observation """

    def __init__(self):
        self.t = gtime_t()
        self.P = []
        self.L = []
        self.D = []
        self.S = []
        self.lli = []
        self.sat = []
        self.sig = {}


class Eph():
    """ class to define ephemeris """
    sat = 0
    iode = 0
    iodc = 0
    af0 = 0.0
    af1 = 0.0
    af2 = 0.0
    week = 0
    toc = 0
    toe = 0
    tot = 0
    wn_op = 0
    top = 0
    crs = 0.0
    crc = 0.0
    cus = 0.0
    cus = 0.0
    cis = 0.0
    cic = 0.0
    e = 0.0
    i0 = 0.0
    A = 0.0
    Adot = 0.0
    deln = 0.0
    delnd = 0.0
    M0 = 0.0
    OMG0 = 0.0
    OMGd = 0.0
    omg = 0.0
    idot = 0.0
    tgd = 0.0
    tgd_b = 0.0
    tgd_c = 0.0
    sva = 0
    svh = 0
    fit = 0
    toes = 0
    tops = 0
    l2p = 0
    sattype = 0
    sismai = 0
    code = 0
    urai = None
    sisai = None
    isc = None
    integ = 0
    # 0:LNAV,INAV,D1/D2, 1:CNAV/CNAV1/FNAV, 2: CNAV2, 3: CNAV3, 4:FDMA, 5:SBAS
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class Geph():
    """ class to define GLONASS ephemeris """
    sat = 0
    iode = 0  # IODE: 0-6bit of tb field
    frq = 0
    svh = 0
    sva = 0
    age = 0.0
    toes = 0.0
    taun = 0.0         # SV clock bias [s]
    gamn = 0.0         # SV clock drift [s/s]
    beta = 0.0         # SV clock drift rate [s/s^2]
    dtaun = 0.0        # delta between L1 and L2 [s]
    mode = 0
    status = 0  # data validity
    flag = 0

    tau_c = 0.0  # GLONASS time scale correction to UTC(SU) time
    dtau_c = 0.0
    tau_gps = 0.0  # correction to GPS time relative to GLONASS time

    psi = 0.0  # yaw angle [rad]
    sn = 0  # sign flag
    win = 0.0  # angular rate [rad/s]
    dw = 0.0  # angular accel [rad/s^2]
    wmax = 0.0  # max angular rate[rad/s]
    aode = 0
    aodc = 0  # age of data orbit/clock [days]
    tin = 0.0
    tau1 = 0.0
    tau2 = 0.0

    src = 0  # source flags (b0-1: Rt, b2-3: Re)
    sattype = 0  # 0 - M(L3), 1 - K1(L3), 3 - K1(L2/L3), 2 - K2 (L1/L2/L3)

    def __init__(self, sat=0):
        self.sat = sat
        # Mutable state lives on the instance. As class attributes these
        # arrays (and the gtime_t epochs) were one object shared by every
        # ephemeris ever constructed; pos/vel/acc escaped because the
        # decoder rebinds them, but urai is written in place there, so
        # every GLONASS satellite in a file reported whichever URAI was
        # decoded last. Same repair the RINEX-4 parameter records got.
        self.toe = gtime_t()
        self.tof = gtime_t()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        # for CDMA
        self.urai = np.zeros(2, dtype=int)
        self.dpos = np.zeros(3)
        self.isc = np.zeros(3)  # 0: ISC_L1OC, 1: ISC_L2OC, 2: ISC_L3OC


class Seph():
    """ class to define SBAS ephemeris """
    sat = 0
    iodn = 0
    svh = 0
    sva = 0
    af0 = 0.0
    af1 = 0.0
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat
        # Per instance for the same reason as Geph above.
        self.t0 = gtime_t()
        self.tof = gtime_t()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)


class Alm():
    """ class to define almanac """
    sat = 0
    af0 = 0.0
    af1 = 0.0
    toa = gtime_t()
    toas = 0.0
    week = 0
    e = 0.0
    i0 = 0.0
    A = 0.0
    M0 = 0.0
    OMG0 = 0.0
    OMGd = 0.0
    omg = 0.0
    svh = 0
    sattype = 0
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class NavData():
    """Navigation data: ephemerides, corrections, antennas.

    Read-mostly, and identical for every receiver in a session -- which is
    why rover and base can share one instance instead of the base getting a
    deepcopy of everything.
    """

    def __init__(self):
        self.eph = []
        self.geph = []
        self.seph = []
        self.peph = []
        self.pclk = []
        self.ne = 0
        self.nc = 0

        self.ion = np.array([
            [0.1118E-07, -0.7451E-08, -0.5961E-07, 0.1192E-06],
            [0.1167E+06, -0.2294E+06, -0.1311E+06, 0.1049E+07]])
        self.ion_gim = np.zeros(9)
        self.ion_region = 0  # 0: wide-area, 1: Japan-aera (QZSS only)

        self.sto_prm = {}
        self.eop_prm = {}
        self.ion_prm = {}
        self.eop = np.zeros(9)
        self.leaps = 18  # leap seconds [s]

        # GLONASS frequency channel table
        self.glo_ch = {}

        self.sat_ant = None
        # One receiver antenna. There was a second, rcv_ant_b, for the base
        # -- never set by anything, and reachable only through an rtype flag
        # nothing passed.
        self.rcv_ant = None

        # SSR correction placeholder
        self.dorb = np.zeros(uGNSS.MAXSAT)
        self.dclk = np.zeros(uGNSS.MAXSAT)
        self.dsis = np.zeros(uGNSS.MAXSAT)
        self.sis = np.zeros(uGNSS.MAXSAT)


class ProcConfig():
    """How to process: models, thresholds, error budget.

    Fixed for the duration of a run. The difference between an RTK and a
    PPP-RTK session is a difference in these values, not in class.
    """

    def __init__(self, nf=2):
        self.nf = nf
        self.pmode = 1   # 0: static, 1: kinematic
        self.rmode = 0   # 0: IF not applied, 1: IF for L1/L2, 2: IF for L1/L5
        self.ephopt = 2  # 0: BRDC, 1: SBAS, 2: SSR-APC, 3: SSR-CG, 4: PREC

        self.elmin = np.deg2rad(15.0)
        self.elmaskar = np.deg2rad(20.0)  # elevation mask for AR

        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.armode = 0
        self.parmode = 2   # LAMBDA search: 1 full ILS, 2 partial AR
        self.par_P0 = 0.995
        self.thresar = 3.0  # AR acceptance threshold

        # cycle-slip threshold of geometry-free combination of phase [m]
        self.thresslip = 0.15

        self.trpModel = uTropoModel.SAAST
        self.ionoModel = uIonoModel.KLOBUCHAR
        # uTideModel.NONE, by name: the previous `False` compared equal
        # to uTideModel.SIMPLE (== 0), so a bare Nav() silently applied
        # the solid-earth tide model. The config factories overwrite
        # this, which is how it went unnoticed.
        self.tidecorr = uTideModel.NONE

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.trop_opt = 0
        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.iono_opt = 0
        # 0: none, 1: full model, 2: local/regional model
        self.phw_opt = 1
        self.csmooth = False

        self.monlevel = 1
        self.cnr_min = 25
        self.cnr_min_gpy = 15
        # Judge each satellite over the bands it actually transmits.
        #
        # The gate always judges over the bands the system selected; with
        # this False (the default) a satellite missing one of them -- a
        # pre-IIF GPS that carries no L5, a BeiDou-2 that carries only
        # B1I -- is dropped outright, all session. On tokyo run2 that
        # discards 19 of 47 satellites structurally, among them a 91%-
        # present, 48 dB-Hz GPS. With True, the judgment set per satellite
        # is the selected bands it has ever produced this session
        # (ReceiverState.band_seen); WITHIN that set the gate stays
        # strict, so a satellite whose transmitted band degrades is
        # dropped exactly as before. 2026-07 measured admitting these
        # populations harmful on that estimator ("L5-less GPS / B1I-only
        # BDS-2 poison the urban float"); this flag exists so the current
        # estimator can measure it again rather than inherit the verdict.
        self.sat_band_plan = False
        self.maxout = 5  # maximum outage [epochs]

        self.excl_sat = []   # Excluded satellites
        self.rb = [0, 0, 0]  # base station position in ECEF [m]
        self.baseline = 0    # baseline length [km]

        # Observation error budget.
        self.eratio = np.ones(nf) * 50
        self.err = [0, 0.003, 0.003]

        # Initial state standard deviations.
        # NOTE sig_p0 is 100 m, not the 30 m that rtkpos and ppprtkpos used
        # to assign: they set it *after* gnssobs.__init__ had already built
        # P from it, and nothing reads sig_p0 again, so the 30 never took
        # effect. The value here is what the filter actually starts with;
        # changing it to 30 is a tuning decision, not a refactor.
        self.sig_p0 = 100.0    # [m]
        self.sig_v0 = 1.0      # [m/s]
        self.sig_ztd0 = 0.1    # [m]
        self.sig_ion0 = 10.0   # [m]
        self.sig_n0 = 30.0     # [cyc]

        # Process noise standard deviations; sig_qp / sig_qv depend on pmode
        # and are set by the configuration factories.
        self.sig_qp = 0.01 / np.sqrt(1)          # [m/sqrt(s)]
        self.sig_qv = 1.0 / np.sqrt(1)           # [m/s/sqrt(s)]
        self.sig_qztd = 0.05 / np.sqrt(3600)     # [m/sqrt(s)]
        self.sig_qion = 10.0 / np.sqrt(1)        # [m/s/sqrt(s)]
        self.sig_qb = 1e-4 / np.sqrt(1)          # [m/s/sqrt(s)]

        # RTKLIB-compatible AR extras
        self.maxtdiff = 30.0     # [s] max age of base observations
        self.rtklib_mode = False
        self.arfilter = True     # drop newly-acquired sats that hurt ratio
        self.minfixsats = 4      # minimum sats required to attempt AR


class ReceiverState():
    """Per-receiver bookkeeping for one epoch and its history.

    One instance per receiver. rtkpos gives the base its own, which is what
    let the rover/base pair gf / gf_r collapse into a single ``gf``.
    """

    def __init__(self, nf=2):
        self.fix = np.zeros((uGNSS.MAXSAT, nf), dtype=int)
        self.edt = np.zeros((uGNSS.MAXSAT, nf), dtype=int)
        # Measurement outage indicator
        self.outc = np.zeros((uGNSS.MAXSAT, nf), dtype=int)
        # Carrier-phase processed indicator
        self.vsat = np.zeros((uGNSS.MAXSAT, nf), dtype=int)
        # Lock counter (RTKLIB ssat[].lock equivalent): consecutive epochs
        # the carrier phase has been valid. Resets to 0 on outage; used by
        # rtklib_mode arfilter to demote newly-acquired satellites.
        self.lock = np.zeros((uGNSS.MAXSAT, nf), dtype=int)
        # The demo5 retry's memory, per receiver like the lock counters it
        # works with: the round-robin cursor, and arfilter's two previous
        # ratios (pass-1 of the last epoch / of the last successful one).
        # These are runtime state, not configuration -- excsat lived in
        # ProcConfig for a while and prev_ratio1/2 fell through the gap
        # between the containers entirely (cold-start AttributeError).
        self.excsat = 0
        self.prev_ratio1 = 0.0
        self.prev_ratio2 = 0.0
        # Cycle-slip flag (LLI or GF slip detected at qcedit). Causes
        # ambiguity reset in udstate without dropping the observation.
        # Cleared by udstate after the reset is applied.
        self.slip = np.zeros((uGNSS.MAXSAT, nf), dtype=int)

        # Which selected bands this satellite has ever produced (L and P
        # both nonzero at least once this session), per receiver. This is
        # the observable proxy for the satellite's signal plan: a Block
        # IIR GPS never shows L5, a BeiDou-2 never shows B1C/B2a. Sticky
        # for the session -- a band once seen stays seen -- so a tracking
        # dropout does not masquerade as "not transmitted".
        self.band_seen = np.zeros((uGNSS.MAXSAT, nf), dtype=bool)

        # geometry-free combination for cycle-slip detection, this
        # receiver's own. There used to be a second table, gf_r, because one
        # Nav had to hold the base's as well.
        self.gf = np.zeros(uGNSS.MAXSAT)

        self.el = np.zeros(uGNSS.MAXSAT)
        self.phw = np.zeros(uGNSS.MAXSAT)

        self.sat = np.zeros(0, dtype=int)
        self.t = gtime_t()
        self.tt = 0
        # SSR signal-in-space bookkeeping (models/ephemeris.py): the
        # previous ORBIT-correction epoch. It was read there before any
        # code ever assigned it -- AttributeError on the first SSR epoch.
        self.time_p = gtime_t()

        self.smode = 0  # 0:NONE,1:std,2:DGPS,4:fix,5:float
        # number of satellites (observed, calculated, corrected)
        self.nsat = [0, 0, 0]


class FilterState():
    """The estimator's state vector and covariance.

    Sized by :class:`cssrlib.estimation.layout.StateLayout`; an external estimator (the
    GTSAM double-difference workflow) simply does not create one.
    """

    def __init__(self):
        self.x = np.zeros(0)
        self.P = np.zeros((0, 0))
        self.xa = np.zeros(0)
        self.Pa = np.zeros((0, 0))
        self.y = np.zeros(0)

        self.na = 0
        self.nq = 0
        self.nx = 0
        self.ntrop = 0
        self.niono = 0


# Which container owns which attribute. Nav delegates through this map, so
# every existing `nav.<field>` call site keeps working while the data is
# actually stored in the component it belongs to.
_NAV_FIELDS = {
    "data": ("eph", "geph", "seph", "peph", "pclk", "ne", "nc",
             "ion", "ion_gim", "ion_region", "sto_prm", "eop_prm",
             "ion_prm", "eop", "leaps", "glo_ch",
             "sat_ant", "rcv_ant",
             "dorb", "dclk", "dsis", "sis"),
    "cfg": ("nf", "pmode", "rmode", "ephopt", "elmin", "elmaskar",
            "armode", "parmode", "par_P0", "thresar", "thresslip",
            "trpModel", "ionoModel", "tidecorr", "trop_opt", "iono_opt",
            "phw_opt", "csmooth", "monlevel", "cnr_min", "cnr_min_gpy",
            "maxout", "excl_sat", "rb", "baseline", "eratio", "err",
            "sig_p0", "sig_v0", "sig_ztd0", "sig_ion0", "sig_n0",
            "sig_qp", "sig_qv", "sig_qztd", "sig_qion", "sig_qb",
            "maxtdiff", "rtklib_mode", "arfilter",
            "minfixsats", "sat_band_plan"),
    "rcv": ("fix", "edt", "outc", "vsat", "lock", "slip", "gf",
            "excsat", "prev_ratio1", "prev_ratio2",
            "el", "phw", "sat", "t", "tt", "smode", "nsat", "time_p"),
    "flt": ("x", "P", "xa", "Pa", "y", "na", "nq", "nx", "ntrop", "niono"),
}


def _nav_property(field, component):
    def getter(self):
        return getattr(getattr(self, component), field)

    def setter(self, value):
        setattr(getattr(self, component), field, value)

    getter.__name__ = field
    return property(getter, setter,
                    doc=f"Delegated to ``Nav.{component}.{field}``.")


class Nav():
    """ class to define the navigation message

    Kept as the single object every call site already passes around, but it
    now owns four containers rather than 57 loose fields:

    ``data``
        :class:`NavData` -- ephemerides and corrections, shareable.
    ``cfg``
        :class:`ProcConfig` -- how to process; the only thing that differs
        between an RTK and a PPP-RTK run.
    ``rcv``
        :class:`ReceiverState` -- per-receiver bookkeeping.
    ``flt``
        :class:`FilterState` -- the estimator's x and P.

    Attribute access is delegated, so ``nav.eph`` and ``nav.data.eph`` are
    the same object and no existing caller has to change.
    """

    def __init__(self, nf=2):
        self.data = NavData()
        self.cfg = ProcConfig(nf=nf)
        self.rcv = ReceiverState(nf=nf)
        self.flt = FilterState()


for _component, _fields in _NAV_FIELDS.items():
    for _field in _fields:
        setattr(Nav, _field, _nav_property(_field, _component))
del _component, _fields, _field
