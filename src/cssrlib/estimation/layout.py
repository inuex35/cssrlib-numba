"""Layout of the EKF state vector.

The estimator packs several kinds of unknown into one vector. Until now the
index arithmetic for that packing lived as three one-line methods on the
engine class and was reproduced at a dozen call sites, which let two of the
formulas drift out of step with the sizing code that allocates the vector.

The vector is laid out as::

    [0,             npos)          position, or position + velocity
    [npos,          npos+ntrop)    zenith tropospheric delay
    [npos+ntrop,    na)            slant ionospheric delay, one per satellite
    [na,            nx)            carrier-phase ambiguities, nf per satellite

``na`` marks the end of the "fixed-size" block the ambiguity resolver treats
as nuisance parameters; ``nx`` is the full length.
"""

from cssrlib.domain.gnss import uGNSS


class StateLayout:
    """Where each unknown sits in the state vector.

    Parameters
    ----------
    pmode : int
        0 for a position-only state, non-zero to also estimate velocity.
    nf : int
        Number of frequency slots carried per satellite.
    ntrop : int
        1 if a zenith tropospheric delay is estimated, else 0.
    niono : int
        ``uGNSS.MAXSAT`` if slant ionospheric delays are estimated, else 0.
    maxsat : int
        Satellite slot count; overridable for testing.
    """

    __slots__ = ("pmode", "nf", "ntrop", "niono", "maxsat",
                 "npos", "na", "nq", "nx")

    def __init__(self, pmode, nf, ntrop, niono, maxsat=None):
        self.pmode = int(pmode)
        self.nf = int(nf)
        self.ntrop = int(ntrop)
        self.niono = int(niono)
        self.maxsat = int(uGNSS.MAXSAT if maxsat is None else maxsat)

        self.npos = 3 if self.pmode == 0 else 6
        self.na = self.npos + self.ntrop + self.niono
        self.nq = self.na
        self.nx = self.na + self.maxsat * self.nf

    @classmethod
    def from_nav(cls, nav):
        """Build the layout implied by an already-configured ``Nav``."""
        return cls(getattr(nav, "pmode", 0), nav.nf,
                   getattr(nav, "ntrop", 0), getattr(nav, "niono", 0))

    def apply_to(self, nav):
        """Publish the sizes onto ``nav``, which the rest of the code reads."""
        nav.ntrop = self.ntrop
        nav.niono = self.niono
        nav.na = self.na
        nav.nq = self.nq
        nav.nx = self.nx
        return nav

    # -- slices ---------------------------------------------------------
    @property
    def position(self):
        """Slice covering position (and velocity, when estimated)."""
        return slice(0, self.npos)

    @property
    def ambiguities(self):
        """Slice covering every carrier-phase ambiguity."""
        return slice(self.na, self.nx)

    # -- individual indices ---------------------------------------------
    def tropo(self, na=None):
        """Index of the zenith tropospheric delay.

        The old formula was ``na - MAXSAT - 1``, which silently assumed the
        ionospheric block was present and full length; with tropo estimated
        but iono not, it addressed off the front of the vector.
        """
        na = self.na if na is None else na
        return na - self.niono - self.ntrop

    def iono(self, sat, na=None):
        """Index of the slant ionospheric delay for satellite ``sat``."""
        na = self.na if na is None else na
        return na - self.niono + sat - 1

    def ambiguity(self, sat, f, na=None):
        """Index of the phase ambiguity for satellite ``sat``, band ``f``."""
        na = self.na if na is None else na
        return na + self.maxsat * f + sat - 1

    def __repr__(self):
        return (f"StateLayout(npos={self.npos}, ntrop={self.ntrop}, "
                f"niono={self.niono}, nf={self.nf}, na={self.na}, "
                f"nx={self.nx})")
