"""GNSS types, time scales, coordinates and data structures.

Facade. The module grew to 1,657 lines and 74 top-level definitions
covering six separable subjects; each now has its own file:

    cssrlib.domain.enums    constellations, signal bands, model selectors
    cssrlib.domain.signal   rSigRnx, the RINEX signal identifier
    cssrlib.domain.timescale     gtime_t and the time-scale conversions
    cssrlib.domain.sat      satellite numbering and identifiers
    cssrlib.domain.structs    Obs / Eph / Nav and the four Nav containers
    cssrlib.domain.coords   frames, geodesy, DOP, atmosphere wrappers

Everything is re-exported here, so ``from cssrlib.domain.gnss import Nav`` and the
rest of the existing imports are unaffected.
"""

from cssrlib.domain.enums import *      # noqa: F401,F403
from cssrlib.domain.signal import *     # noqa: F401,F403
from cssrlib.domain.timescale import *       # noqa: F401,F403
from cssrlib.domain.sat import *        # noqa: F401,F403
from cssrlib.domain.structs import *      # noqa: F401,F403
from cssrlib.domain.coords import *     # noqa: F401,F403

# `import *` skips underscore-prefixed names, but these are imported by name
# elsewhere in the package and by the tests.
from cssrlib.domain.enums import _ensure_vec                      # noqa: F401
from cssrlib.domain.sat import _SAT2PRN_CACHE                     # noqa: F401
from cssrlib.domain.structs import _NAV_FIELDS, _nav_property       # noqa: F401
