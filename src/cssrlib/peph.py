"""Precise products: orbits, clocks, antennas, biases, Earth frames.

Facade. The four groups that shared this 1,464-line module live in
:mod:`cssrlib.peph_sp3`, :mod:`cssrlib.antex`, :mod:`cssrlib.frames` and
:mod:`cssrlib.bsx`; everything is re-exported so existing imports work.
"""

from cssrlib.peph_sp3 import peph_t, peph                       # noqa: F401
from cssrlib.antex import (pcv_t, atxdec, searchpcv,            # noqa: F401
                           substSigTx, substSigRx,              # noqa: F401
                           antModelTx, antModelRx, apc2com)     # noqa: F401
from cssrlib.frames import (Rx, Ry, Rz, nut_iau1980,            # noqa: F401
                            time2sec, utc2gmst, orb2ecef,       # noqa: F401
                            eci2ecef, ast_args,                 # noqa: F401
                            sunmoonpos_eci, sunmoonpos)         # noqa: F401
from cssrlib.bsx import bias_t, biasdec                         # noqa: F401

# ppp.py imports these from here rather than from gnss; kept re-exported.
from cssrlib.gnss import gpst2utc, time2epoch                   # noqa: F401
