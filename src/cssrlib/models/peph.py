"""Precise products: orbits, clocks, antennas, biases, Earth frames.

Facade. The four groups that shared this 1,464-line module live in
:mod:`cssrlib.models.precise`, :mod:`cssrlib.models.antenna`, :mod:`cssrlib.models.frames` and
:mod:`cssrlib.models.bias`; everything is re-exported so existing imports work.
"""

from cssrlib.models.precise import peph_t, peph                       # noqa: F401
from cssrlib.models.antenna import (pcv_t, atxdec, searchpcv,            # noqa: F401
                           substSigTx, substSigRx,              # noqa: F401
                           antModelTx, antModelRx, apc2com)     # noqa: F401
from cssrlib.models.frames import (Rx, Ry, Rz, nut_iau1980,            # noqa: F401
                            time2sec, utc2gmst, orb2ecef,       # noqa: F401
                            eci2ecef, ast_args,                 # noqa: F401
                            sunmoonpos_eci, sunmoonpos)         # noqa: F401
from cssrlib.models.bias import bias_t, biasdec                         # noqa: F401

# ppp.py imports these from here rather than from gnss; kept re-exported.
from cssrlib.domain.gnss import gpst2utc, time2epoch                   # noqa: F401
