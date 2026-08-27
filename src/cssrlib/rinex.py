"""RINEX file access.

Facade over the decoder in :mod:`cssrlib.fileio.reader`; everything is
re-exported here so ``from cssrlib.rinex import rnxdec`` keeps working.
(The rnxenc encoder was deleted: it could not run -- it read a time
field Obs does not have -- and nothing called it.)
"""

from cssrlib.fileio.reader import (  # noqa: F401
    pclk_t,
    rnxdec,
    sync_obs,
    sync_obs_hold,
    auto_detect_signals,
)

__all__ = ['pclk_t', 'rnxdec', 'sync_obs', 'sync_obs_hold',
           'auto_detect_signals']
