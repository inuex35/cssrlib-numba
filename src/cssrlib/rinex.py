"""RINEX file access.

Facade over the decoder and the encoder, which live in
:mod:`cssrlib.rinex_reader` and :mod:`cssrlib.rinex_writer`. They shared
a 1,685-line module without sharing any code; everything is re-exported
here so ``from cssrlib.rinex import rnxdec`` keeps working.
"""

from cssrlib.rinex_reader import (  # noqa: F401
    pclk_t,
    rnxdec,
    sync_obs,
    sync_obs_hold,
    auto_detect_signals,
)
from cssrlib.rinex_writer import rnxenc  # noqa: F401

__all__ = ['pclk_t', 'rnxdec', 'rnxenc', 'sync_obs', 'sync_obs_hold',
           'auto_detect_signals']
