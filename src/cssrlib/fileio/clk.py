"""RINEX clock files."""

import numpy as np
from cssrlib.gnss import uGNSS
from cssrlib.gnss import gtime_t


class pclk_t:
    """ class for precise clock data """

    def __init__(self, time=None):
        if time is not None:
            self.time = time
        else:
            self.time = gtime_t()
        self.clk = np.zeros(uGNSS.MAXSAT)
        self.std = np.zeros(uGNSS.MAXSAT)


class ClockFileMixin:
    """Mixed into :class:`~cssrlib.fileio.reader.rnxdec`."""

