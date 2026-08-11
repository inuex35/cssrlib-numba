"""RINEX clock files."""

import numpy as np
from cssrlib.domain.gnss import uGNSS
from cssrlib.domain.gnss import timediff, gtime_t
from cssrlib.domain.gnss import prn2sat, char2sys


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

    def decode_clk(self, clkfile, nav):
        """decode Clock-RINEX data from file """

        # Offset for Clock-RINEX v3.x data section
        #
        offs = None

        nav.pclk = []
        fnav = open(clkfile, 'rt')

        # Read header section
        #
        for line in fnav:

            if 'RINEX VERSION / TYPE' in line:
                ver = float(line[0:20])
                offs = 0 if ver < 3.04 else 5

            if 'END OF HEADER' in line:
                break

        # Read data section
        #
        for line in fnav:

            if line[0:2] != 'AS':
                continue

            sys = char2sys(line[3])
            prn = int(line[4:7])
            if sys == uGNSS.QZS:
                prn += 192
            sat = prn2sat(sys, prn)

            t = self.decode_time(line, offs+8, 9)

            if nav.nc <= 0 or abs(timediff(nav.pclk[-1].time, t)) > 1e-9:
                nav.nc += 1
                pclk = pclk_t()
                pclk.time = t
                nav.pclk.append(pclk)

            nrec = int(line[offs+35:offs+37])
            clk = float(line[offs+40:offs+59])
            std = float(line[offs+61:offs+80]) if nrec >= 2 else 0.0

            nav.pclk[nav.nc-1].clk[sat-1] = clk
            nav.pclk[nav.nc-1].std[sat-1] = std

        return nav

