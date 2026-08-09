"""RINEX signal identifiers.

rSigRnx parses three-character RINEX observation codes and resolves them
to frequencies and wavelengths."""


from cssrlib.gnss_enums import *  # noqa: F401,F403
from cssrlib.gnss_sat import *  # noqa: F401,F403


class rSigRnx():
    # Class-level memoization for frequency()/wavelength(): keyed by
    # (sys, sig, k) since frequency depends only on those values, not on
    # which instance carries them.
    _FREQ_CACHE = {}

    def __init__(self, *args, **kwargs):
        """ Constructor """

        self.sys = uGNSS.NONE
        self.typ = uTYP.NONE
        self.sig = uSIG.NONE

        # Empty
        if len(args) == 0:

            self.sys = uGNSS.NONE
            self.typ = uTYP.NONE
            self.sig = uSIG.NONE

        # Four char string e.g. GC1W
        elif len(args) == 1:

            [x] = args
            if isinstance(x, str) and 3 <= len(x) <= 4:
                tmp = rSigRnx()
                tmp.str2sig(char2sys(x[0]), x[1:])
                self.sys = tmp.sys
                self.typ = tmp.typ
                self.sig = tmp.sig
            else:
                raise ValueError

        # System and three char string e.g. GPS, C1W
        elif len(args) == 2:

            sys, sig = args
            if isinstance(sys, uGNSS) and isinstance(sig, str) and \
                    2 <= len(sig) <= 3:
                tmp = rSigRnx()
                tmp.str2sig(sys, sig)
                self.sys = tmp.sys
                self.typ = tmp.typ
                self.sig = tmp.sig
            else:
                raise ValueError

        # System, type and signal
        elif len(args) == 3:

            sys, typ, sig = args
            if isinstance(sys, uGNSS) and \
                    isinstance(typ, uTYP) and \
                    isinstance(sig, uSIG):
                self.sys = sys
                self.typ = typ
                self.sig = sig
            else:
                raise ValueError

        else:

            raise ValueError

    def __repr__(self) -> str:
        """ string representation """
        return sys2char(self.sys)+self.str()

    def __eq__(self, other):
        """ equality operator """
        return self.sys == other.sys and \
            self.typ == other.typ and \
            self.sig == other.sig

    def __hash__(self):
        """ hash operator """
        return hash((self.sys, self.typ, self.sig))

    def toTyp(self, typ):
        """ Replace signal type """
        if isinstance(typ, uTYP):
            return rSigRnx(self.sys, typ, self.sig)
        else:
            raise ValueError

    def toAtt(self, att=""):
        """ Replace signal attribute """
        if isinstance(att, str):
            return rSigRnx(self.sys, self.str()[0:2]+att)
        else:
            raise ValueError

    def isGPS_PY(self):
        """
        Check if signal is GPS P(Y) tracking
        """
        return self.sys == uGNSS.GPS and \
            (self.sig == uSIG.L1W or self.sig == uSIG.L2W)

    def str2sig(self, sys, s):
        """ string to signal code conversion """

        if isinstance(sys, uGNSS) and isinstance(s, str):
            self.sys = sys
        else:
            raise ValueError

        s = s.strip()
        if len(s) < 2:
            raise ValueError

        if s[0] == 'C':
            self.typ = uTYP.C
        elif s[0] == 'L':
            self.typ = uTYP.L
        elif s[0] == 'D':
            self.typ = uTYP.D
        elif s[0] == 'S':
            self.typ = uTYP.S
        else:
            raise ValueError

        # Convert frequency ID
        #
        sig = int(s[1])*100

        # Check for valid tracking attribute
        #
        if len(s) == 3:
            if sys == uGNSS.GPS:
                if (s[1] == '1' and s[2] not in 'CSLXPWYM') or \
                   (s[2] == '2' and s[2] not in 'CDSLXPWYMN') or \
                   (s[2] == '5' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.GLO:
                if (s[1] == '1' and s[2] not in 'CPX') or \
                   (s[1] == '2' and s[2] not in 'CPX') or \
                   (s[1] == '3' and s[2] not in 'IQX') or \
                   (s[1] == '4' and s[2] not in 'ABX') or \
                   (s[1] == '6' and s[2] not in 'ABX'):
                    raise ValueError
            elif sys == uGNSS.GAL:
                if (s[1] == '1' and s[2] not in 'ABCXZ') or \
                   (s[1] == '5' and s[2] not in 'IQX') or \
                   (s[1] == '6' and s[2] not in 'ABCXZ') or \
                   (s[1] == '7' and s[2] not in 'IQX') or \
                   (s[1] == '8' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.SBS:
                if (s[1] == '1' and s[2] not in 'C') or \
                   (s[1] == '5' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.QZS:
                if (s[1] == '1' and s[2] not in 'CESLXZB') or \
                   (s[1] == '2' and s[2] not in 'SLX') or \
                   (s[1] == '5' and s[2] not in 'IQXDPZ') or \
                   (s[1] == '6' and s[2] not in 'SLXEZ'):
                    raise ValueError
            elif sys == uGNSS.BDS:
                if (s[1] == '2' and s[2] not in 'IQX') or \
                   (s[1] == '1' and s[2] not in 'DPXSLZ') or \
                   (s[1] == '5' and s[2] not in 'DPX') or \
                   (s[1] == '7' and s[2] not in 'IQXDPZ') or \
                   (s[1] == '8' and s[2] not in 'DPX') or \
                   (s[1] == '6' and s[2] not in 'IQXDPZ'):
                    raise ValueError
            elif sys == uGNSS.IRN:
                if (s[1] == '1' and s[2] not in 'DPX') or \
                   (s[1] == '5' and s[2] not in 'ABCX') or \
                   (s[1] == '9' and s[2] not in 'ABCX'):
                    raise ValueError

            sig += ord(s[2]) - ord('A') + 1

        self.sig = uSIG(sig)

    def str(self):
        """ signal code to string conversion """
        cached = getattr(self, '_str_cache', None)
        if cached is not None:
            return cached

        s = ''

        if self.typ == uTYP.C:
            s += 'C'
        elif self.typ == uTYP.L:
            s += 'L'
        elif self.typ == uTYP.D:
            s += 'D'
        elif self.typ == uTYP.S:
            s += 'S'
        else:
            return '???'

        s += '{}'.format(int(self.sig / 100))

        if self.sig % 100 == 0:
            s += ' '
        else:
            s += '{}'.format(chr(self.sig % 100+ord('A')-1))

        self._str_cache = s
        return s

    def band(self):
        """
        Retrieve signal band
        """
        return uSIG((self.sig//100)*100)

    def frequency(self, k=None):
        """ frequency in Hz (cached) """
        cache = self.__class__._FREQ_CACHE
        key = (int(self.sys), int(self.sig), k)
        cached = cache.get(key)
        if cached is not None:
            return cached[0]  # Sentinel-wrapped to allow None entries.
        f = self._frequency_compute(k)
        cache[key] = (f,)
        return f

    def _frequency_compute(self, k=None):
        if self.sys == uGNSS.GPS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_G1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_G2
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_G5
            else:
                return None
        elif self.sys == uGNSS.GLO:
            if int(self.sig / 100) == 1 and k is not None:
                return rCST.FREQ_R1 + k * rCST.FREQ_R1k
            elif int(self.sig / 100) == 2 and k is not None:
                return rCST.FREQ_R2 + k * rCST.FREQ_R2k
            elif int(self.sig / 100) == 3:
                return rCST.FREQ_R3
            elif int(self.sig / 100) == 4:
                return rCST.FREQ_R1a
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_R2a
            else:
                return None
        elif self.sys == uGNSS.GAL:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_E1
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_E5a
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_E6
            elif int(self.sig / 100) == 7:
                return rCST.FREQ_E5b
            elif int(self.sig / 100) == 8:
                return rCST.FREQ_E5
            else:
                return None
        elif self.sys == uGNSS.BDS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_C1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_C12
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_C2a
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_C3
            elif int(self.sig / 100) == 7:
                return rCST.FREQ_C2b
            elif int(self.sig / 100) == 8:
                return rCST.FREQ_C2
            else:
                return None
        if self.sys == uGNSS.QZS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_J1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_J2
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_J5
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_J6
            else:
                return None
        if self.sys == uGNSS.SBS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_S1
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_S5
        elif self.sys == uGNSS.IRN:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_I1
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_I5
            elif int(self.sig / 100) == 9:
                return rCST.FREQ_IS
        else:
            return None

    def wavelength(self, k=None):
        """ wavelength in [m] """

        frq = self.frequency(k)
        return rCST.CLIGHT/frq if frq is not None else None
