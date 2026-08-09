"""
module for PPP-RTK positioning
"""

import numpy as np
from cssrlib.gnssobs import gnssobs
from cssrlib.config import ppprtk_config
from cssrlib.gnss import ecef2pos, sat2prn, uTYP, tropmapf


class PPPMeasurements(dict):
    """Return value of :meth:`ppprtkpos.prepare_ppp_measurements`.

    A ``dict`` with attribute access (``ppp.rs`` as well as ``ppp['rs']``),
    mirroring :class:`cssrlib.rtk.DDMeasurements`. It bundles everything an
    external estimator (e.g. a GTSAM factor graph using the
    ``Undifferenced{Pseudorange,CarrierPhase}Factor`` family) needs to build
    undifferenced PPP-RTK factors, without running the internal EKF.

    Fields
    ------
    rs, vs, dts, svh : np.ndarray
        CLAS-corrected satellite ECEF position [m], velocity [m/s], clock
        offset [s] and health, aligned with ``sat``.
    sat : np.ndarray of int
        Satellite numbers (``obs.sat``).
    el : np.ndarray of float
        Elevation angles [rad] at ``pos_pred``.
    y : np.ndarray, shape (n, 2*nf)
        Undifferenced residuals from :meth:`gnssobs.zdres` with all SSR/CLAS
        corrections applied: columns ``0..nf-1`` carrier phase, ``nf..2nf-1``
        code, in metres. The position-independent corrected observation for a
        factor is ``m = y[i, col] + geodist(rs[i], pos_pred)``.
    e : np.ndarray, shape (n, 3)
        Line-of-sight unit vectors (receiver -> satellite).
    mapfw : np.ndarray, shape (n,)
        Tropospheric wet mapping function per satellite (for the ZTD factor).
    mu : np.ndarray, shape (n, nf)
        First-order ionospheric coefficient (f1/ff)^2 per satellite/frequency.
    lam : np.ndarray, shape (n, nf)
        Carrier wavelength [m] per satellite/frequency.
    iono_sig : np.ndarray, shape (n,)
        CLAS STEC a-priori 1-sigma [m] per satellite (slant-iono prior;
        residual iono is estimated about 0). NaN when unavailable.
    ztd_sig : float
        CLAS tropospheric a-priori 1-sigma [m] (ZTD residual prior). NaN when
        unavailable.
    pos_pred : np.ndarray
        Receiver position used to linearise the geometry / compute ``y`` [m].
    """

    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class ppprtkpos(gnssobs):
    """ class for PPP-RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, cfg=None):
        """ initialize variables for PPP-RTK

        Everything that used to distinguish this class from rtkpos is now in
        :func:`cssrlib.config.ppprtk_config`; pass ``cfg`` to adjust it.
        """
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         cfg=ppprtk_config(nf=nav.nf, pmode=nav.pmode)
                         if cfg is None else cfg)

    def prepare_ppp_measurements(self, obs, cs=None, bsx=None, pos_pred=None,
                                 rs=None, vs=None, dts=None, svh=None):
        """Prepare undifferenced PPP-RTK observations for a GTSAM factor graph.

        Computes the building blocks the undifferenced PPP factors need --
        CLAS-corrected satellite states, corrected residuals (``zdres``),
        tropo mapping / iono coefficients / wavelengths, and the CLAS
        atmosphere a-priori sigmas -- *without* running the EKF. Mirrors
        :meth:`cssrlib.rtk.rtkpos.prepare_double_difference_measurements` and
        shares :meth:`gnssobs._prepare_sat_states`.

        Returns
        -------
        PPPMeasurements or None
            ``None`` when there are too few satellites.
        """
        if len(obs.sat) == 0:
            return None

        rs, vs, dts, svh, nsat, pos_pred = self._prepare_sat_states(
            obs, cs=cs, pos_pred=pos_pred, rs=rs, vs=vs, dts=dts, svh=svh)
        if nsat < 4:
            return None

        # Undifferenced residuals with all SSR/CLAS corrections applied.
        y, e, el = self.zdres(obs, cs, bsx, rs, vs, dts, pos_pred)

        sat = np.asarray(obs.sat)
        n = len(sat)
        nf = self.nav.nf
        pos = ecef2pos(pos_pred)

        mapfw = np.zeros(n)
        mu = np.zeros((n, nf))
        lam = np.zeros((n, nf))
        for i, s in enumerate(sat):
            if el[i] <= 0.0:
                continue
            _, mapfw[i] = tropmapf(obs.t, pos, el[i], model=self.nav.trpModel)
            sys = sat2prn(int(s))[0]
            sigL = obs.sig[sys][uTYP.L]
            f0 = sigL[0].frequency()
            for f in range(min(nf, len(sigL))):
                ff = sigL[f].frequency()
                if ff:
                    mu[i, f] = (f0 / ff) ** 2
                lam[i, f] = sigL[f].wavelength() or 0.0

        # CLAS atmosphere a-priori sigmas (residual estimated about 0).
        iono_sig = np.full(n, np.nan)
        ztd_sig = np.nan
        if cs is not None:
            inet = cs.find_grid_index(pos)
            lc = cs.lc[inet] if 0 <= inet < len(cs.lc) else None
            if lc is not None:
                sq = getattr(lc, 'stec_quality', None) or {}
                for i, s in enumerate(sat):
                    q = sq.get(int(s))
                    if q is not None and np.isfinite(q) and q > 0:
                        iono_sig[i] = float(q)
                tq = getattr(lc, 'trop_quality', None)
                if tq is not None and np.isfinite(tq) and tq > 0:
                    ztd_sig = float(tq)

        return PPPMeasurements({
            'rs': rs, 'vs': vs, 'dts': dts, 'svh': svh,
            'sat': sat, 'el': el, 'y': y, 'e': e,
            'mapfw': mapfw, 'mu': mu, 'lam': lam,
            'iono_sig': iono_sig, 'ztd_sig': ztd_sig,
            'pos_pred': pos_pred,
        })
