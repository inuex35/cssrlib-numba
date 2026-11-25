"""Observation quality control.

Editing decisions -- elevation mask, health, C/N0, cycle slips -- recorded
into the ReceiverState of whichever receiver is being edited."""

import numpy as np

from cssrlib.gnss import sat2id, sat2prn, uTYP, uGNSS
from cssrlib.gnss import ecef2pos, geodist, satazel
from cssrlib.gnss import time2str, timediff, gpst2utc
from cssrlib.models.tides import tidedisp, tidedispIERS2010, uTideModel


class QualityControlMixin:
    """Quality control, mixed into :class:`~cssrlib.engine.gnssobs.gnssobs`."""

    def _log_qc(self, msg: str):
        if self.nav.monlevel > 0:
            self.nav.fout.write(msg)

    def _predict_receiver_ecef(self, obs, rr):
        """Predict receiver ECEF and apply solid Earth tide displacement."""
        tt = timediff(obs.t, self.nav.t)
        if rr is None:
            rr_ = self.nav.x[0:3].copy()
            if self.nav.pmode > 0:
                rr_ += self.nav.x[3:6] * tt
        else:
            # A copy: tide displacement adds in place without mutating caller's array.
            rr_ = np.array(rr, dtype=float)

        if self.nav.tidecorr == uTideModel.SIMPLE:
            disp = tidedisp(gpst2utc(obs.t), ecef2pos(rr_))
        elif self.nav.tidecorr == uTideModel.IERS2010:
            disp = tidedispIERS2010(gpst2utc(obs.t), ecef2pos(rr_))
        else:
            disp = np.zeros(3)
        return rr_ + disp

    def _check_bands(self, obs, rcv, i, j, sat_i, sigsPR, sigsCP, sigsCN):
        """Run per-band LLI slip, zero-obs, and C/N0 checks."""
        for f in range(self.nav.nf):
            # Slot not carried by this constellation (mixed nf): mark edited.
            if f >= len(sigsCP):
                rcv.edt[i, f] = 1
                continue

            # LLI=1 flags the band for ambiguity reset but keeps the measurement.
            if obs.lli[j, f] & 1:
                rcv.slip[i, f] = 1
                self._log_qc("{}  {} - slip {:4s} - LLI\n".format(
                    time2str(obs.t), sat2id(sat_i), sigsCP[f].str()))

            if obs.P[j, f] == 0.0:
                rcv.edt[i, f] = 1
                self._log_qc("{}  {} - edit {:4s} - invalid PR obs\n".format(
                    time2str(obs.t), sat2id(sat_i), sigsPR[f].str()))
                continue

            if obs.L[j, f] == 0.0:
                rcv.edt[i, f] = 1
                self._log_qc("{}  {} - edit {:4s} - invalid CP obs\n".format(
                    time2str(obs.t), sat2id(sat_i), sigsCP[f].str()))
                continue

            cnr_min = (self.nav.cnr_min_gpy
                       if sigsCN[f].isGPS_PY() else self.nav.cnr_min)
            if obs.S[j, f] < cnr_min:
                rcv.edt[i, f] = 1
                self._log_qc("{}  {} - edit {:4s} - low C/N0 {:4.1f} dB-Hz\n".format(
                    time2str(obs.t), sat2id(sat_i), sigsCN[f].str(), obs.S[j, f]))

    def _check_gf_slip(self, obs, rcv, i, j, sat_i, sigsCP):
        """Geometry-free L1-L2 cycle-slip detector."""
        if obs.L.shape[1] <= 1 or len(sigsCP) <= 1:
            return
        L1R, L2R = obs.L[j, 0:2]
        sys_i, _ = sat2prn(sat_i)
        sig1, sig2 = sigsCP[0:2]
        if sys_i == uGNSS.GLO:
            ch = self.nav.glo_ch.get(sat_i)
            lam1 = sig1.wavelength(ch) if ch is not None else 0.0
            lam2 = sig2.wavelength(ch) if ch is not None else 0.0
        else:
            lam1 = sig1.wavelength()
            lam2 = sig2.wavelength()
        if L1R == 0.0 or L2R == 0.0:
            return
        gf1 = L1R * lam1 - L2R * lam2
        gf0 = rcv.gf[sat_i - 1]
        if gf1 != 0.0:
            rcv.gf[sat_i - 1] = gf1
        if gf0 != 0.0 and gf1 != 0.0 and abs(gf1 - gf0) > self.nav.thresslip:
            rcv.slip[i, 0:2] = 1
            self._log_qc(
                " {}  {} - slip {:4s} - GF slip gf0 {:6.3f} gf1 {:6.3f} gf0-gf1 {:6.3f} \n"
                .format(time2str(obs.t), sat2id(sat_i), sig1.str(),
                        gf0, gf1, gf0 - gf1))

    def _edit_single_satellite(self, obs, rs, dts, svh, rr_, pos, rcv, i):
        """Evaluate one satellite slot; return ``sat_i`` if admitted else ``None``."""
        sat_i = i + 1
        sys_i, _ = sat2prn(sat_i)

        if sat_i not in obs.sat:
            rcv.edt[i, :] = 1
            return None

        if sat_i in self.nav.excl_sat:
            rcv.edt[i, :] = 1
            self._log_qc("{}  {} - edit - satellite excluded\n".format(
                time2str(obs.t), sat2id(sat_i)))
            return None

        j = np.where(obs.sat == sat_i)[0][0]
        if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
            rcv.edt[i, :] = 1
            self._log_qc("{}  {} - edit - invalid eph\n".format(
                time2str(obs.t), sat2id(sat_i)))
            return None

        if svh[j] > 0:
            rcv.edt[i, :] = 1
            self._log_qc("{}  {} - edit - satellite unhealthy\n".format(
                time2str(obs.t), sat2id(sat_i)))
            return None

        _, e = geodist(rs[j, :], rr_)
        _, el = satazel(pos, e)
        rcv.el[sat_i - 1] = el
        if el < self.nav.elmin:
            rcv.edt[i][:] = 1
            self._log_qc("{}  {} - edit - low elevation {:5.1f} deg\n".format(
                time2str(obs.t), sat2id(sat_i), np.rad2deg(el)))
            return None

        sigsPR = obs.sig[sys_i][uTYP.C]
        sigsCP = obs.sig[sys_i][uTYP.L]
        sigsCN = obs.sig[sys_i][uTYP.S]

        for f in range(min(self.nav.nf, obs.L.shape[1])):
            if obs.L[j, f] != 0.0 and obs.P[j, f] != 0.0:
                rcv.band_seen[sat_i - 1, f] = True

        self._check_bands(obs, rcv, i, j, sat_i, sigsPR, sigsCP, sigsCN)
        self._check_gf_slip(obs, rcv, i, j, sat_i, sigsCP)

        nf_sys = min(self.nav.nf, len(sigsCP), len(sigsPR))
        if nf_sys <= 0:
            rcv.edt[i, :] = 1
            return None
        judged = np.ones(nf_sys, dtype=bool)
        if self.nav.sat_band_plan:
            judged = rcv.band_seen[sat_i - 1, :nf_sys].copy()
            rcv.edt[i, :nf_sys][~judged] = 1
        if not judged.any() or np.any(rcv.edt[i, :nf_sys][judged] > 0):
            rcv.edt[i, :] = 1
            return None

        return sat_i

    def qcedit(self, obs, rs, dts, svh, rr=None, rcv=None):
        """Coarse quality control and editing of observations.

        ``rcv`` is the :class:`ReceiverState` to record the results in,
        defaulting to this engine's rover; pass the base's state to edit
        base observations.
        """
        rcv = self.nav.rcv if rcv is None else rcv
        rr_ = self._predict_receiver_ecef(obs, rr)
        pos = ecef2pos(rr_)

        ns = uGNSS.MAXSAT
        rcv.edt = np.zeros((ns, self.nav.nf), dtype=int)
        rcv.slip[:, :] = 0

        sat = []
        for i in range(ns):
            admitted = self._edit_single_satellite(
                obs, rs, dts, svh, rr_, pos, rcv, i)
            if admitted is not None:
                sat.append(admitted)
        return np.array(sat, dtype=int)
