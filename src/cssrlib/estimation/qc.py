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

    def qcedit(self, obs, rs, dts, svh, rr=None, rcv=None):
        """ Coarse quality control and editing of observations

        ``rcv`` is the :class:`ReceiverState` to record the results in,
        defaulting to this engine's rover; pass the base's state to edit
        base observations.
        """
        rcv = self.nav.rcv if rcv is None else rcv

        # Predicted position at next epoch
        #
        tt = timediff(obs.t, self.nav.t)
        if rr is None:
            rr_ = self.nav.x[0:3].copy()
            if self.nav.pmode > 0:
                rr_ += self.nav.x[3:6]*tt
        else:
            # A copy: the tide correction below adds to rr_ in place
            # and must not write through to the caller's array.
            rr_ = np.array(rr, dtype=float)

        # Solid Earth tide corrections
        #
        if self.nav.tidecorr == uTideModel.SIMPLE:
            pos = ecef2pos(rr_)
            disp = tidedisp(gpst2utc(obs.t), pos)
        elif self.nav.tidecorr == uTideModel.IERS2010:
            pos = ecef2pos(rr_)
            disp = tidedispIERS2010(gpst2utc(obs.t), pos)
        else:
            disp = np.zeros(3)
        rr_ += disp

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        # Total number of satellites
        #
        ns = uGNSS.MAXSAT

        # Reset previous editing results
        #
        rcv.edt = np.zeros((ns, self.nav.nf), dtype=int)
        # Slip flags are per-epoch signals: clear before re-detecting.
        rcv.slip[:, :] = 0

        # Loop over all satellites
        #
        sat = []
        for i in range(ns):

            sat_i = i+1
            sys_i, _ = sat2prn(sat_i)

            if sat_i not in obs.sat:
                rcv.edt[i, :] = 1
                continue

            # Check satellite exclusion
            #
            if sat_i in self.nav.excl_sat:
                rcv.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite excluded\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            j = np.where(obs.sat == sat_i)[0][0]

            # Check for valid orbit and clock offset
            #
            if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
                rcv.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - invalid eph\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check satellite health
            #
            if svh[j] > 0:
                rcv.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite unhealthy\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check elevation angle
            #
            _, e = geodist(rs[j, :], rr_)
            _, el = satazel(pos, e)
            rcv.el[sat_i - 1] = el  # persist for weighting / AR elev mask
            if el < self.nav.elmin:
                rcv.edt[i][:] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write(
                        "{}  {} - edit - low elevation {:5.1f} deg\n"
                        .format(time2str(obs.t), sat2id(sat_i),
                                np.rad2deg(el)))
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR = obs.sig[sys_i][uTYP.C]
            sigsCP = obs.sig[sys_i][uTYP.L]
            sigsCN = obs.sig[sys_i][uTYP.S]

            # Record which bands this satellite demonstrably transmits
            # before any editing: L and P both present is transmission
            # evidence regardless of what the quality tests make of it.
            for f in range(min(self.nav.nf, obs.L.shape[1])):
                if obs.L[j, f] != 0.0 and obs.P[j, f] != 0.0:
                    rcv.band_seen[sat_i - 1, f] = True

            # Loop over signals
            #
            for f in range(self.nav.nf):

                # Slot not carried by this constellation (mixed nf):
                # mark it edited so per-band consumers skip it.
                if f >= len(sigsCP):
                    rcv.edt[i, f] = 1
                    continue

                # Cycle  slip check by LLI
                #
                # LLI=1 flags the band for the ambiguity reset in
                # udstate() but keeps the measurement (RTKLIB-style);
                # the validity checks below still apply to it.
                if obs.lli[j, f] & 1:  # b0 = cycle slip
                    rcv.slip[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write("{}  {} - slip {:4s} - LLI\n"
                                            .format(time2str(obs.t),
                                                    sat2id(sat_i),
                                                    sigsCP[f].str()))

                # Check for measurement consistency
                #
                if obs.P[j, f] == 0.0:
                    rcv.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid PR obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsPR[f].str()))
                    continue

                if obs.L[j, f] == 0.0:
                    rcv.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid CP obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCP[f].str()))
                    continue

                # Check C/N0
                #
                cnr_min = self.nav.cnr_min_gpy \
                    if sigsCN[f].isGPS_PY() else self.nav.cnr_min
                if obs.S[j, f] < cnr_min:
                    rcv.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - low C/N0 {:4.1f} dB-Hz\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCN[f].str(),
                                    obs.S[j, f]))
                    continue

            # cycle-slip detection by geometry-free combination
            # obs.L is nf wide for every system, so the array width alone
            # does not mean this constellation selected two bands: a
            # single-band system (e.g. GLONASS L1 only in an nf=2 setup)
            # has just one entry in sigsCP and cannot form a GF combination.
            if obs.L.shape[1] > 1 and len(sigsCP) > 1:
                L1R, L2R = obs.L[j, 0:2]
                sys, _ = sat2prn(sat_i)
                sig1, sig2 = sigsCP[0:2]
                if sys == uGNSS.GLO:
                    # FDMA channel may be unknown (no GLO eph decoded);
                    # lam=0 keeps the GF test a no-op instead of KeyError.
                    ch = self.nav.glo_ch.get(sat_i)
                    lam1 = sig1.wavelength(ch) if ch is not None else 0.0
                    lam2 = sig2.wavelength(ch) if ch is not None else 0.0
                else:
                    lam1 = sig1.wavelength()
                    lam2 = sig2.wavelength()
                if L1R != 0.0 and L2R != 0.0:
                    gf1 = (L1R*lam1-L2R*lam2)
                    # Previously nav.gf for the rover and nav.gf_r for the
                    # base, selected by whether rr was passed. Each receiver
                    # now carries its own table, so there is one name.
                    gf0 = rcv.gf[sat_i - 1]
                    if gf1 != 0.0:
                        rcv.gf[sat_i - 1] = gf1
                    if gf0 != 0.0 and gf1 != 0.0 and \
                            abs(gf1-gf0) > self.nav.thresslip:
                        # A GF jump is a cycle slip, not a bad range: flag
                        # both bands for an ambiguity reset and keep them.
                        # Editing them out drops the whole satellite through
                        # the strict all-band gate below, and again leaves
                        # nav.slip unset.
                        rcv.slip[i, 0:2] = 1
                        if self.nav.monlevel > 0:
                            self.nav.fout.write(" {}  {} - slip {:4s} - GF slip gf0 {:6.3f} gf1 {:6.3f} gf0-gf1 {:6.3f} \n"
                                                .format(time2str(obs.t),
                                                        sat2id(sat_i),
                                                        sig1.str(), gf0, gf1,
                                                        gf0-gf1))

            # Admit a satellite only when every judged band passes.
            # The judgment set is the bands its system selected (a
            # mixed-nf system is judged on the bands it carries); with
            # sat_band_plan it narrows further to the bands this
            # satellite has ever produced. Within that set the gate is
            # strict: one edited band drops the satellite. The per-band
            # edt verdicts above still feed the per-band consumers.
            nf_sys = min(self.nav.nf, len(sigsCP), len(sigsPR))
            if nf_sys <= 0:
                rcv.edt[i, :] = 1
                continue
            judged = np.ones(nf_sys, dtype=bool)
            if self.nav.sat_band_plan:
                # Judge over the bands this satellite transmits (see
                # ProcConfig.sat_band_plan). A band it has never produced
                # is not a failure, exactly as a band its system never
                # selected is not one; within the transmitted set the
                # strict gate below applies unchanged.
                judged = rcv.band_seen[sat_i - 1, :nf_sys].copy()
                rcv.edt[i, :nf_sys][~judged] = 1   # unusable, not failed
            if not judged.any() or np.any(rcv.edt[i, :nf_sys][judged] > 0):
                rcv.edt[i, :] = 1
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)
