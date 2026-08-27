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
        defaulting to this engine's rover. Passing the base's state is how
        a caller edits base observations; previously the only way was to
        swap ``self.nav`` out from under the engine.
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
            rr_ = rr

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
        # Slip flags are per-epoch signals too. Only the rover's were
        # ever cleared (rtk.update_ambiguities zeroes nav.slip), so a
        # single base-side LLI/GF slip latched forever and the merge
        # np.maximum(rover.slip, base.slip) reset that satellite's
        # ambiguity on every following epoch.
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

            # Loop over signals
            #
            for f in range(self.nav.nf):

                # Slot not carried by this constellation (mixed nf): treat as
                # absent. Do NOT set edt -- the downstream
                # "np.any(edt[sat,:]>0)" check would otherwise drop the whole
                # satellite (the padded slot is never observed).
                if f >= len(sigsCP):
                    continue

                # Cycle  slip check by LLI
                #
                # LLI=1 is a cycle-slip notification, not a bad observation:
                # flag the band so udstate() resets its ambiguity, but keep
                # the measurement (RTKLIB-style). Editing it out instead
                # leaves nav.slip without a producer, so the reset in
                # udstate() never fires and the stale ambiguity survives the
                # slip.
                if obs.lli[j, f] == 1:
                    rcv.slip[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write("{}  {} - slip {:4s} - LLI\n"
                                            .format(time2str(obs.t),
                                                    sat2id(sat_i),
                                                    sigsCP[f].str()))
                    continue

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

            # Store satellite which have passed all tests, judged over the
            # bands its SYSTEM actually selected (a constellation offering
            # fewer than nav.nf common bands — e.g. GPS L1+L2 in an nf=3
            # setup — is judged on those bands only, so its satellites are
            # not punished for a slot that was never selected). Within the
            # selected bands the classic strict gate applies: any edited
            # band drops the whole satellite — a missing or degraded band
            # on a satellite whose system does provide it is a tracking /
            # multipath canary (admitting L5-less GPS or B1I-only BeiDou-2
            # measurably poisons the urban float solution).
            nf_sys = min(self.nav.nf, len(sigsCP), len(sigsPR))
            if nf_sys <= 0 or np.any(rcv.edt[i, :nf_sys] > 0):
                rcv.edt[i, :] = 1
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)
