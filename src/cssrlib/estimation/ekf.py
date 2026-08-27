"""The extended Kalman filter and the PPP / PPP-RTK driver.

Owns the time update, the measurement update and the per-epoch sequencing.
RTK is not driven from here -- see rtkpos.prepare_double_difference_measurements."""

import numpy as np

from cssrlib.models.ephemeris import satposs
from cssrlib.gnss import sat2id, sat2prn, uTYP, uGNSS, rCST
from cssrlib.gnss import time2str, timediff


class FilterMixin:
    """Filtering and the epoch driver, mixed into :class:`~cssrlib.engine.gnssobs.gnssobs`."""

    def udstate(self, obs):
        """ time propagation of states and initialize """

        tt = timediff(obs.t, self.nav.t)

        ns = len(obs.sat)
        sys = []
        sat = obs.sat
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel,ztd,ion,amb
        #
        nx = self.nav.nx
        Phi = np.eye(nx)
        # if self.nav.niono > 0:
        #    ni = self.nav.na-uGNSS.MAXSAT
        #    Phi[ni:self.nav.na, ni:self.nav.na] = np.zeros(
        #        (uGNSS.MAXSAT, uGNSS.MAXSAT))
        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
        self.nav.P[0:nx, 0:nx] = Phi@self.nav.P[0:nx, 0:nx]@Phi.T

        # Process noise
        #
        dP = np.einsum('ii->i', self.nav.P)  # writable diagonal view
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*abs(tt)

        # Update Kalman filter state elements
        #
        for f in range(self.nav.nf):

            # Reset phase-ambiguity if instantaneous AR
            # or expire obs outage counter
            #
            for i in range(uGNSS.MAXSAT):

                sat_ = i+1
                sys_i, _ = sat2prn(sat_)

                self.nav.outc[i, f] += 1
                # Reset the ambiguity on outage, edit or cycle slip
                # (nav.slip, per band; cleared below once applied).
                reset = (self.nav.outc[i, f] > self.nav.maxout
                         or self.nav.edt[i, f] > 0
                         or self.nav.slip[i, f] > 0)
                if sys_i not in obs.sig.keys():
                    continue

                if f >= self.nsig_sys(obs, sys_i):  # slot not carried (mixed nf)
                    continue

                # Reset ambiguity estimate
                #
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0
                    self.nav.slip[i, f] = 0

                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - reset ambiguity  {}\n"
                            .format(time2str(obs.t), sat2id(sat_),
                                    obs.sig[sys_i][uTYP.L][f]))

                if self.nav.niono > 0:
                    # Reset slant ionospheric delay estimate
                    #
                    j = self.II(sat_, self.nav.na)
                    if reset and self.nav.x[j] != 0.0:
                        self.initx(0.0, 0.0, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write("{}  {} - reset ionosphere\n"
                                                .format(time2str(obs.t),
                                                        sat2id(sat_)))

            # Ambiguity
            #
            bias = np.zeros(ns)
            ion = np.zeros(ns)
            f1 = 0

            """
            offset = 0
            na = 0
            """
            for i in range(ns):

                # Do not initialize invalid observations
                #
                # Per band: editing is per band now, so a satellite kept for
                # its good band must still have that band initialized.
                if self.nav.edt[sat[i]-1, f] > 0:
                    continue

                if f >= self.nsig_sys(obs, sys[i]):  # slot not carried (mixed nf)
                    continue

                if self.nav.nf > 1 and self.nav.niono > 0:
                    # Get dual-frequency pseudoranges for this constellation
                    #
                    sig1 = obs.sig[sys[i]][uTYP.C][0]
                    sig2 = obs.sig[sys[i]][uTYP.C][1]

                    pr1 = obs.P[i, 0]
                    pr2 = obs.P[i, 1]

                    # Skip zero observations
                    #
                    if pr1 == 0.0 or pr2 == 0.0:
                        continue

                    if sys[i] == uGNSS.GLO:
                        if sat[i] not in self.nav.glo_ch:
                            print("glonass channel not found: {:d}"
                                  .format(sat[i]))
                            continue
                        f1 = sig1.frequency(self.nav.glo_ch[sat[i]])
                        f2 = sig2.frequency(self.nav.glo_ch[sat[i]])
                    else:
                        f1 = sig1.frequency()
                        f2 = sig2.frequency()

                    # Get iono delay at frequency of first signal
                    #
                    ion[i] = (pr1-pr2)/(1.0-(f1/f2)**2)

                # Get pseudorange and carrier-phase observation of signal f
                #
                sig = obs.sig[sys[i]][uTYP.L][f]

                if sys[i] == uGNSS.GLO:
                    fi = sig.frequency(self.nav.glo_ch[sat[i]])
                else:
                    fi = sig.frequency()

                lam = rCST.CLIGHT/fi

                cp = obs.L[i, f]
                pr = obs.P[i, f]
                if cp == 0.0 or pr == 0.0 or lam is None:
                    continue

                bias[i] = cp - pr/lam + 2.0*ion[i]/lam*(f1/fi)**2

                """
                amb = nav.x[IB(sat[i], f, nav.na)]
                if amb != 0.0:
                    offset += bias[i] - amb
                    na += 1
                """
            """
            # Adjust phase-code coherency
            #
            if na > 0:
                db = offset/na
                for i in range(uGNSS.MAXSAT):
                    if nav.x[IB(i+1, f, nav.na)] != 0.0:
                        nav.x[IB(i+1, f, nav.na)] += db
            """

            # Initialize ambiguity
            #
            for i in range(ns):

                sys_i, _ = sat2prn(sat[i])

                j = self.IB(sat[i], f, self.nav.na)
                if bias[i] != 0.0 and self.nav.x[j] == 0.0:

                    self.initx(bias[i], self.nav.sig_n0**2, j)

                    if self.nav.monlevel > 0:
                        sig = obs.sig[sys_i][uTYP.L][f]
                        self.nav.fout.write(
                            "{}  {} - init  ambiguity  {} {:12.3f}\n"
                            .format(time2str(obs.t), sat2id(sat[i]),
                                    sig, bias[i]))

                if self.nav.niono > 0:
                    j = self.II(sat[i], self.nav.na)
                    if ion[i] != 0 and self.nav.x[j] == 0.0:

                        self.initx(ion[i], self.nav.sig_ion0**2, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write(
                                "{}  {} - init  ionosphere      {:12.3f}\n"
                                .format(time2str(obs.t), sat2id(sat[i]),
                                        ion[i]))

        return 0

    def kfupdate(self, x, P, H, v, R):
        """
        Kalman filter measurement update.

        Parameters:
        x (ndarray): State estimate vector
        P (ndarray): State covariance matrix
        H (ndarray): Observation model matrix
        v (ndarray): Innovation vector
                     (residual between measurement and prediction)
        R (ndarray): Measurement noise covariance

        Returns:
        x (ndarray): Updated state estimate vector
        P (ndarray): Updated state covariance matrix
        S (ndarray): Innovation covariance matrix
        """

        # Update only states with initialized covariance: for the rest
        # P's row and column are zero, so their gain is zero and they
        # contribute nothing to S. Reduces the Joseph update from nx
        # (hundreds, mostly never-initialized ambiguities) to the
        # states actually in use.
        act = np.flatnonzero(np.diag(P) > 0.0)
        if act.size < P.shape[0]:
            H_ = H[:, act]
            P_ = P[np.ix_(act, act)]
            PHt = P_@H_.T
            S = H_@PHt+R
            K = PHt@np.linalg.inv(S)
            x[act] += K@v
            IKH = np.eye(act.size)-K@H_
            P[np.ix_(act, act)] = IKH@P_@IKH.T + K@R@K.T
            return x, P, S

        PHt = P@H.T
        S = H@PHt+R
        K = PHt@np.linalg.inv(S)
        x += K@v
        # P = P - K@H@P
        IKH = np.eye(P.shape[0])-K@H
        P = IKH@P@IKH.T + K@R@K.T  # Joseph stabilized version

        return x, P, S

    def valpos(self, v, R, thres=4.0):
        """Post-fit residual MONITOR — always returns True.

        Logs offending residuals (monlevel > 1) without vetoing the
        solution; callers treating the return as a gate get a
        constant-True gate.
        """
        nv = len(v)
        fact = thres**2
        for i in range(nv):
            if v[i]**2 <= fact*R[i, i]:
                continue
            if self.nav.monlevel > 1:
                txt = "{:3d} is large: {:8.4f} ({:8.4f})".format(
                    i, v[i], R[i, i])
                if self.nav.fout is None:
                    print(txt)
                else:
                    self.nav.fout.write(txt+"\n")
        return True

    def initx(self, x0, v0, i):
        """ initialize x and P for index i """
        self.nav.x[i] = x0
        for j in range(self.nav.nx):
            self.nav.P[j, i] = self.nav.P[i, j] = v0 if i == j else 0

    def _prepare_sat_states(self, obs, cs=None, orb=None, pos_pred=None,
                            rs=None, vs=None, dts=None, svh=None):
        """Shared GTSAM front-end helper: satellite states + min-sat count +
        linearisation position.

        Common to the double-difference (rtkpos) and PPP-RTK (ppprtkpos)
        ``prepare_*_measurements`` front-ends. Runs ``satposs`` (applying SSR
        corrections when ``cs``/``orb`` are given) unless pre-computed states
        are passed, and defaults ``pos_pred`` to the current estimate.

        Returns ``(rs, vs, dts, svh, nsat, pos_pred)``.
        """
        if rs is None or vs is None or dts is None or svh is None:
            rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        else:
            nsat = int(np.count_nonzero(~np.isnan(dts)))
        if pos_pred is None:
            pos_pred = self.nav.x[0:3].copy()
        return rs, vs, dts, svh, nsat, np.asarray(pos_pred, dtype=float)

    def process(self, obs, cs=None, orb=None, bsx=None):
        """
        PPP/PPP-RTK positioning

        RTK is not driven from here. The EKF's rover-minus-base residuals
        were removed with the minimal core, leaving this method's old
        ``obsb`` branch without a base ``zdres`` to difference against; use
        ``rtkpos.prepare_double_difference_measurements`` instead, which is
        what the GTSAM examples do.
        """

        # Skip empty epochs
        #
        if len(obs.sat) == 0:
            return

        self.nav.nsat[0] = len(obs.sat)

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        self.nav.nsat[1] = nsat

        if nsat < 6:
            print(" too few satellites < 6: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        iu = np.where(np.isin(obs.sat, sat_ed))[0]
        obs_ = obs

        # y / e are filled from zdres below.
        ns = len(iu)
        y = np.zeros((ns, self.nav.nf*2))
        e = np.zeros((ns, 3))

        self.nav.nsat[2] = ns

        if ns < 6:
            print(" too few satellites < 6: ns={:d}".format(ns))
            return

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs_)

        xa = np.zeros(self.nav.nx)
        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 6 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        # Non-differential residuals after measurement update
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        y = yu[iu, :]
        e = eu[iu, :]
        ny = y.shape[0]
        if ny < 6:
            return -1

        # Residuals for float solution
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        if self.valpos(v, R):
            self.nav.x = xp
            self.nav.P = Pp
            self.nav.ns = 0
            for i in range(ns):
                j = sat[i]-1
                for f in range(self.nav.nf):
                    if self.nav.vsat[j, f] == 0:
                        continue
                    self.nav.outc[j, f] = 0
                    if f == 0:
                        self.nav.ns += 1
            self.nav.smode = 5   # 4: fixed, 5: float
        else:
            # do not overwrite a valpos reject with float status
            self.nav.smode = 0

        if self.nav.armode > 0:
            res = self.resolve_ambiguities(sat)
            nb, xa = res.nb, res.xa
            if nb > 0:
                # Use position with fixed ambiguities xa
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[0:3])
                y = yu[iu, :]
                e = eu[iu, :]
                v, H, R = self.sdres(obs, xa, y, e, sat, el)
                # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
                if self.valpos(v, R):
                    if self.nav.armode == 3:     # fix and hold
                        self.holdamb(xa)    # hold fixed ambiguity
                    self.nav.smode = 4           # fix
                else:
                    pass
            else:
                pass

        # Store epoch for solution
        #
        self.nav.t = obs.t

        return 0
