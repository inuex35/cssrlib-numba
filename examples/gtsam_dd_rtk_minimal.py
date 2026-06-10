"""
Minimal GTSAM double-difference RTK using the cssrlib observation model.

Distilled from inuex35/tightly-coupled-gnss-imu-fgo down to the essential
GNSS part. The full project couples IMU pre-integration and uses custom C++
DD factors compiled into its own GTSAM fork; here the DD factors are defined
in pure Python via ``gtsam.CustomFactor``, so this runs on a stock
``pip install gtsam`` (4.2.x) with no custom build.

What cssrlib provides (the whole GNSS front-end):
  rtk.prepare_double_difference_measurements(obs, obsb, ...)
      -> satellite ECEF positions (rs), common-sat indices (iu),
         single-difference rover-base observations (obs_sd), elevations (el).

What this file adds (the GTSAM part):
  * two CustomFactors -- DD pseudorange and DD carrier-phase,
  * per-constellation reference-satellite selection (highest elevation),
  * a single static rover position + per-satellite float ambiguities,
  * incremental ISAM2 estimation.

Assumptions kept minimal: static rover, single frequency (L1/E1), float
ambiguities (no integer AR). Extending to kinematic (one pose per epoch +
a motion/IMU factor) or to fixed ambiguities (cssrlib.mlambda) is additive.
"""

import os
import sys

import numpy as np
import gtsam

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import rSigRnx, uGNSS, uTYP, sat2prn, sat2id
from cssrlib.rtk import rtkpos

# Position is a single static unknown shared by every epoch.
X = gtsam.symbol('x', 0)


def _amb_key(sat, gen):
    """Ambiguity variable key for a satellite, versioned by generation.

    The generation counter is bumped on a cycle slip / reference change so a
    fresh ambiguity is estimated instead of reusing a stale one.
    """
    return gtsam.symbol('n', sat * 1000 + gen)


def _dd_geometry(rr, rs_ref, rs_j, rb):
    """Double-difference geometric range and its Jacobian w.r.t. rr.

    pred = (|rs_j - rr| - |rs_ref - rr|) - (|rs_j - rb| - |rs_ref - rb|)

    Satellite clock and (short-baseline) atmosphere cancel in the DD, so the
    only unknown left in the range is the rover position rr.
    """
    d_ref = rs_ref - rr
    d_j = rs_j - rr
    r_ref = np.linalg.norm(d_ref)
    r_j = np.linalg.norm(d_j)
    r_ref_b = np.linalg.norm(rs_ref - rb)
    r_j_b = np.linalg.norm(rs_j - rb)
    pred = (r_j - r_ref) - (r_j_b - r_ref_b)
    # d|rs - rr|/drr = -(rs - rr)/|rs - rr| = -u  (u: unit rr->sat)
    u_ref = d_ref / r_ref
    u_j = d_j / r_j
    dpred_drr = u_ref - u_j          # d(r_j - r_ref)/drr
    return pred, dpred_drr


def _make_dd_pr_factor(noise, key_x, measured, rs_ref, rs_j, rb):
    """DD pseudorange factor: residual = pred - measured, unknown = rr."""
    def error(this, values, jac):
        rr = values.atPoint3(key_x)
        pred, g = _dd_geometry(rr, rs_ref, rs_j, rb)
        if jac is not None:
            jac[0] = g.reshape(1, 3)
        return np.array([pred - measured])
    return gtsam.CustomFactor(noise, gtsam.KeyVector([key_x]), error)


def _make_dd_cp_factor(noise, key_x, key_n, measured_m, lam, rs_ref, rs_j, rb):
    """DD carrier-phase factor: residual = pred + lam*N - measured_m.

    Unknowns: rover position rr and the (float) DD ambiguity N [cycles].
    """
    def error(this, values, jac):
        rr = values.atPoint3(key_x)
        n = values.atVector(key_n)[0]
        pred, g = _dd_geometry(rr, rs_ref, rs_j, rb)
        if jac is not None:
            jac[0] = g.reshape(1, 3)
            jac[1] = np.array([[lam]])
        return np.array([pred + lam * n - measured_m])
    return gtsam.CustomFactor(noise, gtsam.KeyVector([key_x, key_n]), error)


def main():
    bdir = os.path.join(os.path.dirname(__file__),
                        '..', 'src', 'cssrlib', 'data') + os.sep
    navfile = bdir + 'SEPT078M.21P'
    obsfile = bdir + 'SEPT078M1.21O'
    basefile = bdir + '3034078M1.21O'

    xyz_ref = np.array([-3962108.673, 3381309.574, 3668678.638])  # true rover
    pos_ref = gn.ecef2pos(xyz_ref)

    # GPS L1/L2 + Galileo E1/E5 (DD is built on L1/E1 only, column f=0).
    sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("GL1C"), rSigRnx("GL2W"),
            rSigRnx("GS1C"), rSigRnx("GS2W"),
            rSigRnx("EC1C"), rSigRnx("EC5Q"), rSigRnx("EL1C"), rSigRnx("EL5Q"),
            rSigRnx("ES1C"), rSigRnx("ES5Q")]
    sigsb = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("GL1C"), rSigRnx("GL2W"),
             rSigRnx("GS1C"), rSigRnx("GS2W"),
             rSigRnx("EC1X"), rSigRnx("EC5X"), rSigRnx("EL1X"), rSigRnx("EL5X"),
             rSigRnx("ES1X"), rSigRnx("ES5X")]

    dec = rn.rnxdec()
    dec.setSignals(sigs)
    nav = gn.Nav()
    dec.decode_nav(navfile, nav)
    decb = rn.rnxdec()
    decb.setSignals(sigsb)
    decb.decode_obsh(basefile)
    dec.decode_obsh(obsfile)

    nav.rb = [-3959400.631, 3385704.533, 3667523.111]   # base ECEF (known)
    rb = np.array(nav.rb)
    rtk = rtkpos(nav, dec.pos)

    # ISAM2 with the rover position anchored by a loose prior.
    isam = gtsam.ISAM2()
    g0 = gtsam.NonlinearFactorGraph()
    v0 = gtsam.Values()
    v0.insert(X, gtsam.Point3(*dec.pos))
    g0.add(gtsam.PriorFactorPoint3(
        X, gtsam.Point3(*dec.pos),
        gtsam.noiseModel.Isotropic.Sigma(3, 30.0)))
    isam.update(g0, v0)

    ref_sat = {}        # sys -> reference satellite (fixed once it is chosen)
    amb_gen = {}        # sat -> ambiguity generation (bumped on slip/ref reset)
    have_n = set()      # ambiguity keys already inserted into ISAM2

    sync = rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)
    nep = 60
    for ne, (obs, obsb, dt) in enumerate(sync):
        if ne >= nep:
            break
        if obsb is None:
            continue

        rr_est = isam.calculateEstimate().atPoint3(X)
        dd = rtk.prepare_double_difference_measurements(
            obs, obsb, pos_pred=rr_est, dd_only=True, compute_zdres=True)
        if dd is None:
            continue

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        # Group common satellites by constellation.
        by_sys = {}
        for i, sat in enumerate(dd.sat):
            if dd.obs_sd.P[i, 0] == 0.0 or dd.obs_sd.L[i, 0] == 0.0:
                continue
            by_sys.setdefault(sat2prn(int(sat))[0], []).append(i)

        for sys, idxs in by_sys.items():
            if len(idxs) < 2:
                continue
            lam = obs.sig[sys][uTYP.L][0].wavelength()

            # Reference satellite: highest elevation, fixed across epochs.
            cur_ref_idx = max(idxs, key=lambda k: dd.el[k])
            ref = ref_sat.get(sys)
            ref_idx = next((k for k in idxs if int(dd.sat[k]) == ref), None)
            if ref_idx is None:                    # ref absent -> re-pick, reset
                ref = int(dd.sat[cur_ref_idx])
                ref_sat[sys] = ref
                ref_idx = cur_ref_idx
                for k in idxs:
                    amb_gen[int(dd.sat[k])] = amb_gen.get(int(dd.sat[k]), 0) + 1

            rs_ref = dd.rs[dd.iu[ref_idx], :3]
            sd_p_ref = dd.obs_sd.P[ref_idx, 0]
            sd_l_ref = dd.obs_sd.L[ref_idx, 0]
            el_ref = dd.el[ref_idx]

            for j in idxs:
                if j == ref_idx:
                    continue
                j_sat = int(dd.sat[j])

                # Cycle-slip on either end -> fresh ambiguity generation.
                if rtk.nav.slip[j_sat - 1, 0] or rtk.nav.slip[ref - 1, 0]:
                    amb_gen[j_sat] = amb_gen.get(j_sat, 0) + 1

                rs_j = dd.rs[dd.iu[j], :3]
                dd_p = dd.obs_sd.P[j, 0] - sd_p_ref
                dd_l_m = lam * (dd.obs_sd.L[j, 0] - sd_l_ref)

                # Elevation-weighted DD noise (rough but standard).
                s = 1.0 / max(np.sin(min(dd.el[j], el_ref)), 0.1)
                noise_pr = gtsam.noiseModel.Isotropic.Sigma(1, 0.3 * s)
                noise_cp = gtsam.noiseModel.Isotropic.Sigma(1, 0.01 * s)

                graph.add(_make_dd_pr_factor(
                    noise_pr, X, dd_p, rs_ref, rs_j, rb))

                gen = amb_gen.get(j_sat, 0)
                key_n = _amb_key(j_sat, gen)
                if key_n not in have_n:
                    n0 = (dd_l_m - dd_p) / lam          # carrier - code init
                    values.insert(key_n, np.array([n0]))
                    have_n.add(key_n)
                graph.add(_make_dd_cp_factor(
                    noise_cp, X, key_n, dd_l_m, lam, rs_ref, rs_j, rb))

        rtk.nav.slip[:] = 0   # slip flags consumed (we own the state, not udstate)
        if graph.size() == 0:
            continue

        isam.update(graph, values)
        rr = isam.calculateEstimate().atPoint3(X)
        enu = gn.ecef2enu(pos_ref, rr - xyz_ref)
        print("ep {:2d}  nsat={:2d}  ENU err [m]: E{:+.3f} N{:+.3f} U{:+.3f}  "
              "|h|={:.3f}".format(ne, len(dd.sat), enu[0], enu[1], enu[2],
                                  np.hypot(enu[0], enu[1])))

    dec.fobs.close()
    decb.fobs.close()


if __name__ == "__main__":
    sys.exit(main())
