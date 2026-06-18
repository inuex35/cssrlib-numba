"""GTSAM double-difference RTK using cssrlib as the observation front-end.

Architecture (gtsam-first), the DD counterpart of examples/gtsam_ppprtk.py:
  cssrlib -> rtkpos.prepare_double_difference_measurements(obs, obsb):
             rover/base satellite states, common-satellite indices and
             elevations. No EKF.
  GTSAM   -> DoubleDifferencePseudorangeFactor / DoubleDifferenceCarrierPhaseFactor
             (C++, from the inuex35/gtsam fork) form the rover-base double
             difference internally (Sagnac-corrected geodist) and estimate a
             single static rover position + per-satellite float ambiguities,
             updated incrementally with ISAM2.

Static rover, single frequency (L1/E1), float ambiguities. Runs on the data
bundled with cssrlib (src/cssrlib/data) and the custom gtsam build.
"""
import os
import numpy as np

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import rSigRnx, uTYP, sat2prn
from cssrlib.rtk import rtkpos
import gtsam
from gtsam import symbol

X = symbol('x', 0)


def _amb(sat, gen):
    return symbol('n', sat * 100 + gen)


def main():
    bdir = os.path.join(os.path.dirname(__file__),
                        '..', 'src', 'cssrlib', 'data') + os.sep
    navfile = bdir + 'SEPT078M.21P'
    obsfile = bdir + 'SEPT078M1.21O'
    basefile = bdir + '3034078M1.21O'
    xyz_ref = np.array([-3962108.673, 3381309.574, 3668678.638])
    pos_ref = gn.ecef2pos(xyz_ref)

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
    nav.rb = [-3959400.631, 3385704.533, 3667523.111]
    rb = np.array(nav.rb)
    rtk = rtkpos(nav, dec.pos)

    isam = gtsam.ISAM2()
    g0 = gtsam.NonlinearFactorGraph()
    v0 = gtsam.Values()
    v0.insert(X, gtsam.Point3(*dec.pos))
    g0.add(gtsam.PriorFactorPoint3(
        X, gtsam.Point3(*dec.pos), gtsam.noiseModel.Isotropic.Sigma(3, 30.0)))
    isam.update(g0, v0)

    ref_sat, amb_gen, have_n = {}, {}, set()
    sync = rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)
    nep = 60
    for ne, (obs, obsb, dt) in enumerate(sync):
        if ne >= nep:
            break
        if obsb is None:
            continue
        rr_est = isam.calculateEstimate().atPoint3(X)
        dd = rtk.prepare_double_difference_measurements(obs, obsb,
                                                        pos_pred=rr_est)
        if dd is None:
            continue
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        by_sys = {}
        for k, sat in enumerate(dd.sat):
            if obs.P[dd.iu[k], 0] == 0 or obsb.P[dd.ir[k], 0] == 0 \
               or obs.L[dd.iu[k], 0] == 0 or obsb.L[dd.ir[k], 0] == 0:
                continue
            by_sys.setdefault(sat2prn(int(sat))[0], []).append(k)

        for sys, ks in by_sys.items():
            if len(ks) < 2:
                continue
            lam = obs.sig[sys][uTYP.L][0].wavelength()
            ref = ref_sat.get(sys)
            ridx = next((k for k in ks if int(dd.sat[k]) == ref), None)
            if ridx is None:
                ridx = max(ks, key=lambda k: dd.el[k])
                ref = int(dd.sat[ridx])
                ref_sat[sys] = ref
                for k in ks:
                    amb_gen[int(dd.sat[k])] = amb_gen.get(int(dd.sat[k]), 0) + 1

            rs_ref = dd.rs[dd.iu[ridx], :3]
            rsb_ref = dd.rsb[dd.ir[ridx], :3]
            pr_rr, pr_br = obs.P[dd.iu[ridx], 0], obsb.P[dd.ir[ridx], 0]
            cp_rr = obs.L[dd.iu[ridx], 0]*lam
            cp_br = obsb.L[dd.ir[ridx], 0]*lam
            for k in ks:
                if k == ridx:
                    continue
                js = int(dd.sat[k])
                rs_j = dd.rs[dd.iu[k], :3]
                rsb_j = dd.rsb[dd.ir[k], :3]
                s = 1.0/max(np.sin(min(dd.el[k], dd.el[ridx])), 0.1)

                graph.add(gtsam.DoubleDifferencePseudorangeFactor(
                    X, pr_rr, pr_br, obs.P[dd.iu[k], 0], obsb.P[dd.ir[k], 0],
                    gtsam.Point3(*rs_ref), gtsam.Point3(*rs_j),
                    gtsam.Point3(*rsb_ref), gtsam.Point3(*rsb_j),
                    gtsam.Point3(*rb),
                    gtsam.noiseModel.Isotropic.Sigma(1, 0.3*s)))

                gen = amb_gen.get(js, 0)
                gref = amb_gen.get(ref, 0)
                kref, kj = _amb(ref, gref), _amb(js, gen)
                # Reference ambiguity is the per-system datum: pin it (gauge).
                if kref not in have_n:
                    values.insert(kref, 0.0)
                    graph.addPriorDouble(
                        kref, 0.0, gtsam.noiseModel.Isotropic.Sigma(1, 1e-3))
                    have_n.add(kref)
                # Target ambiguity: init from the DD carrier-minus-code.
                if kj not in have_n:
                    dd_pr = (pr_rr - pr_br) - (obs.P[dd.iu[k], 0]
                                               - obsb.P[dd.ir[k], 0])
                    dd_cp = (cp_rr - cp_br) - (obs.L[dd.iu[k], 0]*lam
                                              - obsb.L[dd.ir[k], 0]*lam)
                    values.insert(kj, float((dd_cp - dd_pr)/lam))
                    graph.addPriorDouble(
                        kj, float((dd_cp - dd_pr)/lam),
                        gtsam.noiseModel.Isotropic.Sigma(1, 1e2))
                    have_n.add(kj)
                graph.add(gtsam.DoubleDifferenceCarrierPhaseFactor(
                    X, kref, kj, cp_rr, cp_br,
                    obs.L[dd.iu[k], 0]*lam, obsb.L[dd.ir[k], 0]*lam,
                    gtsam.Point3(*rs_ref), gtsam.Point3(*rs_j),
                    gtsam.Point3(*rsb_ref), gtsam.Point3(*rsb_j),
                    gtsam.Point3(*rb), lam,
                    gtsam.noiseModel.Isotropic.Sigma(1, 0.01*s)))

        if graph.size() == 0:
            continue
        isam.update(graph, values)
        rr = isam.calculateEstimate().atPoint3(X)
        enu = gn.ecef2enu(pos_ref, np.array(rr) - xyz_ref)
        print("ep {:2d}  nsat={:2d}  ENU [m]: E{:+.3f} N{:+.3f} U{:+.3f}  "
              "|h|={:.3f}".format(ne, len(dd.sat), enu[0], enu[1], enu[2],
                                  np.hypot(enu[0], enu[1])))


if __name__ == "__main__":
    main()
