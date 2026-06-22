"""GTSAM double-difference RTK with integer ambiguity resolution.

Architecture (gtsam-first), the DD counterpart of examples/gtsam_ppprtk.py:
  cssrlib -> rtkpos.prepare_double_difference_measurements(obs, obsb):
             rover/base satellite states, common-satellite indices, elevations.
  GTSAM   -> DoubleDifference{Pseudorange,CarrierPhase}Factor (C++, inuex35/gtsam
             fork) form the rover-base double difference internally
             (Sagnac-corrected geodist) for a single static rover position +
             per-satellite, per-frequency float ambiguities. Batch LM.
  AR      -> the float (between-receiver SD) ambiguities + their joint marginal
             covariance (incl. position cross-covariance) are written into the
             cssrlib state and resolved with resamb_lambda (LAMBDA + ddidx SD +
             ratio test) -- the AR *algorithm* only, not the cssrlib EKF.

All available frequencies are used (GPS L1/L2, Galileo E1/E5a). Static rover.
Runs on the data bundled with cssrlib (src/cssrlib/data).
"""
import os
import numpy as np

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import rSigRnx, uTYP, sat2prn
from cssrlib.rtk import rtkpos
import gtsam
from gtsam import symbol

from gnss_ar import resolve_ar

X = symbol('x', 0)                       # static rover ECEF position
SYSS = (gn.uGNSS.GPS, gn.uGNSS.GAL)


def AM(sat, f):                          # between-receiver SD ambiguity node
    return symbol('n', int(sat) * 10 + f)


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
    nav.x[0:3] = np.array(dec.pos)   # seed position so qcedit computes elevations
    nf = nav.nf

    # ---- collect per-epoch DD measurements (front-end) ----------------------
    frames = []
    sync = rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)
    nep = 60
    for ne, (obs, obsb, dt) in enumerate(sync):
        if ne >= nep:
            break
        if obsb is None:
            continue
        dd = rtk.prepare_double_difference_measurements(obs, obsb,
                                                        pos_pred=dec.pos)
        if dd is not None:
            frames.append((obs, obsb, dd))
    print(f"frames: {len(frames)}")

    # ---- per-system reference satellite = max cumulative elevation ----------
    el_cum = {}
    for (_, _, dd) in frames:
        for k, s in enumerate(dd.sat):
            if dd.el[k] > 0:
                el_cum[int(s)] = el_cum.get(int(s), 0.0) + dd.el[k]
    ref_of = {}
    for s, e in el_cum.items():
        sys = sat2prn(s)[0]
        if sys in SYSS and (sys not in ref_of or e > el_cum[ref_of[sys]]):
            ref_of[sys] = s

    # ---- incremental ISAM2 (QR): one update per epoch, like ------------------
    # tightly-coupled-gnss-imu-fgo. The incrementally-built Bayes tree is
    # better conditioned than a one-shot batch and its jointMarginalCovariance
    # is robust.
    params = gtsam.ISAM2Params()
    params.setFactorization('QR')
    isam = gtsam.ISAM2(params)
    seen_am, pinned = set(), set()

    def report(tag, xh):
        enu = gn.ecef2enu(pos_ref, xh - xyz_ref)
        print(f"{tag}: E{enu[0]:+.3f} N{enu[1]:+.3f} U{enu[2]:+.3f}  "
              f"2D={np.hypot(enu[0], enu[1]):.3f}  "
              f"3D={np.linalg.norm(xh - xyz_ref):.3f} m")

    nfac = 0
    first_fix = None
    n_fix = 0
    for ei, (obs, obsb, dd) in enumerate(frames):
        graph = gtsam.NonlinearFactorGraph()
        val = gtsam.Values()
        if ei == 0:
            val.insert(X, gtsam.Point3(*dec.pos))
            graph.add(gtsam.PriorFactorPoint3(
                X, gtsam.Point3(*dec.pos),
                gtsam.noiseModel.Isotropic.Sigma(3, 30.0)))

        def ensure_amb(sat, f, init):
            if AM(sat, f) not in seen_am:
                val.insert(AM(sat, f), float(init))
                seen_am.add(AM(sat, f))

        by_sys = {}
        for k, s in enumerate(dd.sat):
            by_sys.setdefault(sat2prn(int(s))[0], []).append(k)
        for sys, ks in by_sys.items():
            ref = ref_of.get(sys)
            ridx = next((k for k in ks if int(dd.sat[k]) == ref), None)
            if ridx is None:
                continue
            for f in range(nf):
                lam = obs.sig[sys][uTYP.L][f].wavelength()
                pr_rr, pr_br = obs.P[dd.iu[ridx], f], obsb.P[dd.ir[ridx], f]
                cp_rr = obs.L[dd.iu[ridx], f] * lam
                cp_br = obsb.L[dd.ir[ridx], f] * lam
                if 0.0 in (pr_rr, pr_br, cp_rr, cp_br):
                    continue
                rs_ref = dd.rs[dd.iu[ridx], :3]
                rsb_ref = dd.rsb[dd.ir[ridx], :3]
                # Between-receiver SD ambiguities (cycles): gauge is the
                # reference SD, pinned to its carrier-minus-code value. Any
                # value fixes the rank deficiency (DDs are gauge-independent)
                # and keeps nav.x[IB] non-zero so resamb_lambda's ddidx (which
                # skips x==0) can use the reference as the pivot.
                sd_ref = ((cp_rr - cp_br) - (pr_rr - pr_br)) / lam
                ensure_amb(ref, f, sd_ref)
                if (ref, f) not in pinned:
                    graph.addPriorDouble(
                        AM(ref, f), sd_ref,
                        gtsam.noiseModel.Isotropic.Sigma(1, 0.5))
                    pinned.add((ref, f))
                for k in ks:
                    js = int(dd.sat[k])
                    if k == ridx:
                        continue
                    pr_tr, pr_tb = obs.P[dd.iu[k], f], obsb.P[dd.ir[k], f]
                    cp_tr = obs.L[dd.iu[k], f] * lam
                    cp_tb = obsb.L[dd.ir[k], f] * lam
                    if 0.0 in (pr_tr, pr_tb, cp_tr, cp_tb):
                        continue
                    rs_j = dd.rs[dd.iu[k], :3]
                    rsb_j = dd.rsb[dd.ir[k], :3]
                    s = 1.0 / max(np.sin(min(dd.el[k], dd.el[ridx])), 0.1)

                    graph.add(gtsam.DoubleDifferencePseudorangeFactor(
                        X, pr_rr, pr_br, pr_tr, pr_tb,
                        gtsam.Point3(*rs_ref), gtsam.Point3(*rs_j),
                        gtsam.Point3(*rsb_ref), gtsam.Point3(*rsb_j),
                        gtsam.Point3(*rb),
                        gtsam.noiseModel.Isotropic.Sigma(1, 0.3 * s)))

                    sd_tgt = ((cp_tr - cp_tb) - (pr_tr - pr_tb)) / lam
                    ensure_amb(js, f, sd_tgt)
                    graph.add(gtsam.DoubleDifferenceCarrierPhaseFactor(
                        X, AM(ref, f), AM(js, f), cp_rr, cp_br, cp_tr, cp_tb,
                        gtsam.Point3(*rs_ref), gtsam.Point3(*rs_j),
                        gtsam.Point3(*rsb_ref), gtsam.Point3(*rsb_j),
                        gtsam.Point3(*rb), lam,
                        gtsam.noiseModel.Isotropic.Sigma(1, 0.01 * s)))
        nfac += graph.size()
        isam.update(graph, val)

        res = isam.calculateEstimate()
        nb, xa = resolve_ar(rtk, isam, res, X, AM, dd.sat, dd.el,
                            seen_am, nf, SYSS)
        xh = xa if nb > 0 else np.array(res.atPoint3(X))
        enu = gn.ecef2enu(pos_ref, xh - xyz_ref)
        mode = 'FIX  ' if nb > 0 else 'float'
        if nb > 0:
            n_fix += 1
            if first_fix is None:
                first_fix = ei
        print(f"ep{ei:3d} {mode} nb={nb:2d} 2D={np.hypot(enu[0], enu[1]):.3f} "
              f"3D={np.linalg.norm(xh - xyz_ref):.3f} m")

    print(f"\nupdates: {len(frames)} epochs, {nfac} factors, "
          f"{len(seen_am)} ambiguities")
    print(f"first fix: epoch {first_fix}   fixed {n_fix}/{len(frames)} epochs")


if __name__ == "__main__":
    main()
