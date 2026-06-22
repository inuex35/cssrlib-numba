"""Shared GTSAM-first integer ambiguity-resolution bridge.

`resolve_ar` runs cssrlib's LAMBDA (``resamb_lambda``) on a GTSAM float solution:
it writes the float between-receiver SD ambiguities and their joint covariance
into the cssrlib nav state, then resolves the integers. This is the AR
*algorithm* only -- not the cssrlib EKF -- and is the bridge used by
tightly-coupled-gnss-imu-fgo.

The full position+ambiguity joint marginal is numerically ill-conditioned
(Point3 together with many strongly-correlated ambiguities -> NaN), so the
covariance is assembled from the ambiguity-only joint (stable) plus pairwise
(position, ambiguity) cross terms, with a non-finite guard.
"""
import numpy as np
import gtsam
from cssrlib.gnss import sat2prn


def resolve_ar(engine, isam, res, x_key, amb_key, sats, el, seen_am, nf, syss,
               conv_sigma=None, elmaskar_deg=15.0):
    """Resolve integer ambiguities on the current GTSAM (ISAM2) float estimate.

    Parameters
    ----------
    engine : gnssobs subclass (rtkpos / ppprtkpos) providing nav, IB,
        resamb_lambda.
    isam : gtsam.ISAM2 holding the current Bayes-tree factorization.
    res : gtsam.Values, current estimate (must contain x_key and ambiguities).
    x_key : gtsam Key of the receiver position (Point3).
    amb_key : callable (sat, freq) -> gtsam Key of the SD ambiguity.
    sats, el : satellites and their elevations [rad] at this epoch.
    seen_am : set of ambiguity Keys present in the graph.
    nf : number of frequencies.
    syss : iterable of accepted constellations.
    conv_sigma : if set, skip AR until the position 1-sigma [m] is below it
        (PPP needs convergence; RTK can fix instantaneously -> leave None).
    elmaskar_deg : elevation mask [deg] for AR.

    Returns
    -------
    (nb, fixed_xyz) : number of fixed SD ambiguities and the fixed ECEF
        position, or (0, None) if AR was skipped / not accepted.
    """
    nav = engine.nav
    nav.x[nav.na:] = 0.0
    nav.P[:, :] = 0.0
    nav.vsat[:, :] = 0
    nav.x[0:3] = np.array(res.atPoint3(x_key))

    amb = [(int(s), f) for s in sats for f in range(nf)
           if sat2prn(int(s))[0] in syss and amb_key(int(s), f) in seen_am
           and res.exists(amb_key(int(s), f))]
    if len(amb) < 4:
        return 0, None

    p_pos = isam.marginalCovariance(x_key)
    if conv_sigma is not None and np.sqrt(np.trace(p_pos)) > conv_sigma:
        return 0, None

    el_now = {int(s): el[i] for i, s in enumerate(sats)}
    for (s_, f) in amb:
        j = engine.IB(s_, f, nav.na)
        nav.x[j] = res.atDouble(amb_key(s_, f))
        nav.vsat[s_ - 1, f] = 1
        if s_ in el_now:
            nav.el[s_ - 1] = el_now[s_]

    nav.P[0:3, 0:3] = p_pos
    kv = gtsam.KeyVector()
    for (s_, f) in amb:
        kv.append(amb_key(s_, f))
    jm = isam.jointMarginalCovariance(kv)
    for (s_, f) in amb:
        j = engine.IB(s_, f, nav.na)
        nav.P[j, j] = jm.at(amb_key(s_, f), amb_key(s_, f))[0, 0]
        kvx = gtsam.KeyVector()
        kvx.append(x_key)
        kvx.append(amb_key(s_, f))
        pxn = isam.jointMarginalCovariance(kvx).at(x_key, amb_key(s_, f))[:, 0]
        nav.P[0:3, j] = pxn
        nav.P[j, 0:3] = pxn
    for a in range(len(amb)):
        s1, f1 = amb[a]
        j1 = engine.IB(s1, f1, nav.na)
        for b in range(a + 1, len(amb)):
            s2, f2 = amb[b]
            j2 = engine.IB(s2, f2, nav.na)
            c = jm.at(amb_key(s1, f1), amb_key(s2, f2))[0, 0]
            nav.P[j1, j2] = c
            nav.P[j2, j1] = c

    bad = ~np.isfinite(nav.P)
    if bad.any():
        nav.P[bad] = 0.0
        d = np.where(np.diag(bad))[0]
        nav.P[d, d] = 1e10

    nav.elmaskar = np.deg2rad(elmaskar_deg)
    sat_ar = np.array(sorted({s_ for (s_, f) in amb}))
    nb, _ = engine.resamb_lambda(sat_ar, nav.parmode, nav.par_P0)
    return nb, (np.array(nav.xa[0:3]) if nb > 0 else None)
