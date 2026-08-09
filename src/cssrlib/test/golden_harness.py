"""Deterministic numerical fingerprint of the positioning pipeline.

Not a test module (pytest does not collect it) -- it builds the arrays that
``test_golden.py`` compares against the committed reference.

Why this exists: commit 20c0df1 reverted 20 Numba kernels and broke four
code paths without anything noticing, because the unit-test workflow fired
only on branches that were dead or absent. A committed numerical reference
turns "the merge quietly changed the maths" into a red test.

Everything here is driven by the bundled 2021 doy-078 dataset (SEPT rover /
3034 base) and fixed seeds, so repeated runs on any machine agree bit for
bit. Regenerate the reference with:

    python -m cssrlib.test.golden_harness
"""

import os

import numpy as np

import cssrlib.gnss as gn
import cssrlib.rinex as rn
from cssrlib.ephemeris import satposs
from cssrlib.gnss import Obs, rSigRnx, uGNSS, uTYP, prn2sat
from cssrlib.rtk import rtkpos

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "data") + os.sep
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden",
                      "pipeline.npz")

OBSFILE = DATA + "SEPT078M1.21O"
BASEFILE = DATA + "3034078M1.21O"
NAVFILE = DATA + "SEPT078M.21P"

# Base station ECEF, as used by the GTSAM examples on this dataset.
BASE_ECEF = [-3959400.631, 3385704.533, 3667523.111]

SIGS = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("GS1C"), rSigRnx("GS2W")]

N_DD_EPOCH = 25       # double-difference path
N_MODEL_EPOCH = 20    # zdres / sdres / ddidx / ddcov

# Signals per constellation for the synthetic multi-GNSS cases. The bundled
# dataset is GPS-only, so nothing in it exercises sdres's per-system
# reference-satellite loop.
SYNTH_SIGS = {
    uGNSS.GPS: ("GC1C", "GC2W", "GL1C", "GL2W"),
    uGNSS.GAL: ("EC1C", "EC5Q", "EL1C", "EL5Q"),
    uGNSS.BDS: ("CC2I", "CC6I", "CL2I", "CL6I"),
}
SYNTH_CASES = [
    ((uGNSS.GPS,), 8),
    ((uGNSS.GPS, uGNSS.GAL), 6),
    ((uGNSS.GPS, uGNSS.GAL, uGNSS.BDS), 5),
    ((uGNSS.GAL, uGNSS.BDS), 7),
]


def setup():
    """A configured rtkpos plus its rover/base RINEX decoders."""
    dec = rn.rnxdec()
    dec.setSignals(SIGS)
    nav = gn.Nav()
    dec.decode_nav(NAVFILE, nav)

    decb = rn.rnxdec()
    decb.setSignals(list(SIGS))
    decb.decode_obsh(BASEFILE)
    dec.decode_obsh(OBSFILE)

    nav.rb = list(BASE_ECEF)
    return dec, decb, rtkpos(nav, dec.pos)


def _store(out, key, value):
    out[key] = np.zeros(0) if value is None else np.asarray(value,
                                                            dtype=np.float64)


def _dd_path(out):
    """satposs -> qcedit -> single differences, over real epochs."""
    dec, decb, rtk = setup()
    n = 0
    for ep in range(N_DD_EPOCH):
        obs = dec.decode_obs()
        obsb = decb.decode_obs()
        if obs is None or obs.sat is None or len(obs.sat) == 0:
            break
        if obsb is None or obsb.sat is None or len(obsb.sat) == 0:
            break

        dd = rtk.prepare_double_difference_measurements(obs, obsb)
        if dd is None:
            continue
        for name in ("rs", "dts", "iu", "sat", "el"):
            _store(out, f"dd.{name}#{ep}", dd[name])
        _store(out, f"dd.sdL#{ep}", dd["obs_sd"].L)
        _store(out, f"dd.sdP#{ep}", dd["obs_sd"].P)
        n += 1
    _store(out, "dd.epochs", [n])


def _observation_model(out):
    """zdres / sdres / ddidx / ddcov, replaying what the PPP driver does."""
    dec, decb, rtk = setup()
    nf = rtk.nav.nf

    for ep in range(N_MODEL_EPOCH):
        obs = dec.decode_obs()
        obsb = decb.decode_obs()
        if obs is None or len(obs.sat) == 0:
            break
        if obsb is None or len(obsb.sat) == 0:
            break

        rs, vs, dts, svh, nsat = satposs(obs, rtk.nav)
        if nsat < 6:
            continue

        rtk.qcedit(obs, rs, dts, svh)
        iu, obs_ = rtk.single_differences(obs, obsb, rs, dts, svh)
        ns = len(iu)
        if ns < 6:
            continue

        rtk.udstate(obs_)
        xp = rtk.nav.x.copy()

        yu, eu, elu = rtk.zdres(obs, None, None, rs, vs, dts, xp[0:3])
        _store(out, f"zdres.y#{ep}", yu)
        _store(out, f"zdres.e#{ep}", eu)
        _store(out, f"zdres.el#{ep}", elu)

        sat = obs.sat[iu]
        y = np.zeros((ns, nf * 2))
        e = np.zeros((ns, 3))
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]
        rtk.nav.sat = sat
        rtk.nav.el[sat - 1] = el

        v, H, R = rtk.sdres(obs, xp, y, e, sat, el)
        _store(out, f"sdres.v#{ep}", v)
        _store(out, f"sdres.H#{ep}", H)
        _store(out, f"sdres.R#{ep}", R)

        # ddidx needs satellites marked valid; the EKF normally does this
        # inside its measurement update.
        rtk.nav.vsat[sat - 1, :] = 1
        _store(out, f"ddidx.ix#{ep}", rtk.ddidx(rtk.nav, sat))
        _store(out, f"ddidx.fix#{ep}", rtk.nav.fix)

        rng = np.random.default_rng(1000 + ep)
        nb = np.array([3, 4, 2], dtype=np.int64)
        nv = int(nb.sum())
        _store(out, f"ddcov.R#{ep}",
               rtk.ddcov(nb, len(nb), rng.uniform(0.1, 2.0, nv),
                         rng.uniform(0.1, 2.0, nv), nv))


def _synthetic_obs(seed, systems, nsat_per_sys, nf):
    rng = np.random.default_rng(seed)
    obs = Obs()
    sats, sig = [], {}
    for sysid in systems:
        codes = SYNTH_SIGS[sysid]
        sig[sysid] = {uTYP.C: [rSigRnx(c) for c in codes[:2][:nf]],
                      uTYP.L: [rSigRnx(c) for c in codes[2:][:nf]]}
        for prn in range(1, nsat_per_sys + 1):
            sats.append(prn2sat(sysid, prn))
    obs.sat = np.array(sats, dtype=int)
    obs.sig = sig

    ns = len(sats)
    y = rng.normal(scale=3.0, size=(ns, nf * 2))
    e = rng.normal(size=(ns, 3))
    e /= np.linalg.norm(e, axis=1)[:, None]
    el = rng.uniform(0.15, 1.5, ns)
    return obs, obs.sat.copy(), y, e, el


def _multignss_sdres(out):
    """sdres over GPS / Galileo / BeiDou mixes the real dataset cannot reach."""
    _, _, rtk = setup()
    nav = rtk.nav
    nf = nav.nf

    for ci, (systems, nps) in enumerate(SYNTH_CASES):
        for seed in range(3):
            obs, sat, y, e, el = _synthetic_obs(100 * ci + seed, systems,
                                                nps, nf)
            rng = np.random.default_rng(9000 + 100 * ci + seed)
            x = np.zeros(nav.nx)
            x[0:3] = [-3962108.7, 3381309.5, 3668678.6]
            for s in sat:
                for f in range(nf):
                    x[rtk.IB(s, f, nav.na)] = rng.normal(scale=1e6)

            nav.x = x.copy()
            nav.el[sat - 1] = el
            nav.vsat[:] = 0
            obs.t = nav.t

            v, H, R = rtk.sdres(obs, x, y, e, sat, el)
            tag = f"{ci}.{seed}"
            _store(out, f"synth.v#{tag}", v)
            _store(out, f"synth.H#{tag}", H)
            _store(out, f"synth.R#{tag}", R)
            _store(out, f"synth.vsat#{tag}", nav.vsat)


def build_golden():
    """Every recorded array, keyed by ``stage.field#index``."""
    out = {}
    _dd_path(out)
    _observation_model(out)
    _multignss_sdres(out)
    return out


def write_golden(path=GOLDEN):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = build_golden()
    np.savez_compressed(path, **data)
    return path, len(data)


if __name__ == "__main__":
    p, n = write_golden()
    print(f"wrote {n} arrays to {p} "
          f"({os.path.getsize(p) / 1024:.1f} KB)")
