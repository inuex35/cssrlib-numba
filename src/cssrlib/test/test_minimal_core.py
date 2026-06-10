"""
Lock in the minimal dependency surface of the broadcast-ephemeris RTK path.

The SSR/PPP machinery (cssrlib.cssrlib, cssrlib.peph, cssrlib.ppp) is imported
lazily, so a plain ``import cssrlib.rtk`` -- and the double-difference RTK
workflow used by external estimators (GTSAM) -- must not pull those heavier
modules. These checks run in a fresh subprocess so the measurement is not
polluted by modules other tests already imported.
"""

import subprocess
import sys
import textwrap


HEAVY = ("cssrlib.ppp", "cssrlib.peph", "cssrlib.cssrlib")


def _run(code):
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_import_rtk_is_lean():
    """import cssrlib.rtk must not load the SSR/PPP/antenna modules."""
    code = """
        import sys
        import cssrlib.rtk            # noqa: F401
        heavy = [m for m in ('cssrlib.ppp', 'cssrlib.peph', 'cssrlib.cssrlib')
                 if m in sys.modules]
        print('HEAVY:' + ','.join(heavy))
    """
    assert _run(code) == "HEAVY:"


def test_dd_only_path_is_lean():
    """The DD-only external RTK workflow must not load SSR/PPP/antenna code."""
    code = """
        import os, sys
        import numpy as np
        import cssrlib.rinex as rn
        import cssrlib.gnss as gn
        from cssrlib.rtk import rtkpos
        from cssrlib.gnss import rSigRnx

        d = os.path.join(os.path.dirname(rn.__file__), 'data') + os.sep
        sigs = [rSigRnx('GC1C'), rSigRnx('GC2W'), rSigRnx('GL1C'),
                rSigRnx('GL2W'), rSigRnx('GS1C'), rSigRnx('GS2W')]
        dec = rn.rnxdec(); dec.setSignals(sigs)
        nav = gn.Nav(); dec.decode_nav(d + 'SEPT078M.21P', nav)
        decb = rn.rnxdec(); decb.setSignals(sigs)
        decb.decode_obsh(d + '3034078M1.21O'); dec.decode_obsh(d + 'SEPT078M1.21O')
        nav.rb = [-3959400.631, 3385704.533, 3667523.111]
        rtk = rtkpos(nav, dec.pos)
        sync = rn.sync_obs_hold(dec, decb, maxage=nav.maxtdiff)
        for k, (obs, obsb, dt) in enumerate(sync):
            if k >= 5:
                break
            if k == 0:
                nav.t = obs.t
            dd = rtk.prepare_double_difference_measurements(
                obs, obsb, pos_pred=nav.x[0:3].copy(),
                dd_only=True, compute_zdres=False)
            if dd is not None:
                rtk.manage_ambiguities_external(dd.obs_sd)
        dec.fobs.close(); decb.fobs.close()
        heavy = [m for m in ('cssrlib.ppp', 'cssrlib.peph', 'cssrlib.cssrlib')
                 if m in sys.modules]
        print('HEAVY:' + ','.join(heavy))
    """
    assert _run(code) == "HEAVY:"


def test_utidemodel_reexport():
    """uTideModel moved to gnss; ppp re-exports the same object."""
    from cssrlib.gnss import uTideModel as a
    from cssrlib.ppp import uTideModel as b
    assert a is b
    assert int(a.IERS2010) == 1 and int(a.NONE) == -1


if __name__ == "__main__":
    test_import_rtk_is_lean()
    test_dd_only_path_is_lean()
    test_utidemodel_reexport()
    print("OK")
