"""
Lock in the minimal module set of the broadcast-ephemeris RTK core.

The SSR/CSSR (cssrlib.cssrlib), antenna-model (cssrlib.peph) and Earth-tide /
phase-wind-up (cssrlib.ppp) modules were removed; this checks they are gone
and that the double-difference RTK workflow used by external estimators
(GTSAM) needs only the lightweight remaining modules.
"""

import importlib

import pytest

REMOVED = ("cssrlib.ppp", "cssrlib.peph", "cssrlib.cssrlib")
CORE = ("cssrlib.gnss", "cssrlib.rinex", "cssrlib.ephemeris", "cssrlib.orbit",
        "cssrlib.glonass", "cssrlib.mlambda", "cssrlib.geometry",
        "cssrlib.pppssr", "cssrlib.rtk", "cssrlib.atmosphere",
        "cssrlib.constants")


@pytest.mark.parametrize("mod", REMOVED)
def test_heavy_modules_removed(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


@pytest.mark.parametrize("mod", CORE)
def test_core_modules_import(mod):
    assert importlib.import_module(mod) is not None


def test_utidemodel_in_gnss():
    """uTideModel now lives in gnss (it left ppp with the minimal core)."""
    from cssrlib.gnss import uTideModel
    assert int(uTideModel.IERS2010) == 1 and int(uTideModel.NONE) == -1


if __name__ == "__main__":
    for m in REMOVED:
        try:
            importlib.import_module(m)
            raise SystemExit(f"{m} should have been removed")
        except ModuleNotFoundError:
            pass
    for m in CORE:
        importlib.import_module(m)
    test_utidemodel_in_gnss()
    print("OK")
