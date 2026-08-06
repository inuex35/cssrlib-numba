"""
Lock in the module set of this branch.

History: a minimal broadcast-ephemeris RTK core once removed the SSR/CSSR
(cssrlib.cssrlib), antenna-model (cssrlib.peph) and Earth-tide / phase-wind-up
(cssrlib.ppp) modules and named the engine module ``pppssr``. Commit 20c0df1
("port full PPP-RTK (CLAS) onto the minimal core") brought those modules back
and renamed ``pppssr`` to ``gnssobs``, but this test was left asserting the
old layout and failed. It now describes what the tree actually contains.
"""

import importlib

import pytest

# Lightweight modules the double-difference RTK workflow needs.
CORE = ("cssrlib.gnss", "cssrlib.rinex", "cssrlib.ephemeris", "cssrlib.orbit",
        "cssrlib.glonass", "cssrlib.mlambda", "cssrlib.geometry",
        "cssrlib.gnssobs", "cssrlib.rtk", "cssrlib.atmosphere",
        "cssrlib.constants")

# Modules the CLAS PPP-RTK port re-introduced on top of the minimal core.
PPPRTK = ("cssrlib.ppp", "cssrlib.peph", "cssrlib.cssrlib", "cssrlib.ppprtk",
          "cssrlib.cssr_bds", "cssrlib.cssr_has", "cssrlib.cssr_mdc",
          "cssrlib.cssr_pvs")

# Modules that stayed removed (recover from the `dev` branch if needed).
# cssrlib.pppssr is here because it was renamed, not deleted -- importing it
# is the mistake this file used to make.
REMOVED = ("cssrlib.rtcm", "cssrlib.sbas", "cssrlib.osnma", "cssrlib.qznma",
           "cssrlib.ewss", "cssrlib.dgps", "cssrlib.pntpos", "cssrlib.rawnav",
           "cssrlib.ionosphere", "cssrlib.tlesim", "cssrlib.plot",
           "cssrlib.utils", "cssrlib.pppssr")


@pytest.mark.parametrize("mod", CORE)
def test_core_modules_import(mod):
    assert importlib.import_module(mod) is not None


@pytest.mark.parametrize("mod", PPPRTK)
def test_ppprtk_modules_import(mod):
    assert importlib.import_module(mod) is not None


@pytest.mark.parametrize("mod", REMOVED)
def test_removed_modules_stay_removed(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


def test_utidemodel_in_gnss():
    """uTideModel is re-exported from gnss."""
    from cssrlib.gnss import uTideModel
    assert int(uTideModel.IERS2010) == 1 and int(uTideModel.NONE) == -1


def test_engine_class_is_gnssobs():
    """rtkpos and ppprtkpos both subclass the unified engine."""
    from cssrlib.gnssobs import gnssobs
    from cssrlib.rtk import rtkpos
    from cssrlib.ppprtk import ppprtkpos

    assert issubclass(rtkpos, gnssobs)
    assert issubclass(ppprtkpos, gnssobs)
