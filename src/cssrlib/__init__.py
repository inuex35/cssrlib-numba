"""CSSRlib - toolkit for PPP / PPP-RTK / RTK positioning.

The package is organised in layers, bottom to top:

    cssrlib.core         pure numerics and vocabulary
    cssrlib.domain       constellations, signals, time, coordinates, structs
    cssrlib.models       orbits, Earth frames, tides, antennas, biases
    cssrlib.fileio       RINEX reading and writing
    cssrlib.ssr          State-Space Representation correction decoding
    cssrlib.estimation   state layout, configuration, observation model, filter
    cssrlib.engine       the composed engine and the modes it is configured into

Dependencies point downwards only; ``test_dependencies.py`` enforces it.

Note this file is not decorative even apart from the aliases below: without
it ``packages = find:`` discovers nothing and a non-editable
``pip install .`` produces a distribution containing no code at all.

Compatibility
-------------
The module names that predate the layer layout still work --
``cssrlib.gnss``, ``cssrlib.rtk``, ``cssrlib.peph`` and the rest. Rather
than nineteen forwarding files cluttering the package root, they are
resolved on import by :data:`LEGACY_MODULES` and the finder below.

The alias is the *same module object* as its target, not a second copy: the
loader hands back the already-imported module instead of executing the
source again. A finder that instead returned the target's own spec would
run the module twice under two names, giving two sets of classes and
breaking ``isinstance`` across the boundary.

Resolution is lazy. Importing ``cssrlib`` loads nothing; a target is
imported the first time its legacy name is, so ``import cssrlib.rtk`` still
pulls in no SSR decoder.

One caveat: with no file behind them, static analysers and IDEs cannot
resolve the legacy names and ``pkgutil.walk_packages`` does not list them.
New code should import from the layer that owns it.
"""

import importlib
import importlib.abc
import importlib.util
import sys

__version__ = "1.2.1"

#: Pre-layer module name -> the module that now provides it.
#:
#: gnss, rinex and peph are NOT here. They aggregate a whole layer, and more
#: importantly third-party code derives paths from them: the official GTSAM
#: gnss_frontend.py locates the bundled RINEX as
#: ``dirname(cssrlib.gnss.__file__)/data``. A module's location is part of
#: its contract when the package ships data beside it, so those three stay
#: real files at the package root.
LEGACY_MODULES = {
    "constants": "core.constants",
    "geometry": "core.geometry",
    "atmosphere": "core.atmosphere",
    "orbit": "core.orbit",
    "mlambda": "core.mlambda",

    "glonass": "models.glonass",
    "ephemeris": "models.ephemeris",
    "ppp": "models.tides",

    "cssrlib": "ssr.base",
    "cssr_bds": "ssr.bds",
    "cssr_has": "ssr.has",
    "cssr_mdc": "ssr.mdc",
    "cssr_pvs": "ssr.pvs",

    "gnssobs": "engine.gnssobs",
    "rtk": "engine.rtk",
    "ppprtk": "engine.ppprtk",
}


class _AliasLoader(importlib.abc.Loader):
    """Loads a legacy name by handing back the module it aliases."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass          # the target module has already executed itself


class _LegacyModuleFinder(importlib.abc.MetaPathFinder):
    """Resolves ``cssrlib.<legacy name>`` to the module that replaced it."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        tail = fullname[len(__name__) + 1:]
        if tail not in LEGACY_MODULES:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(f"{__name__}.{LEGACY_MODULES[tail]}"))


sys.meta_path.insert(0, _LegacyModuleFinder())
