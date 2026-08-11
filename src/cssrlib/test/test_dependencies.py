"""The layering, derived from the tree rather than from a list.

Dependencies must point downwards. A module's layer is now its package
directory, so adding a file puts it in a layer by construction -- there is
no table to forget to update, which is what the previous version of this
test needed and what made its assignments drift (glonass filed as a data
model, ephemeris and the tide model filed as I/O).

This repository lost the rule once already: ephemeris, which applies SSR
corrections, imported the Compact SSR decoder for two IntEnums, so
broadcast-ephemeris RTK loaded a 1,300-line decoder it never used.
"""

import ast
import importlib
import os
import subprocess
import sys

import pytest

import cssrlib

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bottom to top. A package may import its own layer or anything below.
LAYERS = ["core", "domain", "models", "fileio", "ssr", "estimation", "engine"]
DEPTH = {name: i for i, name in enumerate(LAYERS)}

# The three modules that stay at the package root, and the layer each one
# belongs to for dependency purposes. They aggregate a whole layer, and their
# path is part of their contract -- see test_bundled_data_sits_beside_gnss.
# They are still bound by the rule: peph may reach down into domain, gnss may
# not reach up into models.
ROOT_MODULES = {"__init__.py": None, "gnss.py": "domain",
                "rinex.py": "fileio", "peph.py": "models"}

# Names that predate the layer layout. They are no longer files: cssrlib's
# __init__ resolves them with a meta-path finder, so for layering purposes
# each counts as wherever it forwards to.
FACADES = {name: target.split(".")[0]
           for name, target in cssrlib.LEGACY_MODULES.items()}
FACADES.update({fn[:-3]: pkg for fn, pkg in ROOT_MODULES.items() if pkg})


def package_modules():
    """(dotted name, package, path) for every module under the layer rule."""
    out = []
    for pkg in LAYERS:
        d = os.path.join(SRC, pkg)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and fn != "__init__.py":
                out.append((f"{pkg}.{fn[:-3]}", pkg, os.path.join(d, fn)))
    for fn, pkg in sorted(ROOT_MODULES.items()):
        if pkg:
            out.append((fn[:-3], pkg, os.path.join(SRC, fn)))
    return out


def _is_main_guard(node):
    """True for `if __name__ == "__main__":`."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__")


def cssrlib_imports(path):
    """Every cssrlib module this file depends on to be imported.

    A `__main__` demo block is skipped: it runs only when the file is
    executed as a script, so what it reaches for says nothing about where
    the module sits. models/tides.py builds a plot from a RINEX file that
    way, which is not a reason for the tide model to sit above file I/O.
    """
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if not _is_main_guard(n)]

    found = set()
    for top in body:
        for node in ast.walk(top):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("cssrlib."):
                    found.add(node.module.split(".", 1)[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("cssrlib."):
                        found.add(alias.name.split(".", 1)[1])
    return found


def layer_of(target):
    """Which layer an import target belongs to, resolving facades."""
    head = target.split(".")[0]
    if head in DEPTH:
        return head
    return FACADES.get(head)


@pytest.mark.parametrize("name,pkg,path", package_modules(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_module_does_not_import_upwards(name, pkg, path):
    for target in sorted(cssrlib_imports(path)):
        target_pkg = layer_of(target)
        assert target_pkg is not None, (
            f"{name} imports {target}, which is in no layer; put it in a "
            f"package or list it as a facade")
        assert DEPTH[target_pkg] <= DEPTH[pkg], (
            f"{name} ({pkg}, layer {DEPTH[pkg]}) imports {target} "
            f"({target_pkg}, layer {DEPTH[target_pkg]}) -- "
            f"dependencies must point downwards")


@pytest.mark.parametrize("name,target",
                         sorted(cssrlib.LEGACY_MODULES.items()),
                         ids=lambda v: v)
def test_legacy_name_is_the_same_module_as_its_target(name, target):
    """The alias must BE the module, not a second copy of it.

    A finder that returned the target's own spec would execute the source
    again under the legacy name, producing two sets of classes -- so
    isinstance(nav, cssrlib.gnss.Nav) would fail for a Nav built through
    cssrlib.gnss.
    """
    alias = importlib.import_module(f"cssrlib.{name}")
    real = importlib.import_module(f"cssrlib.{target}")
    assert alias is real


@pytest.mark.parametrize("statement", [
    "import cssrlib.gnss; cssrlib.gnss.Nav",
    "from cssrlib.gnss import Nav; Nav",
    "import cssrlib.gnss as gn; gn.Nav",
    "from cssrlib import gnss; gnss.Nav",
    "import importlib; importlib.import_module('cssrlib.gnss').Nav",
])
def test_every_import_form_resolves(statement):
    """Third-party code uses all of these; gnss_frontend.py uses three."""
    out = subprocess.run([sys.executable, "-c", statement],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_isinstance_survives_the_alias_boundary():
    import cssrlib.gnss as legacy
    from cssrlib.gnss import Nav

    assert isinstance(Nav(nf=2), legacy.Nav)


def test_importing_the_package_loads_no_layer():
    """Resolution is lazy: cssrlib itself must stay free."""
    code = ("import cssrlib, sys; "
            "print(','.join(m for m in sys.modules "
            "if m.startswith('cssrlib.')))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert not out.stdout.strip(), f"eagerly loaded {out.stdout.strip()}"


def test_every_module_lives_in_a_layer():
    """The package root holds the three aggregate facades and nothing else.

    gnss, rinex and peph stay here because their *location* is part of their
    contract -- see test_bundled_data_sits_beside_gnss below.
    """
    stray = sorted(fn for fn in os.listdir(SRC)
                   if fn.endswith(".py") and fn not in ROOT_MODULES)
    assert not stray, (
        f"{stray} sit at the package root; put them in the layer that owns "
        f"them, and add a LEGACY_MODULES entry if the name must survive")


def test_bundled_data_sits_beside_gnss():
    """Third-party code locates the bundled RINEX from cssrlib.gnss.__file__.

    The official GTSAM front end (borglab/gtsam
    python/gtsam/examples/gnss_frontend.py) does

        bdir = os.path.join(os.path.dirname(gn.__file__), "data") + os.sep

    so moving gnss.py into a layer package silently redirects that to
    cssrlib/<layer>/data, which does not exist -- the notebook CI failed with
    FileNotFoundError on SEPT078M.21P the one time it happened. A module's
    path is part of its interface when the package ships data next to it.
    """
    import cssrlib.gnss as gn

    bdir = os.path.join(os.path.dirname(gn.__file__), "data")
    assert os.path.isdir(bdir), f"no data directory beside gnss.py: {bdir}"
    for fn in ("SEPT078M.21P", "SEPT078M1.21O", "3034078M1.21O"):
        assert os.path.isfile(os.path.join(bdir, fn)), f"{fn} not in {bdir}"


def test_no_package_shadows_the_standard_library():
    """A layer directory must not share a name with a stdlib module.

    src/cssrlib is on sys.path whenever a module is run directly from that
    directory, which several __main__ demo blocks expect. A package named
    `types` therefore shadowed the stdlib module of that name, and functools
    imports types -- so the interpreter broke before any cssrlib code ran.
    That is why the GNSS type layer is called `domain`.
    """
    stdlib = set(sys.stdlib_module_names)
    clashing = sorted(set(LAYERS) & stdlib)
    assert not clashing, (
        f"{clashing} shadow standard library modules; running anything from "
        f"inside src/cssrlib would import these instead")


def test_running_a_module_from_inside_the_package_still_works():
    """The regression above, reproduced end to end."""
    out = subprocess.run([sys.executable, "-c", "import functools; print('ok')"],
                         cwd=SRC, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, (
        f"the interpreter cannot start from src/cssrlib:\n{out.stderr}")


def test_broadcast_rtk_does_not_load_the_ssr_decoder():
    """Importing rtk must not drag in Compact SSR decoding."""
    code = ("import cssrlib.rtk, sys; "
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith('cssrlib.ssr'))))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    loaded = [m for m in out.stdout.strip().split(",") if m]
    assert not loaded, f"rtk pulled in SSR decoders: {loaded}"


def test_ephemeris_pulls_in_no_products_or_decoders():
    """satposs needs the SSR vocabulary, not the SSR decoder."""
    code = ("import cssrlib.models.ephemeris, sys; "
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith('cssrlib.'))))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    loaded = set(out.stdout.strip().split(","))

    forbidden = {"cssrlib.ssr.base", "cssrlib.models.antenna",
                 "cssrlib.models.precise", "cssrlib.models.bias",
                 "cssrlib.fileio.reader", "cssrlib.engine.gnssobs",
                 "cssrlib.estimation.residuals"}
    assert not (loaded & forbidden), (
        f"ephemeris pulled in {sorted(loaded & forbidden)}")


def test_ssr_vocabulary_has_no_dependencies():
    """core.ssr_types is importable by anything; that is the point of it."""
    assert cssrlib_imports(os.path.join(SRC, "core", "ssr_types.py")) == set()


def test_the_decoder_still_re_exports_the_enums():
    from cssrlib.ssr.base import sCSSRTYPE, sCType
    from cssrlib.core import ssr_types

    assert sCType is ssr_types.sCType
    assert sCSSRTYPE is ssr_types.sCSSRTYPE
