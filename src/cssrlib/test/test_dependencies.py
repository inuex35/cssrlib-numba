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
import os
import subprocess
import sys

import pytest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bottom to top. A package may import its own layer or anything below.
LAYERS = ["core", "types", "models", "fileio", "ssr", "estimation", "engine"]
DEPTH = {name: i for i, name in enumerate(LAYERS)}

# Names that predate the package layout and stay at the top level as public
# API. Each forwards into a package; for layering purposes it counts as
# wherever it forwards to.
FACADES = {
    "gnss": "types", "rinex": "fileio", "peph": "models",
    "constants": "core", "geometry": "core", "atmosphere": "core",
    "orbit": "core", "mlambda": "core",
    "glonass": "models", "ephemeris": "models", "ppp": "models",
    "cssrlib": "ssr", "cssr_bds": "ssr", "cssr_has": "ssr",
    "cssr_mdc": "ssr", "cssr_pvs": "ssr",
    "gnssobs": "engine", "rtk": "engine", "ppprtk": "engine",
}


def package_modules():
    """(dotted name, package, path) for every module inside a layer."""
    out = []
    for pkg in LAYERS:
        d = os.path.join(SRC, pkg)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and fn != "__init__.py":
                out.append((f"{pkg}.{fn[:-3]}", pkg, os.path.join(d, fn)))
    return out


def facade_paths():
    return [(name, os.path.join(SRC, name + ".py")) for name in FACADES]


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


@pytest.mark.parametrize("name,path", facade_paths(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_facade_only_forwards(name, path):
    """A compatibility name must re-export, not grow logic of its own."""
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                      # module docstring
        if isinstance(node, ast.Assign):  # __all__
            continue
        pytest.fail(f"{name}.py contains {type(node).__name__} at line "
                    f"{node.lineno}; facades forward and nothing else")


@pytest.mark.parametrize("name,path", facade_paths(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_facade_is_importable(name, path):
    __import__(f"cssrlib.{name}")


def test_every_module_lives_in_a_layer():
    """Nothing loose at the top level except the documented facades."""
    stray = {fn[:-3] for fn in os.listdir(SRC)
             if fn.endswith(".py") and fn != "__init__.py"}
    stray -= set(FACADES)
    assert not stray, (
        f"{sorted(stray)} sit at the top level; move them into a package or "
        f"declare them facades")


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
