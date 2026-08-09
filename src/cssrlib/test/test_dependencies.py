"""The layering, as an executable rule.

Dependencies must point downwards. This repository lost that once already:
ephemeris (which applies SSR corrections) imported the Compact SSR decoder
for two IntEnums, so broadcast-ephemeris RTK loaded a 1,300-line decoder it
never used. Nothing objected, because nothing was checking.
"""

import ast
import os
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Lower number = lower layer. A module may import its own layer or below.
LAYER = {
    # L0 numeric kernels: constants and arrays in, arrays out
    "constants": 0, "geometry": 0, "atmosphere": 0, "orbit": 0,
    "mlambda": 0, "ssr_types": 0,
    # L1 types and units
    "gnss": 1,
    # L2 data model. glonass sits here rather than with the other kernels
    # because its orbit integration is expressed in gtime_t.
    "state": 2, "config": 2, "glonass": 2,
    # L3 I/O and products
    "rinex_reader": 3, "rinex_writer": 3, "rinex": 3,
    "ephemeris": 3, "peph": 3, "ppp": 3,
    # L4 SSR decoding
    "cssrlib": 4, "cssr_bds": 4, "cssr_has": 4, "cssr_mdc": 4, "cssr_pvs": 4,
    # L5 observation model
    "qc": 5, "residuals": 5, "ambiguity": 5, "ekf": 5,
    # L6 engine and modes
    "gnssobs": 6, "rtk": 7, "ppprtk": 7,
}


def imports_of(module):
    path = os.path.join(SRC, module + ".py")
    found = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("cssrlib."):
                found.add(node.module.split(".", 1)[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cssrlib."):
                    found.add(alias.name.split(".", 1)[1])
    return found


@pytest.mark.parametrize("module", sorted(LAYER))
def test_module_does_not_import_upwards(module):
    """A module may import its own layer or below, never above.

    Import statements inside a function body are exempt: gnssobs defers
    cssrlib.config that way to avoid a construction-time cycle.
    """
    for target in sorted(imports_of(module)):
        if target not in LAYER:
            pytest.fail(f"{module} imports unlayered module {target}; "
                        f"add it to LAYER")
        assert LAYER[target] <= LAYER[module], (
            f"{module} (layer {LAYER[module]}) imports {target} "
            f"(layer {LAYER[target]}) -- dependencies must point downwards")


def test_broadcast_rtk_does_not_load_the_ssr_decoder():
    """Importing rtk must not drag in Compact SSR decoding.

    ephemeris used to import cssrlib.cssrlib for sCType / sCSSRTYPE, so this
    held a 1,359-line decoder that broadcast RTK never calls.
    """
    code = ("import cssrlib.rtk, sys; "
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith('cssrlib.cssr'))))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    loaded = [m for m in out.stdout.strip().split(",") if m]
    assert not loaded, f"rtk pulled in SSR decoders: {loaded}"


def test_ephemeris_stays_small():
    """satposs needs the SSR vocabulary, not the SSR decoder."""
    code = ("import cssrlib.ephemeris, sys; "
            "print(len([m for m in sys.modules if m.startswith('cssrlib')]))")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert int(out.stdout.strip()) <= 10, (
        "ephemeris's import closure grew; check for a new upward import")


def test_ssr_types_has_no_cssrlib_dependencies():
    """The whole point: this module is importable by anything."""
    assert imports_of("ssr_types") == set()


def test_cssrlib_still_re_exports_the_enums():
    """Existing callers import them from the decoder; keep that working."""
    from cssrlib.cssrlib import sCSSRTYPE, sCType
    from cssrlib import ssr_types

    assert sCType is ssr_types.sCType
    assert sCSSRTYPE is ssr_types.sCSSRTYPE
