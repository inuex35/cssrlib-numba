"""Numerical regression guard for the positioning pipeline.

Compares the current code against a committed reference produced from the
bundled dataset. Any change to satposs, qcedit, the single-difference build,
zdres, sdres, ddidx or ddcov that moves a number will fail here.

This is the test that commit 20c0df1 did not have: it reverted 20 Numba
kernels and broke four code paths, and nothing objected.

If a change is *meant* to move the numbers, regenerate the reference and say
so in the commit message:

    python -m cssrlib.test.golden_harness
"""

import numpy as np
import pytest

from cssrlib.test.golden_harness import GOLDEN, build_golden

REGEN = "python -m cssrlib.test.golden_harness"

# The pipeline is deterministic on fixed input, so the reference should
# reproduce exactly. A tiny tolerance absorbs nothing but genuine float
# noise from differing BLAS builds.
RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture(scope="module")
def reference():
    try:
        with np.load(GOLDEN) as fh:
            return {k: fh[k] for k in fh.files}
    except FileNotFoundError:  # pragma: no cover - only if the file is lost
        pytest.fail(f"golden reference missing: {GOLDEN}\nregenerate: {REGEN}")


@pytest.fixture(scope="module")
def current():
    return build_golden()


def test_reference_covers_every_stage(reference):
    """Guard the guard: a truncated reference must not pass silently."""
    stages = {k.split(".")[0] for k in reference}
    assert stages == {"dd", "zdres", "sdres", "ddidx", "ddcov", "synth"}
    assert reference["dd.epochs"][0] >= 20, "too few double-difference epochs"
    assert len(reference) > 300, f"only {len(reference)} arrays recorded"


def test_no_arrays_added_or_lost(current, reference):
    missing = sorted(set(reference) - set(current))
    added = sorted(set(current) - set(reference))
    assert not missing, (
        f"{len(missing)} recorded arrays are no longer produced, e.g. "
        f"{missing[:5]}. If intended: {REGEN}")
    assert not added, (
        f"{len(added)} new arrays appeared, e.g. {added[:5]}. "
        f"If intended: {REGEN}")


def test_pipeline_matches_reference(current, reference):
    shape_diffs, value_diffs = [], []

    for key in sorted(set(current) & set(reference)):
        got, want = current[key], reference[key]
        if got.shape != want.shape:
            shape_diffs.append(f"{key}: {want.shape} -> {got.shape}")
            continue
        if got.size == 0:
            continue
        if not np.allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=True):
            worst = float(np.nanmax(np.abs(got - want)))
            value_diffs.append(f"{key}: max|d|={worst:.6g}")

    report = []
    if shape_diffs:
        report.append(f"{len(shape_diffs)} shape changes:\n  " +
                      "\n  ".join(shape_diffs[:10]))
    if value_diffs:
        report.append(f"{len(value_diffs)} value changes:\n  " +
                      "\n  ".join(value_diffs[:10]))
    assert not report, ("pipeline output moved.\n" + "\n".join(report) +
                        f"\n\nIf intended: {REGEN}")


def test_multignss_sdres_is_covered(reference):
    """The bundled data is GPS-only; the synthetic cases carry the rest.

    sdres builds a reference satellite per constellation, so a GPS-only
    fixture never exercises that loop -- which is exactly where a
    multi-constellation regression would hide.
    """
    synth = [k for k in reference if k.startswith("synth.v#")]
    assert len(synth) == 12, f"expected 12 synthetic cases, got {len(synth)}"
    # Every case produced measurements, i.e. the loop really ran.
    for key in synth:
        assert reference[key].size > 0, f"{key} produced no residuals"
