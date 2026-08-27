"""RINEX navigation decoding must produce the same numbers it does today.

_decode_nav is a single 580-line function spanning RINEX 3.02 through 4.00
and six constellations. The bundled SEPT078M.21P is RINEX 3.04 with Galileo,
GPS and QZSS only, so the GLONASS, SBAS, BeiDou, IRNSS and RINEX-4 branches
had nothing behind them at all.

The reference is built from that real file plus four synthetic ones; see
nav_golden.py for what each is chosen to exercise. Regenerate with

    python -m cssrlib.test.nav_golden

and say so in the commit message, because a change here means a RINEX field
is being read differently.
"""

import os

import pytest

from cssrlib.test.nav_golden import REFERENCE, build, fingerprint, DATA, FIXTURES

REGEN = "python -m cssrlib.test.nav_golden"


@pytest.fixture(scope="module")
def reference():
    if not os.path.exists(REFERENCE):
        pytest.fail(f"{REFERENCE} is missing; generate it with {REGEN}")
    return open(REFERENCE).read().split("\n")


def test_decoding_matches_the_reference(reference):
    current = build().split("\n")

    if current == reference:
        return

    diffs = []
    for i, (got, want) in enumerate(zip(current, reference)):
        if got != want:
            diffs.append(f"  line {i}: {want!r} -> {got!r}")
        if len(diffs) == 10:
            break
    if len(current) != len(reference):
        diffs.append(f"  length {len(reference)} -> {len(current)}")

    pytest.fail("RINEX nav decoding moved.\n" + "\n".join(diffs)
                + f"\n\nIf intended: {REGEN}")


@pytest.mark.parametrize("name", FIXTURES)
def test_every_fixture_decodes_something(name):
    """A fixture that silently decodes to nothing would pin an empty result."""
    lines = fingerprint(os.path.join(DATA, name))
    records = [ln for ln in lines
               if ln.startswith(("eph[", "geph[", "seph[", "sto_prm",
                                 "eop_prm", "ion_prm"))]
    assert records, f"{name} decoded to no records at all"


def test_the_fixtures_cover_what_the_bundled_file_does_not():
    """Guard the point of the fixtures, not just their presence."""
    blob = "\n".join(fingerprint(os.path.join(DATA, "glonass305.rnx")))
    assert "geph[" in blob, "no GLONASS ephemeris decoded"
    assert " frq -126" in blob, (
        "the frequency number above 128 no longer wraps negative")

    blob = "\n".join(fingerprint(os.path.join(DATA, "sbas305.rnx")))
    assert "seph[" in blob, "no SBAS ephemeris decoded"

    blob = "\n".join(fingerprint(os.path.join(DATA, "beidou305.rnx")))
    assert "C03" in blob, "the BeiDou GEO record went missing"

    blob = "\n".join(fingerprint(os.path.join(DATA, "rinex400.rnx")))
    for table in ("sto_prm", "eop_prm", "ion_prm"):
        assert table in blob, f"RINEX-4 {table} record not decoded"
    assert " mode 1" in blob and " mode 2" in blob, (
        "CNAV and CNAV/2 are not both represented")
