"""Deterministic fingerprint of RINEX navigation decoding.

Not a test module (pytest does not collect it) -- it writes the fixtures and
builds the reference that ``test_nav_decode.py`` compares against.

Why it exists: ``_decode_nav`` is one 580-line function covering four RINEX
generations and six constellations, and the bundled ``SEPT078M.21P`` reaches
none of the GLONASS, SBAS, BeiDou, IRNSS or RINEX-4 branches -- it is a
RINEX 3.04 file carrying Galileo, GPS and QZSS only. Roughly half of that
function had no test of any kind behind it, which is precisely the half a
refactor is most likely to break.

The synthetic files are not real observations and are not meant to be. Their
values are chosen to be distinguishable per field so a mis-parsed column
shows up as a wrong number rather than a plausible one, and to exercise the
branch conditions: a GLONASS frequency number above 128 (which must wrap
negative), a BeiDou GEO PRN, a RINEX-4 message type that is not in the type
table (which must be skipped along with its data lines), and GPS CNAV and
CNAV/2, which read a different number of lines.

Regenerate fixtures and reference with:

    python -m cssrlib.test.nav_golden
"""

import os

import numpy as np

import cssrlib.rinex as rn
from cssrlib.gnss import Nav, sat2id

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "navdata")
REFERENCE = os.path.join(DATA, "expected.txt")
BUNDLED = os.path.join(HERE, "..", "data", "SEPT078M.21P")

FIXTURES = ("glonass305.rnx", "sbas305.rnx", "beidou305.rnx", "rinex400.rnx")

EPOCH = "2021 03 19 12 00 00"

EPH_FIELDS = ("sat iode iodc sva svh week toe toc ttr A e i0 OMG0 omg M0 deln "
              "OMGd idot crc crs cuc cus cic cis toes fit af0 af1 af2 tgd "
              "tgd_b sid isc mode Adot delnd urai integ wn_op sisai").split()
GEPH_FIELDS = ("sat iode frq svh sva age toe tof pos vel acc taun gamn dtaun "
               "mode status flag tau_c dtau_c tau_gps urai").split()
SEPH_FIELDS = ("sat t0 toc tof tot sva svh pos vel acc af0 af1 iodn "
               "mode").split()
PARAM_FIELDS = ("sys mode prm a t0 t_ot t_t t_eop ttr sid").split()


def _fld(value):
    if isinstance(value, np.ndarray):
        return " ".join(f"{v:.17g}" for v in np.ravel(value))
    if isinstance(value, (list, tuple)):
        return " ".join(f"{v:.17g}" if isinstance(v, float) else str(v)
                        for v in value)
    if hasattr(value, "time"):
        return f"{value.time}:{value.sec:.17g}"
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _emit(out, tag, obj, fields):
    for name in fields:
        value = getattr(obj, name, None)
        if value is not None:
            out.append(f"{tag} {name} {_fld(value)}")


def fingerprint(path):
    """Every field decode_nav puts on a Nav, as sorted text lines."""
    dec = rn.rnxdec()
    nav = Nav()
    dec.decode_nav(path, nav)
    out = []

    for i, e in enumerate(sorted(nav.eph,
                                 key=lambda x: (x.sat, x.toe.time, x.mode))):
        _emit(out, f"eph[{i:03d}]{sat2id(e.sat)}", e, EPH_FIELDS)
    for i, e in enumerate(sorted(nav.geph, key=lambda x: (x.sat, x.toe.time))):
        _emit(out, f"geph[{i:03d}]{sat2id(e.sat)}", e, GEPH_FIELDS)
    for i, e in enumerate(sorted(nav.seph, key=lambda x: (x.sat, x.toc.time))):
        _emit(out, f"seph[{i:03d}]{sat2id(e.sat)}", e, SEPH_FIELDS)

    out.append("ion " + _fld(nav.ion))
    out.append(f"leaps {getattr(nav, 'leaps', None)}")

    for table in ("sto_prm", "eop_prm", "ion_prm"):
        entries = getattr(nav, table, None) or {}
        for sys_ in sorted(entries, key=int):
            for key in sorted(entries[sys_], key=str):
                _emit(out, f"{table}[{int(sys_)}][{key}]",
                      entries[sys_][key], PARAM_FIELDS)
    return out


def build():
    """The reference text: every fixture plus the bundled file."""
    out = []
    for name in FIXTURES + ("SEPT078M.21P",):
        path = os.path.join(DATA, name) if name in FIXTURES else BUNDLED
        out.append(f"### {name}")
        out.extend(fingerprint(path))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Fixture generation. RINEX nav data fields are fixed width: field c of a
# continuation line occupies columns 19c+4 .. 19(c+1)+4, and a record's first
# line carries the epoch in columns 4..22 followed by three of them.
# --------------------------------------------------------------------------

def _f(v):
    return f"{v: .12E}".replace("E", "D").rjust(19)


def _row(*values):
    return "    " + "".join(_f(v) for v in values)


def _epoch_row(sysc, prn, *values):
    return f"{sysc}{prn:02d} {EPOCH}".ljust(23) + "".join(_f(v) for v in values)


def _timed_row(*values):
    return "    " + EPOCH.ljust(19) + "".join(_f(v) for v in values)


def _record(kind, sysc, prn, itype, stype=""):
    """RINEX-4 record identifier: system at column 6, type at 10..13."""
    head = f"> {kind} {sysc}{prn:02d} {itype}"
    return head + (" " + stype if stype else "")


def _header(ver, extra=()):
    head = [f"{ver:9.2f}           {'N: GNSS NAV DATA':20s}{'M: Mixed':20s}"
            "RINEX VERSION / TYPE",
            "synthetic           cssrlib test        20210319 000000 UTC "
            "PGM / RUN BY / DATE"]
    head += list(extra)
    head.append(" " * 60 + "END OF HEADER")
    return head


def write_fixtures():
    os.makedirs(DATA, exist_ok=True)

    # GLONASS FDMA. Line #4 is read only from v3.05, and a frequency number
    # above 128 is a negative channel in disguise.
    lines = _header(3.05)
    for prn, frq in ((1, 1), (7, -4), (24, 130)):
        lines.append(_epoch_row("R", prn, -1.234567890123e-04,
                                9.094947017729e-13, 43200.0))
        lines += [_row(7.003119140625e+03, -1.234567890123e+00,
                       1.862645149231e-06, 0.0),
                  _row(-1.234567890123e+04, 2.345678901234e+00,
                       9.313225746155e-07, frq),
                  _row(2.098765432109e+04, 3.456789012345e-01,
                       -2.793967723846e-06, 3.0),
                  _row(13.0, 1.396983861923e-09, 2.0, 0.0)]
    _write("glonass305.rnx", lines)

    lines = _header(3.05)
    for prn in (20, 37):
        lines.append(_epoch_row("S", prn, 1.862645149231e-09,
                                3.637978807092e-12, 43200.0))
        lines += [_row(2.456789012345e+04, 1.234567890123e-01,
                       2.5e-07, 0.0),
                  _row(-3.456789012345e+04, -2.345678901234e-01,
                       1.25e-07, 3.0),
                  _row(1.234567890123e+03, 5.678901234567e-02,
                       -6.25e-08, 12.0)]
    _write("sbas305.rnx", lines)

    # BeiDou D1/D2. PRN 3 is a GEO, which eph2pos propagates in its own frame.
    lines = _header(3.05)
    for prn in (3, 21):
        lines.append(_epoch_row("C", prn, -1.234567890123e-04,
                                5.684341886081e-12, 0.0))
        lines += [_row(1.0, -1.234567890123e+02, 1.234567890123e-09,
                       1.234567890123e-01),
                  _row(-6.146728992462e-06, 2.345678901234e-03,
                       8.245930075645e-06, 6.493410308838e+03),
                  _row(43200.0, -1.303851604462e-08, 2.345678901234e+00,
                       7.450580596924e-09),
                  _row(9.876543210987e-01, 1.234567890123e+02,
                       -1.234567890123e+00, -2.345678901234e-09),
                  _row(1.234567890123e-10, 0.0, 790.0, 0.0),
                  _row(2.0, 0.0, -1.1e-08, -1.2e-08),
                  _row(43200.0, 0.0, 0.0, 0.0)]
    _write("beidou305.rnx", lines)

    # RINEX 4: the record-dispatch branches, plus a type that is not in the
    # table and must be skipped with its two data lines.
    lines = _header(4.00, ["    18" + " " * 54 + "LEAP SECONDS"])
    lines += [_record("STO", "G", 1, "LNAV"),
              "    " + EPOCH.ljust(20) + "GPUT" + " " * 40,
              _row(43200.0, 1.0e-09, 2.0e-15, 0.0)]
    lines += [_record("STO", "E", 1, "XXXX"),
              _row(0.0, 0.0, 0.0, 0.0), _row(0.0, 0.0, 0.0, 0.0)]
    lines += [_record("EOP", "G", 1, "LNAV"),
              _timed_row(1.0e-01, 2.0e-04, 3.0e-08),
              _row(0.0, 4.0e-01, 5.0e-04, 6.0e-08),
              _row(43200.0, 7.0e-01, 8.0e-04, 9.0e-08)]
    lines += [_record("ION", "G", 1, "LNAV"),          # Klobuchar
              _timed_row(1.0e-08, 2.0e-08, 3.0e-08),
              _row(4.0e-08, 5.0e+04, 6.0e+04, 7.0e+04),
              _row(8.0e+04, 1.0, 0.0, 0.0)]
    lines += [_record("ION", "E", 1, "IFNV"),          # NeQuick-G
              _timed_row(4.5e+01, 1.5e-01, 2.5e-03),
              _row(1.0, 0.0, 0.0, 0.0)]
    lines += [_record("ION", "C", 1, "CNVX", "BDGM"),  # BDGIM
              _timed_row(1.1, 2.2, 3.3),
              _row(4.4, 5.5, 6.6, 7.7),
              _row(8.8, 9.9, 0.0, 0.0)]
    kepler = [_row(1.0, -3.0e+01, 4.0e-09, 1.5e+00),
              _row(-6.0e-06, 2.0e-03, 8.0e-06, 5.153e+03),
              _row(43200.0, -1.0e-08, 2.0e+00, 7.0e-09),
              _row(9.6e-01, 1.2e+02, -1.1e+00, -2.0e-09),
              _row(1.0e-10, 0.0, 2148.0, 0.0),
              _row(2.0, 0.0, -1.0e-08, 5.0)]
    lines += [_record("EPH", "G", 5, "CNAV"),
              _epoch_row("G", 5, -1.0e-04, 2.0e-12, 0.0)] + kepler + [
              _row(1.0e-09, 2.0e-09, 3.0e-09, 4.0e-09),
              _row(43200.0, 2148.0, 1.0, 0.0)]
    lines += [_record("EPH", "G", 7, "CNV2"),
              _epoch_row("G", 7, -2.0e-04, 3.0e-12, 0.0)] + kepler + [
              _row(1.1e-09, 2.1e-09, 3.1e-09, 4.1e-09),
              _row(5.1e-09, 6.1e-09, 0.0, 0.0),
              _row(43200.0, 2148.0, 1.0, 0.0)]
    _write("rinex400.rnx", lines)


def _write(name, lines):
    with open(os.path.join(DATA, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_fixtures()
    with open(REFERENCE, "w") as fh:
        fh.write(build())
    print(f"wrote {len(FIXTURES)} fixtures and {REFERENCE}")
