"""Tests for the four containers Nav was split into.

Nav used to be 57 loose fields mixing six different lifetimes: navigation
data, processing configuration, per-receiver bookkeeping, the estimator's
state, an open log handle and per-run results. It now owns

    nav.data   NavData       ephemerides / corrections, shareable
    nav.cfg    ProcConfig    how to process
    nav.rcv    ReceiverState per-receiver bookkeeping
    nav.flt    FilterState   x and P

and delegates attribute access, so every existing ``nav.<field>`` call site
keeps working.
"""

from copy import deepcopy

import numpy as np
import pytest

from cssrlib.gnss import (Nav, NavData, ProcConfig, ReceiverState,
                          FilterState, _NAV_FIELDS, uGNSS)

COMPONENTS = ("data", "cfg", "rcv", "flt")


def test_nav_owns_the_four_containers():
    nav = Nav(nf=2)
    assert isinstance(nav.data, NavData)
    assert isinstance(nav.cfg, ProcConfig)
    assert isinstance(nav.rcv, ReceiverState)
    assert isinstance(nav.flt, FilterState)


def test_no_field_is_claimed_by_two_containers():
    seen = {}
    for component, fields in _NAV_FIELDS.items():
        for field in fields:
            assert field not in seen, (
                f"{field} is claimed by both {seen[field]} and {component}")
            seen[field] = component


def test_every_mapped_field_exists_on_its_container():
    nav = Nav(nf=2)
    for component, fields in _NAV_FIELDS.items():
        target = getattr(nav, component)
        for field in fields:
            assert hasattr(target, field), (
                f"{component} does not define {field}, so nav.{field} would "
                f"raise")


@pytest.mark.parametrize("field", sorted(
    f for fields in _NAV_FIELDS.values() for f in fields))
def test_delegation_round_trips(field):
    """nav.<field> and nav.<container>.<field> are the same storage."""
    nav = Nav(nf=2)
    component = next(c for c, fs in _NAV_FIELDS.items() if field in fs)
    container = getattr(nav, component)

    sentinel = object()
    setattr(nav, field, sentinel)
    assert getattr(container, field) is sentinel, "write did not reach it"

    other = object()
    setattr(container, field, other)
    assert getattr(nav, field) is other, "read did not come from it"


def test_containers_are_independent_after_deepcopy():
    """rtkpos still deepcopies nav to build its base; that must stay safe."""
    nav = Nav(nf=2)
    clone = deepcopy(nav)

    clone.edt[0, 0] = 7
    clone.cfg.elmin = 1.0
    clone.data.leaps = 99

    assert nav.edt[0, 0] == 0
    assert nav.elmin != 1.0
    assert nav.leaps == 18


def test_unmapped_attributes_still_work():
    """Callers add fields to nav on the fly; that must keep working."""
    nav = Nav(nf=2)
    nav.fout = None            # set by the engine
    nav.something_new = 42
    assert nav.fout is None and nav.something_new == 42


def test_array_shapes_follow_nf():
    for nf in (1, 2, 3):
        nav = Nav(nf=nf)
        assert nav.nf == nf
        for field in ("fix", "edt", "outc", "vsat", "lock", "slip"):
            assert getattr(nav, field).shape == (uGNSS.MAXSAT, nf)
        assert nav.eratio.shape == (nf,)


def test_receiver_state_is_constructible_on_its_own():
    """The point of the split: a second receiver needs no deepcopy of Nav."""
    base = ReceiverState(nf=2)
    assert base.edt.shape == (uGNSS.MAXSAT, 2)
    assert np.all(base.gf == 0.0)


def test_geometry_free_table_is_per_receiver():
    """gf_r is gone: the base carries its own gf on its own ReceiverState."""
    nav = Nav(nf=2)
    assert hasattr(nav.rcv, "gf")
    assert not hasattr(nav.rcv, "gf_r"), (
        "gf_r came back; the rover and base should each own a gf")

    rover, base = ReceiverState(nf=2), ReceiverState(nf=2)
    base.gf[5] = 1.5
    assert rover.gf[5] == 0.0, "receivers share a table"


def test_receiver_antenna_is_a_single_field():
    """rcv_ant_b is gone, like gf_r before it.

    antModelRx took the whole Nav plus an rtype flag to choose between the
    two; it now takes the antenna itself, so there is nothing to choose.
    """
    nav = Nav(nf=2)
    assert hasattr(nav.data, "rcv_ant")
    assert not hasattr(nav.data, "rcv_ant_b"), (
        "rcv_ant_b came back; antModelRx should take an antenna, not a Nav")


def test_antenna_model_takes_an_antenna_not_a_nav():
    import inspect
    from cssrlib.models.antenna import antModelRx

    params = list(inspect.signature(antModelRx).parameters)
    assert params[0] == "ant", f"first parameter is {params[0]!r}"
    assert "rtype" not in params, "the rover/base flag is back"
    assert "nav" not in params
