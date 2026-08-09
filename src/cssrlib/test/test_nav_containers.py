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


def test_rover_base_pairs_are_still_present():
    """Two fields today, one field on two ReceiverStates after the next step.

    gf/gf_r and rcv_ant/rcv_ant_b are the same quantity for the rover and
    the base, held side by side because one Nav has to serve both. When the
    base gets its own ReceiverState these collapse, and this test is the
    reminder to update.
    """
    nav = Nav(nf=2)
    assert hasattr(nav.rcv, "gf") and hasattr(nav.rcv, "gf_r")
    assert hasattr(nav.data, "rcv_ant") and hasattr(nav.data, "rcv_ant_b")
