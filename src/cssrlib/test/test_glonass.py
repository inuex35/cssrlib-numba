"""GLONASS broadcast propagation: the numbers, and which code produces them.

Nothing exercised this path. The bundled dataset is GPS-only, so the whole
GLONASS orbit integrator -- ``geph2pos`` and the RK4 under it -- ran without
a single assertion behind it. That is how ``models/ephemeris.py`` came to
carry a second, pure-Python copy of the integrator and call *that*, while
the Numba kernel in ``models/glonass.py`` sat unreferenced and 40x faster.

The reference below is the integrator written out longhand, independent of
the library. It pins the arithmetic; ``test_geph2pos_uses_the_compiled_kernel``
pins which implementation the library reaches for.
"""

import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher

import cssrlib.models.ephemeris as ephemeris
import cssrlib.models.glonass as glonass
from cssrlib.gnss import Geph, epoch2time, rCST, timeadd

# One ULP of a 2.55e7 m GLONASS radius. The reference and the kernel group
# the same operations differently (``x/r2/r3`` against ``x/(r2*r3)``), which
# is a last-bit difference, not a different orbit.
POS_ATOL = 4e-9        # m
VEL_ATOL = 1e-12       # m/s

TOE = epoch2time([2021, 3, 19, 0, 15, 0])


def reference_deq(x, acc):
    """Orbital derivative, written out independently of the library."""
    xdot = np.zeros(6)
    r2 = x[0:3] @ x[0:3]
    if r2 <= 0.0:
        return xdot
    r3 = r2 * np.sqrt(r2)
    omg2 = rCST.OMGE_GLO**2
    a = 1.5 * rCST.J2_GLO * rCST.MU_GLO * rCST.RE_GLO**2 / r2 / r3
    b = 5.0 * x[2]**2 / r2
    c = -rCST.MU_GLO / r3 - a * (1.0 - b)
    xdot[0:3] = x[3:6]
    xdot[3] = (c + omg2) * x[0] + 2.0 * rCST.OMGE_GLO * x[4]
    xdot[4] = (c + omg2) * x[1] - 2.0 * rCST.OMGE_GLO * x[3]
    xdot[5] = (c - 2.0 * a) * x[2]
    xdot[3:6] += acc
    return xdot


def reference_propagate(t, pos, vel, acc, step=1.0):
    """RK4 over |t| seconds in ``step``-second increments, then the remainder."""
    x = np.zeros(6)
    x[0:3], x[3:6] = pos, vel
    tt = -step if t < 0.0 else step
    while np.fabs(t) > 1e-9:
        if np.fabs(t) < step:
            tt = t
        k1 = reference_deq(x, acc)
        k2 = reference_deq(x + k1 * tt / 2.0, acc)
        k3 = reference_deq(x + k2 * tt / 2.0, acc)
        k4 = reference_deq(x + k3 * tt, acc)
        x = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * tt / 6.0
        t -= tt
    return x[0:3], x[3:6]


def make_geph(seed):
    """A GLONASS ephemeris on a plausible orbit: r=25.51e6 m, |v|=3.6 km/s."""
    rng = np.random.default_rng(seed)
    radial = rng.normal(size=3)
    radial /= np.linalg.norm(radial)
    along = np.cross(radial, rng.normal(size=3))
    along /= np.linalg.norm(along)

    geph = Geph()
    geph.sat = 1
    geph.toe = TOE
    geph.pos = radial * 25.51e6
    geph.vel = along * 3600.0
    geph.acc = rng.normal(size=3) * 1e-6      # luni-solar, ~1e-6 m/s^2
    geph.taun = rng.normal() * 1e-4
    geph.gamn = rng.normal() * 1e-13
    return geph


# ``toe`` +/- half a GLONASS ephemeris validity window (1800 s), plus the
# sub-step remainder cases and the t == 0 short circuit.
@pytest.mark.parametrize("dt", [0.0, 0.5, -0.5, 1.0, -1.0, 30.25,
                                900.0, -900.0, 1799.5, -1799.5])
def test_geph2pos_matches_the_reference_integrator(dt):
    geph = make_geph(seed=int(abs(dt) * 4) + 1)
    time = timeadd(geph.toe, dt)

    rs, vs, dts = ephemeris.geph2pos(time, geph, flg_v=True)
    ref_pos, ref_vel = reference_propagate(dt, geph.pos, geph.vel, geph.acc)

    assert np.allclose(rs, ref_pos, rtol=0.0, atol=POS_ATOL)
    assert np.allclose(vs, ref_vel, rtol=0.0, atol=VEL_ATOL)
    # The clock offset is evaluated at the *unpropagated* dt, not iterated.
    assert dts == -geph.taun + geph.gamn * dt


def test_geph2pos_without_velocity_returns_the_same_position():
    geph = make_geph(seed=7)
    time = timeadd(geph.toe, 600.0)

    rs, dts = ephemeris.geph2pos(time, geph)
    rs_v, _, dts_v = ephemeris.geph2pos(time, geph, flg_v=True)

    assert np.array_equal(rs, rs_v)
    assert dts == dts_v


def test_geph2pos_does_not_consume_the_ephemeris():
    """The integrator must not write back into ``geph.pos``/``geph.vel``.

    The state vector is built by copying ``pos``/``vel`` into a scratch
    array; propagating in place would corrupt the ephemeris for every later
    epoch that reuses it.
    """
    geph = make_geph(seed=11)
    pos0, vel0 = geph.pos.copy(), geph.vel.copy()

    ephemeris.geph2pos(timeadd(geph.toe, 900.0), geph, flg_v=True)

    assert np.array_equal(geph.pos, pos0)
    assert np.array_equal(geph.vel, vel0)


def test_geph2pos_uses_the_compiled_kernel():
    """A pure-Python integrator here is the regression, not a slow test.

    ``geph2pos`` must reach ``models.glonass``; the module-level binding is
    what a merge would silently replace with a local copy.
    """
    assert ephemeris.propagate_glonass is glonass.propagate_glonass
    assert isinstance(glonass._propagate_state, Dispatcher)
    assert isinstance(glonass._rk4_step, Dispatcher)
    assert isinstance(glonass._rk4_derivative, Dispatcher)

    # And no second copy survives in the ephemeris module.
    assert not hasattr(ephemeris, "glorbit")
    assert not hasattr(ephemeris, "deq")
