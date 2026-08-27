"""GLONASS broadcast-ephemeris propagation, compiled with Numba.

The RK4 integrator for :func:`cssrlib.models.ephemeris.geph2pos`. Splitting
it out is what lets it be a kernel at all: the integrator is arrays and
floats, while ``geph2pos`` handles a ``Geph`` object Numba cannot see.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.gnss import rCST

# Bound as module-level floats: Numba resolves globals at compile time and
# cannot read attributes off the rCST class from inside a kernel.
OMGE_GLO = float(rCST.OMGE_GLO)
MU_GLO = float(rCST.MU_GLO)
J2_GLO = float(rCST.J2_GLO)
RE_GLO = float(rCST.RE_GLO)


@njit(cache=True)
def _rk4_derivative(state: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Orbital derivatives for a GLONASS state vector: J2 + rotating frame."""

    deriv = np.zeros(6, dtype=np.float64)
    r2 = state[0] * state[0] + state[1] * state[1] + state[2] * state[2]
    if r2 <= 0.0:
        return deriv

    r = np.sqrt(r2)
    r3 = r2 * r
    omg2 = OMGE_GLO * OMGE_GLO
    a = 1.5 * J2_GLO * MU_GLO * RE_GLO * RE_GLO / (r2 * r3)
    b = 5.0 * state[2] * state[2] / r2
    c = -MU_GLO / r3 - a * (1.0 - b)

    deriv[0:3] = state[3:6]
    deriv[3] = (c + omg2) * state[0] + 2.0 * OMGE_GLO * state[4]
    deriv[4] = (c + omg2) * state[1] - 2.0 * OMGE_GLO * state[3]
    deriv[5] = (c - 2.0 * a) * state[2]
    deriv[3:6] += acc
    return deriv


@njit(cache=True)
def _rk4_step(dt: float, state: np.ndarray, acc: np.ndarray) -> None:
    """Advance ``state`` in place by ``dt`` seconds using RK4."""

    k1 = _rk4_derivative(state, acc)
    w = state + 0.5 * dt * k1
    k2 = _rk4_derivative(w, acc)
    w = state + 0.5 * dt * k2
    k3 = _rk4_derivative(w, acc)
    w = state + dt * k3
    k4 = _rk4_derivative(w, acc)
    state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)


@njit(cache=True)
def _propagate_state(dt: float, state: np.ndarray, acc: np.ndarray,
                     step: float) -> None:
    """Integrate ``state`` in place over |dt| seconds, ``step`` at a time.

    The final increment is whatever is left over, so the endpoint is exact
    rather than rounded to a whole ``step``.
    """

    t_left = dt
    direction = -step if t_left < 0.0 else step

    while True:
        if np.abs(t_left) <= 1e-9:
            break
        if np.abs(t_left) < step:
            direction = t_left
        _rk4_step(direction, state, acc)
        t_left -= direction


def propagate_glonass(
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    taun: float,
    gamn: float,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Propagate a GLONASS ephemeris ``dt`` seconds past its ``toe``.

    Returns ``(position, velocity, clock offset)`` in ECEF metres, m/s and
    seconds. ``pos``/``vel`` are copied, not propagated in place: one
    ephemeris serves every epoch in its validity window.
    """

    state = np.zeros(6, dtype=np.float64)
    state[0:3] = np.asarray(pos, dtype=np.float64).reshape(3)
    state[3:6] = np.asarray(vel, dtype=np.float64).reshape(3)

    _propagate_state(float(dt), state,
                     np.asarray(acc, dtype=np.float64).reshape(3),
                     float(step))

    clk = -float(taun) + float(gamn) * float(dt)
    return state[0:3].copy(), state[3:6].copy(), clk


__all__ = ["propagate_glonass"]
