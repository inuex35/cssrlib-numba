"""GLONASS propagation helpers compiled with Numba."""

from __future__ import annotations

import numpy as np

from cssrlib.gnss import rCST
from numba import njit

OMGE_GLO = float(rCST.OMGE_GLO)
MU_GLO = float(rCST.MU_GLO)
J2_GLO = float(rCST.J2_GLO)
RE_GLO = float(rCST.RE_GLO)


def _rk4_derivative_kernel(state: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Return orbital derivatives for a GLONASS state vector."""

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


def _rk4_step_kernel(dt: float, state: np.ndarray, acc: np.ndarray) -> None:
    """Advance the state vector by dt seconds using RK4."""

    k1 = _rk4_derivative(state, acc)
    w = state + 0.5 * dt * k1
    k2 = _rk4_derivative(w, acc)
    w = state + 0.5 * dt * k2
    k3 = _rk4_derivative(w, acc)
    w = state + dt * k3
    k4 = _rk4_derivative(w, acc)
    state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0)


def _propagate_state_kernel(dt: float, state: np.ndarray, acc: np.ndarray, step: float) -> None:
    """Integrate the GLONASS state forward/backward by |dt| seconds."""

    t_left = dt
    direction = -step if t_left < 0.0 else step

    while True:
        if np.abs(t_left) <= 1e-9:
            break
        if np.abs(t_left) < step:
            direction = t_left
        _rk4_step(direction, state, acc)
        t_left -= direction


_rk4_derivative = njit(cache=True)(_rk4_derivative_kernel)
_rk4_step = njit(cache=True)(_rk4_step_kernel)
_propagate_state = njit(cache=True)(_propagate_state_kernel)


def deq(state: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Public wrapper around the derivative kernel (compatible with legacy API)."""

    state_arr = np.asarray(state, dtype=np.float64)
    acc_arr = np.asarray(acc, dtype=np.float64)
    return _rk4_derivative(state_arr, acc_arr)


def glorbit(dt: float, state: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Runge–Kutta orbit propagation for legacy callers."""

    result = np.asarray(state, dtype=np.float64).copy()
    acc_arr = np.asarray(acc, dtype=np.float64)
    _rk4_step(float(dt), result, acc_arr)
    return result


def propagate_glonass(
    dt: float,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    taun: float,
    gamn: float,
    step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Propagate GLONASS broadcast ephemeris using the accelerated solver."""

    pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
    vel_arr = np.asarray(vel, dtype=np.float64).reshape(3)
    acc_arr = np.asarray(acc, dtype=np.float64).reshape(3)

    state = np.zeros(6, dtype=np.float64)
    state[0:3] = pos_arr
    state[3:6] = vel_arr
    _propagate_state(float(dt), state, acc_arr, float(step))
    clk = -float(taun) + float(gamn) * float(dt)
    return state[0:3].copy(), state[3:6].copy(), clk


__all__ = [
    "propagate_glonass",
    "deq",
    "glorbit",
]
