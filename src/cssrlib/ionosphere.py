"""Numba-accelerated ionospheric delay helpers."""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.constants import CLIGHT, PI


@njit(cache=True)
def klobuchar_delay(
    tow: float,
    lat: float,
    lon: float,
    az: float,
    el: float,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Compute Klobuchar ionospheric delay for L1 (meters)."""

    psi = 0.0137 / (el / PI + 0.11) - 0.022
    phi = lat / PI + psi * np.cos(az)
    if phi > 0.416:
        phi = 0.416
    elif phi < -0.416:
        phi = -0.416
    lam = lon / PI + psi * np.sin(az) / np.cos(phi * PI)
    phi += 0.064 * np.cos((lam - 1.617) * PI)
    tt = 43200.0 * lam + tow
    tt -= np.floor(tt / 86400.0) * 86400.0
    sf = 1.0 + 16.0 * (0.53 - el / PI) ** 3

    h = np.array((1.0, phi, phi * phi, phi * phi * phi), dtype=np.float64)
    amp = np.dot(h, alpha)
    if amp < 0.0:
        amp = 0.0
    per = np.dot(h, beta)
    if per < 72000.0:
        per = 72000.0

    x = 2.0 * np.pi * (tt - 50400.0) / per
    if abs(x) < 1.57:
        v = 5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0))
    else:
        v = 5e-9
    return CLIGHT * sf * v
