"""Numba-accelerated troposphere helpers."""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.constants import HALFPI


@njit(cache=True)
def meteo(hgt: float, humi: float) -> tuple[float, float, float]:
    """Return pressure, temperature, and vapor pressure for a given height."""

    pres = 1013.25 * (1.0 - 2.2557e-5 * hgt) ** 5.2568
    temp = 15.0 - 6.5e-3 * hgt + 273.16
    e = 6.108 * humi * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))
    return pres, temp, e


@njit(cache=True)
def mapf(el: float, a: float, b: float, c: float) -> float:
    """Simple tropospheric mapping function."""

    sinel = np.sin(el)
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (sinel + (a / (sinel + b / (sinel + c))))


@njit(cache=True)
def tropmapf_niell(doy: float, pos: np.ndarray, el: float) -> tuple[float, float]:
    """Niell mapping function given day-of-year, LLH, and elevation."""

    if pos[2] < -1e3 or pos[2] > 20e3 or el <= 0.0:
        return 0.0, 0.0
    coef = np.array([
        [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
        [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
        [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3],
        [0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5],
        [0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5],
        [0.0, 9.01284e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5],
        [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4],
        [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3],
        [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2],
    ], dtype=np.float64)
    aht = np.array((2.53e-5, 5.49e-3, 1.14e-3), dtype=np.float64)
    lat_deg = pos[0] * 180.0 / np.pi
    hgt = pos[2]
    y = (doy - 28.0) / 365.25
    if lat_deg < 0.0:
        y += 0.5
    cosy = np.cos(2.0 * np.pi * y)
    lat_abs = abs(lat_deg)
    i = int(lat_abs / 15.0)
    if i < 1:
        c = coef[:, 0]
    elif i > 4:
        c = coef[:, 4]
    else:
        d = lat_abs / 15.0 - i
        c = coef[:, i - 1] * (1.0 - d) + coef[:, i] * d
    ah = c[0:3] - c[3:6] * cosy
    aw = c[6:9]
    dm = (1.0 / np.sin(el) - mapf(el, aht[0], aht[1], aht[2])) * hgt * 1e-3
    mapfh = mapf(el, ah[0], ah[1], ah[2]) + dm
    mapfw = mapf(el, aw[0], aw[1], aw[2])
    return mapfh, mapfw


@njit(cache=True)
def tropmodel_saast(pos: np.ndarray, el: float, humi: float) -> tuple[float, float, float]:
    """Saastamoinen tropospheric delay model."""

    hgt = pos[2]
    pres, temp, e = meteo(hgt, humi)
    z = HALFPI - el
    trop_hs = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * pos[0]) - 0.00028e-3 * hgt)
    trop_wet = 0.002277 * (1255.0 / temp + 0.05) * e
    return trop_hs, trop_wet, z
