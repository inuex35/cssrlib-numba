"""Numba-accelerated broadcast orbit propagation for GPS/GAL/QZS/BDS."""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.constants import CLIGHT, COS_5, SIN_5


@njit(cache=True)
def broadcast_orbit(
    dt: float,
    dtc: float,
    n: float,
    Ak: float,
    M: float,
    ecc: float,
    omg: float,
    cuc: float,
    cus: float,
    crc: float,
    crs: float,
    cic: float,
    cis: float,
    i0: float,
    idot: float,
    OMG0: float,
    OMGd: float,
    omge: float,
    toes: float,
    is_bds_geo: int,
    sqrt_mu_A: float,
    af0: float,
    af1: float,
    af2: float,
    compute_vel: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return broadcast orbit position/velocity/clock from ephemeris scalars."""

    E = M
    for _ in range(30):
        E_prev = E
        E = M + ecc * np.sin(E)
        if abs(E - E_prev) < 1e-12:
            break

    sE = np.sin(E)
    cE = np.cos(E)
    nue = 1.0 - ecc * cE
    nus = np.sqrt(1.0 - ecc * ecc) * sE
    nuc = cE - ecc
    nu = np.arctan2(nus, nuc)
    phi = nu + omg
    cos2 = np.cos(2.0 * phi)
    sin2 = np.sin(2.0 * phi)

    u = phi + cuc * cos2 + cus * sin2
    r = Ak * nue + crc * cos2 + crs * sin2
    inc = i0 + idot * dt + cic * cos2 + cis * sin2
    si = np.sin(inc)
    ci = np.cos(inc)

    sin_u = np.sin(u)
    cos_u = np.cos(u)
    xo = np.array((r * cos_u, r * sin_u), dtype=np.float64)

    if is_bds_geo:
        Omg = OMG0 + OMGd * dt - omge * toes
        sOmg = np.sin(Omg)
        cOmg = np.cos(Omg)
        p = np.array((cOmg, sOmg, 0.0), dtype=np.float64)
        q = np.array((-ci * sOmg, ci * cOmg, si), dtype=np.float64)
        rg = np.array(
            (
                xo[0] * p[0] + xo[1] * q[0],
                xo[0] * p[1] + xo[1] * q[1],
                xo[0] * p[2] + xo[1] * q[2],
            ),
            dtype=np.float64,
        )
        so = np.sin(omge * dt)
        co = np.cos(omge * dt)
        Mo = np.array(
            [
                (co, so * COS_5, so * SIN_5),
                (-so, co * COS_5, co * SIN_5),
                (0.0, -SIN_5, COS_5),
            ],
            dtype=np.float64,
        )
        rs = Mo @ rg
    else:
        Omg = OMG0 + OMGd * dt - omge * (toes + dt)
        sOmg = np.sin(Omg)
        cOmg = np.cos(Omg)
        p = np.array((cOmg, sOmg, 0.0), dtype=np.float64)
        q = np.array((-ci * sOmg, ci * cOmg, si), dtype=np.float64)
        rs = np.array(
            (
                xo[0] * p[0] + xo[1] * q[0],
                xo[0] * p[1] + xo[1] * q[1],
                xo[0] * p[2] + xo[1] * q[2],
            ),
            dtype=np.float64,
        )

    dtrel = -2.0 * sqrt_mu_A * ecc * sE / (CLIGHT * CLIGHT)
    dts = af0 + af1 * dtc + af2 * dtc * dtc + dtrel

    vs = np.zeros(3, dtype=np.float64)
    if compute_vel:
        Ed = n / nue
        e_sqrt = np.sqrt(1.0 - ecc * ecc)
        nud = e_sqrt / nue * Ed
        h = np.array((cos_u, sin_u), dtype=np.float64)
        h2d = 2.0 * nud * np.array((-h[1], h[0]), dtype=np.float64)
        ud = nud + cuc * h2d[0] + cus * h2d[1]
        rd = Ak * ecc * sE * Ed + crc * h2d[0] + crs * h2d[1]
        hd = np.array((-h[1], h[0]), dtype=np.float64)
        xod = rd * h + (r * ud) * hd
        incd = idot + cic * h2d[0] + cis * h2d[1]
        omegd = OMGd - omge
        pd = np.array((-p[1], p[0], 0.0), dtype=np.float64) * omegd
        qd = (
            np.array((-q[1], q[0], 0.0), dtype=np.float64) * omegd
            + np.array((si * sOmg, -si * cOmg, ci), dtype=np.float64) * incd
        )
        vs = pd * xo[0] + qd * xo[1] + p * xod[0] + q * xod[1]

    return rs, vs, dts
