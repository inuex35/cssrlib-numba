"""Broadcast Keplerian orbit propagation, compiled with Numba.

The inner arithmetic of :func:`cssrlib.models.ephemeris.eph2pos`, for
GPS / Galileo / QZSS / BeiDou. It is here rather than beside its caller for
the reason every kernel is: it takes scalars and returns arrays, while
``eph2pos`` handles an ``Eph`` object Numba cannot see.

The wide argument list is the cost of that split, and it is what let this
kernel drift out of sync while nothing called it -- it had lost the ``Adot``
and ``delnd`` velocity terms and derived ``h2d`` from the wrong basis. The
adapter in ``eph2pos`` is now the only caller and the only place those
scalars are assembled, so there is one expression of each.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.core.constants import CLIGHT, COS_5, SIN_5

#: Kepler equation iteration budget and convergence tolerance. Shared with
#: models.ephemeris so the broadcast orbit and the standalone anomaly solver
#: cannot disagree about when they have converged.
MAX_ITER_KEPLER = 30
RTOL_KEPLER = 1e-13


@njit(cache=True)
def broadcast_orbit(
    dt: float,
    dtc: float,
    n: float,
    nd: float,
    Ak: float,
    Akd: float,
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
    """Broadcast orbit position, velocity and clock offset from scalars.

    ``n``/``nd`` and ``Ak``/``Akd`` are the mean motion and semi-major axis
    with their rates: CNAV ephemerides carry ``delnd`` and ``Adot``, and both
    contribute to the velocity. ``sqrt_mu_A`` is ``sqrt(mu * A)`` for the
    relativistic clock term, from the broadcast ``A`` rather than ``Ak``.
    """

    E = M
    for _ in range(MAX_ITER_KEPLER):
        E_prev = E
        E = M + ecc * np.sin(E)
        if abs(E - E_prev) < RTOL_KEPLER:
            break

    sE = np.sin(E)
    cE = np.cos(E)
    nue = 1.0 - ecc * cE
    e_sqrt = np.sqrt(1.0 - ecc * ecc)
    nu = np.arctan2(e_sqrt * sE, cE - ecc)

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
    xo0 = r * cos_u
    xo1 = r * sin_u

    # A BeiDou GEO is propagated in a frame that is not spun with the Earth
    # and then rotated in; everything else takes the Earth rotation inside
    # the node.
    if is_bds_geo:
        Omg = OMG0 + OMGd * dt - omge * toes
    else:
        Omg = OMG0 + OMGd * dt - omge * (toes + dt)
    sOmg = np.sin(Omg)
    cOmg = np.cos(Omg)
    p = np.array((cOmg, sOmg, 0.0), dtype=np.float64)
    q = np.array((-ci * sOmg, ci * cOmg, si), dtype=np.float64)

    rs = xo0 * p + xo1 * q
    if is_bds_geo:
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
        rs = Mo @ rs

    dts = af0 + af1 * dtc + af2 * dtc * dtc \
        - 2.0 * sqrt_mu_A * ecc * sE / (CLIGHT * CLIGHT)

    vs = np.zeros(3, dtype=np.float64)
    if compute_vel:
        Ed = (n + nd) / nue
        nud = e_sqrt / nue * Ed
        # d/dt of (cos 2phi, sin 2phi): the harmonic corrections are indexed
        # by 2phi, so the rotation is of that basis, not of (cos u, sin u).
        h2d0 = -2.0 * nud * sin2
        h2d1 = 2.0 * nud * cos2

        ud = nud + cuc * h2d0 + cus * h2d1
        rd = Akd * nue + Ak * ecc * sE * Ed + crc * h2d0 + crs * h2d1
        xod0 = rd * cos_u - (r * ud) * sin_u
        xod1 = rd * sin_u + (r * ud) * cos_u

        incd = idot + cic * h2d0 + cis * h2d1
        omegd = OMGd - omge
        pd = np.array((-p[1], p[0], 0.0), dtype=np.float64) * omegd
        qd = (np.array((-q[1], q[0], 0.0), dtype=np.float64) * omegd
              + np.array((si * sOmg, -si * cOmg, ci), dtype=np.float64) * incd)
        vs = pd * xo0 + qd * xo1 + p * xod0 + q * xod1

    return rs, vs, dts


__all__ = ["broadcast_orbit", "MAX_ITER_KEPLER", "RTOL_KEPLER"]
