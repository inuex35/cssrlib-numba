"""Numba-accelerated geometric helpers used throughout cssrlib."""

from __future__ import annotations

import numpy as np
from numba import njit

from cssrlib.constants import CLIGHT, OMGE, RE_WGS84, FE_WGS84


@njit(cache=True)
def ecef2llh(ecef: np.ndarray) -> np.ndarray:
    """Convert ECEF (meters) to geodetic latitude/longitude/height."""

    e2 = FE_WGS84 * (2.0 - FE_WGS84)
    r2 = ecef[0] * ecef[0] + ecef[1] * ecef[1]
    v = RE_WGS84
    z = ecef[2]
    for _ in range(1000):
        zk = z
        sp = z / np.sqrt(r2 + z * z)
        v = RE_WGS84 / np.sqrt(1.0 - e2 * sp * sp)
        z = ecef[2] + v * e2 * sp
        if abs(z - zk) < 1e-4:
            break
    lat = np.arctan(z / np.sqrt(r2))
    lon = np.arctan2(ecef[1], ecef[0])
    h = np.sqrt(r2 + z * z) - v
    return np.array((lat, lon, h), dtype=np.float64)


@njit(cache=True)
def xyz2enu_matrix(llh: np.ndarray) -> np.ndarray:
    """Return the rotation matrix from ECEF to ENU."""

    sp = np.sin(llh[0])
    cp = np.cos(llh[0])
    sl = np.sin(llh[1])
    cl = np.cos(llh[1])
    return np.array(
        [
            (-sl, cl, 0.0),
            (-sp * cl, -sp * sl, cp),
            (cp * cl, cp * sl, sp),
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def enu2xyz_matrix(llh: np.ndarray) -> np.ndarray:
    """Return the rotation matrix from ENU to ECEF."""

    E = xyz2enu_matrix(llh)
    return np.linalg.inv(E)


@njit(cache=True)
def ecef2enu(llh: np.ndarray, delta_ecef: np.ndarray) -> np.ndarray:
    """Convert a delta vector in ECEF to ENU coordinates."""

    E = xyz2enu_matrix(llh)
    return E @ delta_ecef


@njit(cache=True)
def geodist(rs: np.ndarray, rr: np.ndarray) -> tuple[float, np.ndarray]:
    """Return geometric distance and LOS unit vector."""

    los = rs - rr
    rng = np.linalg.norm(los)
    if rng <= 0.0:
        return 0.0, np.zeros(3, dtype=np.float64)
    los /= rng
    earth_rot = OMGE * (rs[0] * rr[1] - rs[1] * rr[0]) / CLIGHT
    return rng + earth_rot, los


@njit(cache=True)
def satazel(llh: np.ndarray, los_ecef: np.ndarray) -> tuple[float, float]:
    """Return azimuth/elevation for a given LOS vector."""

    enu = ecef2enu(llh, los_ecef)
    az = np.arctan2(enu[0], enu[1])
    el = np.arcsin(enu[2] / np.linalg.norm(enu))
    return az, el
