"""
module for PPP processing
"""
import cssrlib.gnss as gn
from cssrlib.peph import gpst2utc, time2epoch
from enum import IntEnum
from math import sin, cos, atan2, asin
import numpy as np
from numba import njit
try:
    from pysolid.solid import solid_grid
except ImportError:
    solid_grid = None


_SEC2RAD = float(gn.rCST.AS2R)
_TWO_PI = float(2.0*np.pi)
_AU = float(gn.rCST.AU)
_RE_WGS84 = float(gn.rCST.RE_WGS84)
_OMGE = float(gn.rCST.OMGE)
_GME = float(gn.rCST.GME)
_GMS = float(gn.rCST.GMS)
_GMM = float(gn.rCST.GMM)
_CLIGHT = float(gn.rCST.CLIGHT)

_AST_ARGS_COEFF = np.array([
    [134.96340251, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
    [357.52910918, 129596581.0481, -0.5532, 0.000136, -0.00001149],
    [93.27209062, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
    [297.85019547, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
    [125.04455501, -6962890.2665, 7.4722, 0.007702, -0.00005939],
], dtype=np.float64)

_NUTATION_COEFF = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0, -6798.4, -171996.0, -174.2, 92025.0, 8.9],
    [0.0, 0.0, 2.0, -2.0, 2.0, 182.6, -13187.0, -1.6, 5736.0, -3.1],
    [0.0, 0.0, 2.0, 0.0, 2.0, 13.7, -2274.0, -0.2, 977.0, -0.5],
    [0.0, 0.0, 0.0, 0.0, 2.0, -3399.2, 2062.0, 0.2, -895.0, 0.5],
    [0.0, -1.0, 0.0, 0.0, 0.0, -365.3, -1426.0, 3.4, 54.0, -0.1],
    [1.0, 0.0, 0.0, 0.0, 0.0, 27.6, 712.0, 0.1, -7.0, 0.0],
    [0.0, 1.0, 2.0, -2.0, 2.0, 121.7, -517.0, 1.2, 224.0, -0.6],
    [0.0, 0.0, 2.0, 0.0, 1.0, 13.6, -386.0, -0.4, 200.0, 0.0],
    [1.0, 0.0, 2.0, 0.0, 2.0, 9.1, -301.0, 0.0, 129.0, -0.1],
    [0.0, -1.0, 2.0, -2.0, 2.0, 365.2, 217.0, -0.5, -95.0, 0.3],
    [-1.0, 0.0, 0.0, 2.0, 0.0, 31.8, 158.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 2.0, -2.0, 1.0, 177.8, 129.0, 0.1, -70.0, 0.0],
    [-1.0, 0.0, 2.0, 0.0, 2.0, 27.1, 123.0, 0.0, -53.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 27.7, 63.0, 0.1, -33.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 14.8, 63.0, 0.0, -2.0, 0.0],
    [-1.0, 0.0, 2.0, 2.0, 2.0, 9.6, -59.0, 0.0, 26.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 1.0, -27.4, -58.0, -0.1, 32.0, 0.0],
    [1.0, 0.0, 2.0, 0.0, 1.0, 9.1, -51.0, 0.0, 27.0, 0.0],
    [-2.0, 0.0, 0.0, 2.0, 0.0, -205.9, -48.0, 0.0, 1.0, 0.0],
    [-2.0, 0.0, 2.0, 0.0, 1.0, 1305.5, 46.0, 0.0, -24.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 2.0, 7.1, -38.0, 0.0, 16.0, 0.0],
    [2.0, 0.0, 2.0, 0.0, 2.0, 6.9, -31.0, 0.0, 13.0, 0.0],
    [2.0, 0.0, 0.0, 0.0, 0.0, 13.8, 29.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 2.0, -2.0, 2.0, 23.9, 29.0, 0.0, -12.0, 0.0],
    [0.0, 0.0, 2.0, 0.0, 0.0, 13.6, 26.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 2.0, -2.0, 0.0, 173.3, -22.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 2.0, 0.0, 1.0, 27.0, 21.0, 0.0, -10.0, 0.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 182.6, 17.0, -0.1, 0.0, 0.0],
    [0.0, 2.0, 2.0, -2.0, 2.0, 91.3, -16.0, 0.1, 7.0, 0.0],
    [-1.0, 0.0, 0.0, 2.0, 1.0, 32.0, 16.0, 0.0, -8.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 386.0, -15.0, 0.0, 9.0, 0.0],
    [1.0, 0.0, 0.0, -2.0, 1.0, -31.7, -13.0, 0.0, 7.0, 0.0],
    [0.0, -1.0, 0.0, 0.0, 1.0, -346.6, -12.0, 0.0, 6.0, 0.0],
    [2.0, 0.0, -2.0, 0.0, 0.0, -1095.2, 11.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 2.0, 2.0, 1.0, 9.5, -10.0, 0.0, 5.0, 0.0],
    [1.0, 0.0, 2.0, 2.0, 2.0, 5.6, -8.0, 0.0, 3.0, 0.0],
    [0.0, -1.0, 2.0, 0.0, 2.0, 14.2, -7.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 1.0, 7.1, -7.0, 0.0, 3.0, 0.0],
    [1.0, 1.0, 0.0, -2.0, 0.0, -34.8, -7.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 2.0, 0.0, 2.0, 13.2, 7.0, 0.0, -3.0, 0.0],
    [-2.0, 0.0, 0.0, 2.0, 1.0, -199.8, -6.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 1.0, 14.8, -6.0, 0.0, 3.0, 0.0],
    [2.0, 0.0, 2.0, -2.0, 2.0, 12.8, 6.0, 0.0, -3.0, 0.0],
    [1.0, 0.0, 0.0, 2.0, 0.0, 9.6, 6.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 2.0, -2.0, 1.0, 23.9, 6.0, 0.0, -3.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 1.0, -14.7, -5.0, 0.0, 3.0, 0.0],
    [0.0, -1.0, 2.0, -2.0, 1.0, 346.6, -5.0, 0.0, 3.0, 0.0],
    [2.0, 0.0, 2.0, 0.0, 1.0, 6.9, -5.0, 0.0, 3.0, 0.0],
    [1.0, -1.0, 0.0, 0.0, 0.0, 29.8, 5.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, -1.0, 0.0, 411.8, -4.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 29.5, -4.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -2.0, 0.0, -15.4, -4.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -2.0, 0.0, 0.0, -26.9, 4.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, -2.0, 1.0, 212.3, 4.0, 0.0, -2.0, 0.0],
    [0.0, 1.0, 2.0, -2.0, 1.0, 119.6, 4.0, 0.0, -2.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 25.6, -3.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, -1.0, 0.0, -3232.9, -3.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, 2.0, 2.0, 2.0, 9.8, -3.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 2.0, 2.0, 2.0, 7.2, -3.0, 0.0, 1.0, 0.0],
    [1.0, -1.0, 2.0, 0.0, 2.0, 9.4, -3.0, 0.0, 1.0, 0.0],
    [3.0, 0.0, 2.0, 0.0, 2.0, 5.5, -3.0, 0.0, 1.0, 0.0],
    [-2.0, 0.0, 2.0, 0.0, 2.0, 1615.7, -3.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 2.0, 0.0, 0.0, 9.1, 3.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 2.0, 4.0, 2.0, 5.8, -2.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 2.0, 27.8, -2.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 2.0, -2.0, 1.0, -32.6, -2.0, 0.0, 1.0, 0.0],
    [0.0, -2.0, 2.0, -2.0, 1.0, 6786.3, -2.0, 0.0, 1.0, 0.0],
    [-2.0, 0.0, 0.0, 0.0, 1.0, -13.7, -2.0, 0.0, 1.0, 0.0],
    [2.0, 0.0, 0.0, 0.0, 1.0, 13.8, 2.0, 0.0, -1.0, 0.0],
    [3.0, 0.0, 0.0, 0.0, 0.0, 9.2, 2.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 2.0, 0.0, 2.0, 8.9, 2.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 2.0, 1.0, 2.0, 9.3, 2.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 2.0, 1.0, 9.6, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 2.0, 2.0, 1.0, 5.6, -1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, -2.0, 1.0, -34.7, -1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 2.0, 0.0, 14.2, -1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 2.0, -2.0, 0.0, 117.5, -1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -2.0, 2.0, 0.0, -329.8, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -2.0, 2.0, 0.0, 23.8, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -2.0, -2.0, 0.0, -9.5, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 2.0, -2.0, 0.0, 32.8, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, -4.0, 0.0, -10.1, -1.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, -4.0, 0.0, -15.9, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 4.0, 2.0, 4.8, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, -1.0, 2.0, 25.4, -1.0, 0.0, 0.0, 0.0],
    [-2.0, 0.0, 2.0, 4.0, 2.0, 7.3, -1.0, 0.0, 1.0, 0.0],
    [2.0, 0.0, 2.0, 2.0, 2.0, 4.7, -1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 2.0, 0.0, 1.0, 14.2, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -2.0, 0.0, 1.0, -13.6, -1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 4.0, -2.0, 2.0, 12.7, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 2.0, 409.2, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 2.0, -2.0, 2.0, 22.5, 1.0, 0.0, -1.0, 0.0],
    [3.0, 0.0, 2.0, -2.0, 2.0, 8.7, 1.0, 0.0, 0.0, 0.0],
    [-2.0, 0.0, 2.0, 2.0, 2.0, 14.6, 1.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 2.0, -27.3, 1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, -2.0, 2.0, 1.0, -169.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 2.0, 0.0, 1.0, 13.1, 1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 4.0, 0.0, 2.0, 9.1, 1.0, 0.0, 0.0, 0.0],
    [2.0, 1.0, 0.0, -2.0, 0.0, 131.7, 1.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 2.0, 0.0, 7.1, 1.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 2.0, -2.0, 1.0, 12.8, 1.0, 0.0, -1.0, 0.0],
    [2.0, 0.0, -2.0, 0.0, 1.0, -943.2, 1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, -2.0, 0.0, -29.3, 1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0, 1.0, -388.3, 1.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, 0.0, 2.0, 1.0, 35.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 27.3, 1.0, 0.0, 0.0, 0.0],
], dtype=np.float64)


@njit(cache=True)
def ast_args(t_):
    tt = np.zeros(4, dtype=np.float64)
    f = np.zeros(5, dtype=np.float64)
    tt[0] = t_
    for idx in range(3):
        tt[idx+1] = tt[idx]*t_
    for row in range(_AST_ARGS_COEFF.shape[0]):
        accum = _AST_ARGS_COEFF[row, 0]*3600.0
        for col in range(1, 5):
            accum += _AST_ARGS_COEFF[row, col]*tt[col-1]
        angle = np.fmod(accum*_SEC2RAD, _TWO_PI)
        f[row] = angle
    return f


@njit(cache=True)
def nut_iau1980(t_, f):
    dpsi = 0.0
    deps = 0.0
    for i in range(_NUTATION_COEFF.shape[0]):
        ang = 0.0
        for j in range(5):
            ang += _NUTATION_COEFF[i, j]*f[j]
        s_ang = np.sin(ang)
        c_ang = np.cos(ang)
        dpsi += (_NUTATION_COEFF[i, 6]+_NUTATION_COEFF[i, 7]*t_)*s_ang
        deps += (_NUTATION_COEFF[i, 8]+_NUTATION_COEFF[i, 9]*t_)*c_ang
    scale = _SEC2RAD*1e-4
    return dpsi*scale, deps*scale


@njit(cache=True)
def _windupcorr(rs, vs, rr, E, phw):
    we = np.array((0.0, 0.0, _OMGE), dtype=np.float64)
    diff = rr - rs
    norm_diff = np.linalg.norm(diff)
    if norm_diff <= 0.0:
        return phw
    ek = diff / norm_diff
    ezs_vec = -rs
    ezs_norm = np.linalg.norm(ezs_vec)
    if ezs_norm <= 0.0:
        return phw
    ezs = ezs_vec / ezs_norm
    ess_vec = vs + np.cross(we, rs)
    ess_norm = np.linalg.norm(ess_vec)
    if ess_norm <= 0.0:
        return phw
    ess = ess_vec / ess_norm
    eys_vec = np.cross(ezs, ess)
    eys_norm = np.linalg.norm(eys_vec)
    if eys_norm <= 0.0:
        return phw
    eys = eys_vec / eys_norm
    exs = np.cross(eys, ezs)
    exr = E[0, :]
    eyr = E[1, :]
    eks = np.cross(ek, eys)
    ekr = np.cross(ek, eyr)
    ds = exs - ek * np.dot(ek, exs) - eks
    dr = exr - ek * np.dot(ek, exr) + ekr
    ds_norm = np.linalg.norm(ds)
    dr_norm = np.linalg.norm(dr)
    if ds_norm <= 0.0 or dr_norm <= 0.0:
        return phw
    c_p = np.dot(ds, dr) / (ds_norm * dr_norm)
    if c_p > 1.0:
        c_p = 1.0
    elif c_p < -1.0:
        c_p = -1.0
    ph = np.arccos(c_p) / (2.0 * np.pi)
    drs = np.cross(ds, dr)
    if np.dot(ek, drs) < 0.0:
        ph = -ph
    return ph + np.floor(phw - ph + 0.5)


@njit(cache=True)
def _windupcorr_full(rs, vs, rr, E, rsun, phw):
    diff = rr - rs
    norm_diff = np.linalg.norm(diff)
    if norm_diff <= 0.0:
        return phw
    ek = diff / norm_diff
    ezs_vec = -rs
    ezs_norm = np.linalg.norm(ezs_vec)
    if ezs_norm <= 0.0:
        return phw
    ezs = ezs_vec / ezs_norm
    ess_vec = rsun - rs
    ess_norm = np.linalg.norm(ess_vec)
    if ess_norm <= 0.0:
        return phw
    ess = ess_vec / ess_norm
    eys_vec = np.cross(ezs, ess)
    eys_norm = np.linalg.norm(eys_vec)
    if eys_norm <= 0.0:
        return phw
    eys = eys_vec / eys_norm
    exs = np.cross(eys, ezs)
    exr = E[0, :]
    eyr = E[1, :]
    eks = np.cross(ek, eys)
    ekr = np.cross(ek, eyr)
    ds = exs - ek * np.dot(ek, exs) - eks
    dr = exr - ek * np.dot(ek, exr) + ekr
    ds_norm = np.linalg.norm(ds)
    dr_norm = np.linalg.norm(dr)
    if ds_norm <= 0.0 or dr_norm <= 0.0:
        return phw
    c_p = np.dot(ds, dr) / (ds_norm * dr_norm)
    if c_p > 1.0:
        c_p = 1.0
    elif c_p < -1.0:
        c_p = -1.0
    ph = np.arccos(c_p) / (2.0 * np.pi)
    drs = np.cross(ds, dr)
    if np.dot(ek, drs) < 0.0:
        ph = -ph
    return ph + np.floor(phw - ph + 0.5)


@njit(cache=True)
def _sun_moon_eci(t_):
    f = ast_args(t_)
    deg2rad = np.pi/180.0
    eps = (23.439291-0.0130042*t_)*deg2rad
    c_e = np.cos(eps)
    s_e = np.sin(eps)
    Ms = (357.5277233+35999.05034*t_)*deg2rad
    ls = (280.460+36000.770*t_+1.914666471*np.sin(Ms) +
          0.019994643*np.sin(2.0*Ms))*deg2rad
    rs = _AU*(1.000140612-0.016708617*np.cos(Ms)-0.000139589*np.cos(2.0*Ms))
    rsun_eci = np.array(
        [rs*np.cos(ls), rs*c_e*np.sin(ls), rs*s_e*np.sin(ls)],
        dtype=np.float64,
    )

    lm = (218.32+481267.883*t_+6.29*np.sin(f[0])-1.27*np.sin(f[0]-2.0*f[3]) +
          0.66*np.sin(2.0*f[3])+0.21*np.sin(2.0*f[0])-0.19*np.sin(f[1]) -
          0.11*np.sin(2.0*f[2]))*deg2rad
    pm = (5.13*np.sin(f[2])+0.28*np.sin(f[0]+f[2])-0.28*np.sin(f[2]-f[0]) -
          0.17*np.sin(f[2]-2.0*f[3]))*deg2rad
    u = (0.9508+0.0518*np.cos(f[0])+0.0095*np.cos(f[0]-2.0*f[3]) +
         0.0078*np.cos(2.0*f[3])+0.0028*np.cos(2.0*f[0]))*deg2rad
    rm = _RE_WGS84/np.sin(u)
    c_l = np.cos(lm)
    s_l = np.sin(lm)
    c_p = np.cos(pm)
    s_p = np.sin(pm)
    rmoon_eci = rm*np.array(
        [c_p*c_l, c_e*c_p*s_l-s_e*s_p, s_e*c_p*s_l+c_e*s_p],
        dtype=np.float64,
    )
    return rsun_eci, rmoon_eci


def Rx(t_):
    """ x-axis rotation matrix """
    c = cos(t_)
    s = sin(t_)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def Ry(t_):
    """ y-axis rotation matrix """
    c = cos(t_)
    s = sin(t_)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def Rz(t_):
    """ z-axis rotation matrix """
    c = cos(t_)
    s = sin(t_)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def utc2gmst(t_, ut1_utc=0):
    """ UTC to GMST """
    ep0 = gn.epoch2time([2000, 1, 1, 12, 0, 0])
    tut = gn.timeadd(t_, ut1_utc)
    ep = gn.time2epoch(tut)
    tut0 = gn.epoch2time([ep[0], ep[1], ep[2], 0, 0, 0])
    ut = ep[3]*3600+ep[4]*60+ep[5]
    t1 = gn.timediff(tut0, ep0)/gn.rCST.CENTURY_SEC
    t2 = t1**2
    t3 = t2*t1
    gmst0 = 24110.54841+8640184.812866*t1+0.093104*t2-6.2e-6*t3
    gmst = gmst0+1.002737909350795*ut
    return np.fmod(gmst, gn.rCST.DAY_SEC)*(2.0*np.pi/gn.rCST.DAY_SEC)


@njit(cache=True)
def _eci2ecef(t_, erpv, gmst):
    t2 = t_*t_
    t3 = t2*t_
    f = ast_args(t_)
    ze = (2306.2181*t_+0.30188*t2+0.017998*t3)*_SEC2RAD
    th = (2004.3109*t_-0.42665*t2-0.041833*t3)*_SEC2RAD
    z = (2306.2181*t_+1.09468*t2+0.018203*t3)*_SEC2RAD
    eps = (84381.448-46.8150*t_-0.00059*t2+0.001813*t3)*_SEC2RAD

    cze = np.cos(-ze)
    sze = np.sin(-ze)
    Rz1 = np.array([[cze, sze, 0.0], [-sze, cze, 0.0], [0.0, 0.0, 1.0]])
    cth = np.cos(th)
    sth = np.sin(th)
    Ry_mat = np.array([[cth, 0.0, -sth], [0.0, 1.0, 0.0], [sth, 0.0, cth]])
    czz = np.cos(-z)
    szz = np.sin(-z)
    Rz2 = np.array([[czz, szz, 0.0], [-szz, czz, 0.0], [0.0, 0.0, 1.0]])
    P = Rz2 @ Ry_mat @ Rz1

    dpsi, deps = nut_iau1980(t_, f)
    Rx1 = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-eps-deps), np.sin(-eps-deps)], [0.0, -np.sin(-eps-deps), np.cos(-eps-deps)]])
    cpsi = np.cos(-dpsi)
    spsi = np.sin(-dpsi)
    Rz3 = np.array([[cpsi, spsi, 0.0], [-spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    cpe = np.cos(eps)
    spe = np.sin(eps)
    Rx2 = np.array([[1.0, 0.0, 0.0], [0.0, cpe, spe], [0.0, -spe, cpe]])
    N = Rx1 @ Rz3 @ Rx2

    gast = gmst+dpsi*np.cos(eps)
    gast += (0.00264*np.sin(f[4])+0.000063*np.sin(2.0*f[4]))*_SEC2RAD

    sE = np.sin(-erpv[0])
    cE = np.cos(-erpv[0])
    RyE = np.array([[cE, 0.0, -sE], [0.0, 1.0, 0.0], [sE, 0.0, cE]])
    sP = np.sin(-erpv[1])
    cP = np.cos(-erpv[1])
    RxP = np.array([[1.0, 0.0, 0.0], [0.0, cP, sP], [0.0, -sP, cP]])

    cg = np.cos(gast)
    sg = np.sin(gast)
    Rzg = np.array([[cg, sg, 0.0], [-sg, cg, 0.0], [0.0, 0.0, 1.0]])

    W = RyE @ RxP
    U = W @ Rzg @ N @ P
    return U


def eci2ecef(tgps, erpv):
    """ ECI to ECEF conversion matrix """
    tutc = gn.gpst2utc(tgps)
    ep0 = gn.epoch2time([2000, 1, 1, 12, 0, 0])
    dt = gn.timediff(tgps, ep0)
    t_ = (dt+19+32.184)/gn.rCST.CENTURY_SEC
    gmst = utc2gmst(tutc, erpv[2])
    erpv_arr = np.asarray(erpv, dtype=np.float64)
    U = _eci2ecef(float(t_), erpv_arr, float(gmst))
    return U, gmst


def sunmoonpos(tutc, erpv=np.zeros(5)):
    """ calculate sun/moon position in ECEF """
    tut = gn.timeadd(tutc, erpv[2])
    ep0 = gn.epoch2time([2000, 1, 1, 12, 0, 0])
    t_ = gn.timediff(tut, ep0)/gn.rCST.CENTURY_SEC
    rsun_eci, rmoon_eci = _sun_moon_eci(float(t_))
    U, gmst = eci2ecef(tutc, erpv)
    rsun = U@rsun_eci
    rmoon = U@rmoon_eci

    return rsun, rmoon, gmst


def shapiro(rsat, rrcv):
    """ relativistic shapiro effect """
    rs_arr = np.asarray(rsat, dtype=np.float64)
    rr_arr = np.asarray(rrcv, dtype=np.float64)
    return _shapiro_numba(rs_arr, rr_arr)


def windupcorr(time, rs, vs, rr, phw, full=False):
    """ calculate windup correction """
    ek = gn.vnorm(rr-rs)
    pos = gn.ecef2pos(rr)
    E = gn.xyz2enu(pos)
    if full:
        rsun, _, _ = sunmoonpos(gpst2utc(time))
        return _windupcorr_full(
            np.asarray(rs, dtype=np.float64),
            np.asarray(vs, dtype=np.float64),
            np.asarray(rr, dtype=np.float64),
            np.asarray(E, dtype=np.float64),
            np.asarray(rsun, dtype=np.float64),
            float(phw),
        )
    return _windupcorr(
        np.asarray(rs, dtype=np.float64),
        np.asarray(vs, dtype=np.float64),
        np.asarray(rr, dtype=np.float64),
        np.asarray(E, dtype=np.float64),
        float(phw),
    )


class uTideModel(IntEnum):
    """
    Enumeration for Earth tide model selection
    """

    NONE = -1
    SIMPLE = 0
    IERS2010 = 1


def tide_pl(eu, rp, GMp, pos):
    """ pole tide correction """
    return _tide_pl(
        np.asarray(eu, dtype=np.float64),
        np.asarray(rp, dtype=np.float64),
        float(GMp),
        np.asarray(pos, dtype=np.float64),
    )


def solid_tide(rsun, rmoon, pos, E, gmst, flag=True):
    """ solid earth tide correction """
    return _solid_tide(
        np.asarray(rsun, dtype=np.float64),
        np.asarray(rmoon, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(E, dtype=np.float64),
        float(gmst),
        int(flag),
    )


def tidedisp(tutc, pos, erpv=None):
    """ displacement by tide """
    if erpv is None:
        erpv = np.zeros(5)
    rs, rm, gmst = sunmoonpos(tutc, erpv)
    E = gn.xyz2enu(pos)
    return _solid_tide(
        np.asarray(rs, dtype=np.float64),
        np.asarray(rm, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(E, dtype=np.float64),
        float(gmst),
        1,
    )


def tidedispIERS2010(tutc, pos, erpv=None):
    """
    Wrapper for solid_grid() method of PySolid module to compute Earth tide
    displacement corrections according to the IERS2010 conventions
    """
    if solid_grid is None:  # workaround for missing PySolid
        return tidedisp(tutc, pos, erpv)

    e = time2epoch(tutc)
    disp_e, disp_n, disp_u = solid_grid(e[0], e[1], e[2], e[3], e[4],
                                        int(e[5]),
                                        np.rad2deg(pos[0]), 0, 1,
                                        np.rad2deg(pos[1]), 0, 1)
    E = gn.enu2xyz(pos)
    return E@np.array([disp_e[0, 0], disp_n[0, 0], disp_u[0, 0]])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cssrlib.ephemeris import findeph, eph2pos
    from cssrlib.rinex import rnxdec

    tgps_ = gn.epoch2time([2021, 3, 19, 0, 0, 0])
    pos_ = np.array([0.61678759,  2.43512138, 64.94054687])
    erpv_ = np.array([2.1079217879069683e-06, 4.8733853217911866e-07,
                     -0.044509672541668682, -0.0007141, 0])

    flg_tide = False
    flg_pwup = True

    if flg_tide:
        n = 86400//300
        t = np.zeros(n)
        dr_ = np.zeros((n, 3))
        for k in range(n):
            tn = gn.timeadd(tgps_, k*300)
            t[k] = gn.timediff(tn, tgps_)
            dn_ = tidedisp(gn.gpst2utc(tn), pos_, erpv_)
            dr_[k, :] = gn.ecef2enu(pos_, dn_)

        plt.figure()
        plt.plot(t/3600, dr_)
        plt.xlabel('time [h]')
        plt.ylabel('displacement [m]')
        plt.grid()
        plt.axis([0, 24, -0.2, 0.2])
        plt.legend(('east', 'north', 'up'))

    if flg_pwup:
        bdir = '../data/'
        navfile = bdir+'30340780.21q'
        nav = gn.Nav()
        dec = rnxdec()
        nav = dec.decode_nav(navfile, nav)
        rr_ = gn.pos2ecef(pos_)
        sat = gn.prn2sat(gn.uGNSS.QZS, 194)

        n = 86400//300
        t = np.zeros(n)
        ph_ = np.zeros(n)
        d = np.zeros(n)
        phw_ = 0
        for k in range(n):
            tn = gn.timeadd(tgps_, k*300)
            eph = findeph(nav.eph, tn, sat)
            rs_, vs_, dts = eph2pos(tn, eph, True)
            phw_ = windupcorr(tn, rs_, vs_, rr_, phw_)
            t[k] = gn.timediff(tn, tgps_)
            ph_[k] = phw_
            d[k] = shapiro(rs_, rr_)

        plt.figure()
        plt.plot(t/3600, ph_, label='phase windup')
        plt.plot(t/3600, d, label='shapiro')
        plt.xlabel('time [h]')
        plt.ylabel('delta range [m]')
        plt.grid()
        plt.axis([0, 24, -0.2, 0.2])
        plt.legend()
@njit(cache=True)
def _shapiro_numba(rsat, rrcv):
    rs = np.linalg.norm(rsat)
    rr = np.linalg.norm(rrcv)
    rrs = np.linalg.norm(rsat-rrcv)
    corr = (2*_GME/_CLIGHT**2)*np.log((rs+rr+rrs)/(rs+rr-rrs))
    return corr
@njit(cache=True)
def _tide_pl(eu, rp, GMp, pos):
    H3 = 0.293
    L3 = 0.0156
    r = np.linalg.norm(rp)
    if r <= 0.0:
        return np.zeros(3, dtype=np.float64)
    ep = rp/r
    K2 = GMp/_GME*_RE_WGS84**4/r**3
    K3 = K2*_RE_WGS84/r
    latp = np.arcsin(ep[2])
    lonp = np.arctan2(ep[1], ep[0])
    c_p = np.cos(latp)
    c_l = np.cos(pos[0])
    s_l = np.sin(pos[0])

    p = (3.0*s_l**2-1.0)/2.0
    H2 = 0.6078-0.0006*p
    L2 = 0.0847+0.0002*p
    a = np.dot(ep, eu)
    a2 = a*a
    dp = K2*3.0*L2*a
    du = K2*(H2*(1.5*a2-0.5)-3.0*L2*a2)

    dp += K3*L3*(7.5*a2-1.5)
    du += K3*a*(H3*(2.5*a2-1.5)-L3*(7.5*a2-1.5))
    dlon = pos[1]-lonp
    du += 0.75*0.0025*K2*np.sin(2.0*latp)*np.sin(2.0*pos[0])*np.sin(dlon)
    du += 0.75*0.0022*K2*(c_p*c_l)**2*np.sin(2.0*dlon)

    return dp*ep+du*eu
@njit(cache=True)
def _solid_tide(rsun, rmoon, pos, E, gmst, flag):
    eu = E[2, :]
    dr1 = _tide_pl(eu, rsun, _GMS, pos)
    dr2 = _tide_pl(eu, rmoon, _GMM, pos)
    s_2l = np.sin(2.0*pos[0])
    du = -0.012*s_2l*np.sin(gmst+pos[1])
    dr = dr1+dr2+du*eu
    if flag:
        s_l = np.sin(pos[0])
        du = 0.1196*(1.5*s_l**2-0.5)
        dn = 0.0247*s_2l
        dr += du*E[2, :]+dn*E[1, :]
    return dr
