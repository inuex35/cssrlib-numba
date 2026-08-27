"""Coordinate frames, geodesy and atmospheric model wrappers.

Thin wrappers over the Numba kernels in geometry / atmosphere, plus DOP and
ionospheric pierce-point geometry."""

from math import sin, cos, sqrt
import numpy as np
import bitstruct.c as bs

from cssrlib.core import geometry as _geom_fast
from cssrlib.core import atmosphere as _atm_fast
from cssrlib.domain.enums import *  # noqa: F401,F403
# star imports skip underscore names
from cssrlib.domain.enums import _ensure_vec
from cssrlib.domain.timescale import *  # noqa: F401,F403


def vnorm(r):
    """ calculate norm of a vector """
    return r/np.linalg.norm(r)


def geodist(rs, rr):
    """ calculate geometric distance """
    return _geom_fast.geodist(_ensure_vec(rs), _ensure_vec(rr))


def dops_h(H):
    """ calculate DOP from H """
    Qinv = np.linalg.inv(np.dot(H.T, H))
    dop = np.diag(Qinv)
    hdop = np.sqrt(dop[0]+dop[1])
    vdop = np.sqrt(dop[2])
    pdop = np.sqrt(np.sum(dop[0:3]))
    gdop = np.sqrt(np.sum(dop[0:4]))
    dop = np.array([gdop, pdop, hdop, vdop])
    return dop


def dops(az, el, elmin=0):
    """ calculate DOP from az/el """
    nm = az.shape[0]
    H = np.zeros((nm, 4))
    n = 0
    for i in range(nm):
        if el[i] < elmin:
            continue
        cel = cos(el[i])
        sel = sin(el[i])
        H[n, 0] = cel*sin(az[i])
        H[n, 1] = cel*cos(az[i])
        H[n, 2] = sel
        H[n, 3] = 1
        n += 1
    if n < 4:
        return None
    return dops_h(H)


def xyz2enu(pos):
    """ return ECEF to ENU conversion matrix from LLH """
    return _geom_fast.xyz2enu_matrix(_ensure_vec(pos))


def enu2xyz(pos):
    """ return ENU to ECEF conversion matrix from LLH """
    return _geom_fast.enu2xyz_matrix(_ensure_vec(pos))


def ecef2pos(r):
    """  ECEF to LLH position conversion """
    return _geom_fast.ecef2llh(_ensure_vec(r))


def pos2ecef(pos, isdeg: bool = False):
    """ LLH (rad/deg) to ECEF position conversion  """
    if isdeg:
        pos = [pos[0]*np.pi/180.0, pos[1]*np.pi/180.0, pos[2]]
    s_p = sin(pos[0])
    c_p = cos(pos[0])
    s_l = sin(pos[1])
    c_l = cos(pos[1])
    e2 = rCST.FE_WGS84*(2.0-rCST.FE_WGS84)
    v = rCST.RE_WGS84/sqrt(1.0-e2*s_p**2)
    r = np.array([(v+pos[2])*c_p*c_l,
                  (v+pos[2])*c_p*s_l,
                  (v*(1.0-e2)+pos[2])*s_p])
    return r


def ecef2enu(pos, r):
    """ relative ECEF to ENU conversion """
    return _geom_fast.ecef2enu(_ensure_vec(pos), _ensure_vec(r))


def satazel(pos, e):
    """ calculate az/el from LOS vector in ECEF (e) """
    return _geom_fast.satazel(_ensure_vec(pos), _ensure_vec(e))


def interpc(coef, lat):
    """ linear interpolation (lat step=15) """
    i = int(lat/15.0)
    if i < 1:
        return coef[:, 0]
    if i > 4:
        return coef[:, 4]
    d = lat/15.0-i
    return coef[:, i-1]*(1.0-d)+coef[:, i]*d


def tropmapf(t, pos, el, model=uTropoModel.SAAST):
    """
    Tropo mapping function
    """

    if model == uTropoModel.SAAST:
        return tropmapfNiell(t, pos, el)
    elif model == uTropoModel.HOPF:
        return tropmapfHpf(el)
    else:
        return 0


def tropmodel(t, pos, el=np.pi/2, humi=0.7, model=uTropoModel.SAAST):
    """
    Tropospheric delay model
    """

    if model == uTropoModel.SAAST:
        return tropmodelSaast(t, pos, el, humi)
    elif model == uTropoModel.HOPF:
        return tropmodelHpf()
    else:
        return 0


def meteo(hgt, humi):
    """ standard atmosphere model """
    return _atm_fast.meteo(float(hgt), float(humi))


def mapf(el, a, b, c):
    """ simple tropospheric mapping function """
    return _atm_fast.mapf(float(el), float(a), float(b), float(c))


def tropmapfNiell(t, pos, el):
    """ tropospheric mapping function by Niell (NMF) """
    doy = float(time2doy(t))
    return _atm_fast.tropmapf_niell(doy, _ensure_vec(pos), float(el))


def tropmodelSaast(t, pos, el=np.pi/2, humi=0.7):
    """ saastamoinen tropospheric delay model """
    return _atm_fast.tropmodel_saast(_ensure_vec(pos), float(el), float(humi))


def meteoHpf():
    """
    Hopfield atmosphere model
    """
    pres = 1010.0
    temp = 291.1
    e = 10.4
    return pres, temp, e


def tropmapfHpf(el):
    """
    Tropospheric mapping functions for Hopfield model
    """

    mapfh = 1.0/(np.sin(np.sqrt(el**2+(np.pi/72.0)**2)))
    mapfw = 1.0/(np.sin(np.sqrt(el**2+(np.pi/120.0)**2)))

    return mapfh, mapfw,


def tropmodelHpf():
    """
    Hopfield tropospheric delay model
    """
    # standard atmosphere for Hopfield model
    pres, temp, e = meteoHpf()
    # Hopfield
    trop_dry = (77.6e-6 * (-613.3768/temp+148.98)*pres)/5.0
    trop_wet = (77.6e-6 * 11000.0 * 4810.0 * e/temp**2)/5.0

    return trop_dry, trop_wet, None


def copy_buff(src, dst, ofst_s=0, ofst_d=0, blen=0):
    """ copy bit-wise buffer copy """
    b = blen//32
    r = blen-b*32
    for k in range(b):
        d = bs.unpack_from('u32', src, k*32+ofst_s)[0]
        bs.pack_into('u32', dst, k*32+ofst_d, d)
    if r > 0:
        fmt = 'u'+str(r)
        d = bs.unpack_from(fmt, src, b*32+ofst_s)[0]
        bs.pack_into(fmt, dst, b*32+ofst_d, d)
