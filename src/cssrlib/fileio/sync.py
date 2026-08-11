"""Pairing two receivers' observation streams, and choosing signals."""

import numpy as np
from cssrlib.domain.gnss import uTYP
from cssrlib.domain.gnss import timediff


def sync_obs(dec, decb, dt_th=0.1):
    """ sync observation between rover and base """
    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    while True:
        dt = timediff(obs.t, obsb.t)
        if np.abs(dt) <= dt_th:
            break
        if dt > dt_th:
            obsb = decb.decode_obs()
        elif dt < dt_th:
            obs = dec.decode_obs()
    return obs, obsb


def _obs_is_eof(obs):
    """ EOF check for rnxdec.decode_obs (returns default Obs() at EOF) """
    return obs.t.time == 0 and obs.t.sec == 0.0


def sync_obs_hold(dec, decb, maxage=30.0):
    """
    Rover-driven sync generator with base-station hold (RTKLIB maxtdiff-style).

    Yields `(obs_rover, obs_base, dt)` for every rover epoch:
      - obs_base: nearest base observation with |t_rover - t_base| <= maxage,
        reused across rover epochs until a newer base arrives. ``None`` when
        no base is within ``maxage`` (e.g. base stream ended or not yet
        started). ``dt`` is set even when base is out of range so the caller
        can log it.
      - dt: ``timediff(t_rover, t_base)`` (NaN when no base decoded yet).

    Works for arbitrary rate combinations, e.g. 5 Hz rover + 1 Hz base: each
    1 Hz base record is reused for ~5 rover epochs until the next base record
    is closer.

    Parameters
    ----------
    dec, decb : rnxdec
        Rover / base decoders positioned after ``decode_obsh``.
    maxage : float
        Maximum |t_rover - t_base| (seconds) for which the base obs is still
        considered usable. Mirrors RTKLIB ``prcopt.maxtdiff`` (default 30 s).
    """
    obsb_curr = decb.decode_obs()
    if _obs_is_eof(obsb_curr):
        obsb_curr = None
        obsb_next = None
    else:
        obsb_next = decb.decode_obs()
        if _obs_is_eof(obsb_next):
            obsb_next = None

    while True:
        obs = dec.decode_obs()
        if _obs_is_eof(obs):
            return

        # Advance base while the next base record is strictly closer to the
        # current rover epoch than the held one (nearest-neighbor hold).
        while obsb_next is not None:
            if obsb_curr is None:
                obsb_curr = obsb_next
                nxt = decb.decode_obs()
                obsb_next = None if _obs_is_eof(nxt) else nxt
                continue
            dt_curr = abs(timediff(obs.t, obsb_curr.t))
            dt_next = abs(timediff(obs.t, obsb_next.t))
            if dt_next < dt_curr:
                obsb_curr = obsb_next
                nxt = decb.decode_obs()
                obsb_next = None if _obs_is_eof(nxt) else nxt
            else:
                break

        if obsb_curr is None:
            yield obs, None, float('nan')
            continue

        dt = timediff(obs.t, obsb_curr.t)
        if abs(dt) <= maxage:
            yield obs, obsb_curr, dt
        else:
            yield obs, None, dt


# Preferred band order when a receiver offers more than max_freq of them.
_BAND_PRIORITY = (1, 2, 5, 7, 6, 8, 3, 4, 9)


def _group_by_band(sigs, typ):
    """Group the rSigRnx values of one sig_map system by frequency band.

    Returns ``{band: rSigRnx}`` for the requested observation type, keeping
    the first signal seen per band (sig_map preserves RINEX header order).
    """
    out = {}
    for s in sigs.values():
        if s.typ != typ:
            continue
        band = int(s.sig) // 100
        out.setdefault(band, s)
    return out


def auto_detect_signals(sig_map_rov, sig_map_base=None, max_freq=2,
                        required=(uTYP.C, uTYP.L, uTYP.S),
                        systems=None, strict_freq=False):
    """Build signal list(s) directly from RINEX header signal maps.

    Mimics RTKLIB's "use whatever the file declares" behaviour, so the caller
    need not hand-craft per-system signal lists. With a base ``sig_map`` it
    returns matching rover/base lists covering the same (sys, typ, band).

    Typical usage::

        dec = rnxdec();  dec.decode_obsh(rover_obs)
        decb = rnxdec(); decb.decode_obsh(base_obs)
        sigs, sigsb = auto_detect_signals(dec.sig_map, decb.sig_map, max_freq=2)
        dec.setSignals(sigs); decb.setSignals(sigsb)

    Parameters
    ----------
    sig_map_rov : dict
        ``rnxdec.sig_map`` of the rover, populated by ``decode_obsh``.
    sig_map_base : dict, optional
        ``rnxdec.sig_map`` of the base. When omitted, only the rover list is
        built and the second return value is an empty list.
    max_freq : int
        Number of frequency bands to keep per system (RTKLIB ``nf``).
    required : tuple of uTYP
        Observation types each band must provide (default C+L+S =
        pseudorange, carrier phase, SNR).
    systems : iterable of uGNSS, optional
        Constellations to consider; default = all common to both receivers.
    strict_freq : bool, default False
        Drop systems that cannot supply ``max_freq`` common bands. No longer
        required for safety (qcedit tolerates short-band systems), so it
        defaults to False to keep single-frequency systems usable.

    Returns
    -------
    (sigs, sigsb) : tuple of list of rSigRnx
        Ready to pass to ``rnxdec.setSignals``. ``sigsb`` is empty when no
        base sig_map is given.
    """
    rov_systems = set(sig_map_rov.keys())
    have_base = sig_map_base is not None
    base_systems = set(sig_map_base.keys()) if have_base else rov_systems
    if systems is None:
        systems = rov_systems & base_systems
    else:
        systems = set(systems) & rov_systems & base_systems

    sigs, sigsb = [], []
    for sys in systems:
        rov_by_typ = {t: _group_by_band(sig_map_rov[sys], t) for t in required}
        if have_base:
            base_by_typ = {t: _group_by_band(sig_map_base[sys], t)
                           for t in required}

        # Bands fully covered (every required type) on both sides.
        common_bands = set(rov_by_typ[required[0]].keys())
        for t in required[1:]:
            common_bands &= rov_by_typ[t].keys()
        if have_base:
            for t in required:
                common_bands &= base_by_typ[t].keys()
        if not common_bands:
            continue
        if strict_freq and len(common_bands) < max_freq:
            continue

        # Canonical band order (L1, then L2, then L5 ...).
        ordered = [b for b in _BAND_PRIORITY if b in common_bands]
        ordered += sorted(b for b in common_bands if b not in ordered)
        for band in ordered[:max_freq]:
            for t in required:
                sigs.append(rov_by_typ[t][band])
                if have_base:
                    sigsb.append(base_by_typ[t][band])
    return sigs, sigsb
