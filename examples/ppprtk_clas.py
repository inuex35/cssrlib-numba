"""QZSS CLAS PPP-RTK example (minimal-core cssrlib).

Runs PPP-RTK positioning with QZSS CLAS (Compact SSR via the L6 message) on the
cssrlib-data sample (doy 2025-233). Requires the companion data repository:

    git clone https://github.com/hirokawa/cssrlib-data

and an ANTEX file at cssrlib-data/data/antex/igs20.atx (see data/igs_files.txt).

This exercises the full PPP-RTK stack ported into the minimal core:
  cssrlib.cssrlib (CLAS decoder) -> cssrlib.ppprtk.ppprtkpos -> cm positioning.
"""
import os
from copy import deepcopy
from binascii import unhexlify
import numpy as np

from cssrlib.cssrlib import cssr
from cssrlib.gnss import (ecef2pos, ecef2enu, Nav, time2gpst, time2str,
                          time2doy, rSigRnx, epoch2time)
from cssrlib.peph import atxdec, searchpcv
from cssrlib.ppprtk import ppprtkpos
from cssrlib.rinex import rnxdec

# --- input data (edit DATADIR to point at your cssrlib-data checkout) --------
DATADIR = os.environ.get(
    'CSSRLIB_DATA',
    os.path.join(os.path.dirname(__file__), '..', 'cssrlib-data', 'data'))
ep = [2025, 8, 21, 7, 0, 0]
xyz_ref = [-3962108.7007, 3381309.5532, 3668678.6648]   # known marker (ECEF)
prn_ref, l6_ch = 199, 0                                  # QZSS CLAS L6D
nep = int(os.environ.get('NEP', '120'))

time = epoch2time(ep)
doy = int(time2doy(time))
let = chr(ord('a') + ep[3])
bdir = f'{DATADIR}/doy{ep[0]:04d}-{doy:03d}/'
navfile = bdir + f'{doy:03d}{let}_rnx.nav'
obsfile = bdir + f'{doy:03d}{let}_rnx.obs'
file_l6 = bdir + f'{doy:03d}{let}_qzsl6.txt'
atxfile = f'{DATADIR}/antex/igs20.atx'
griddef = f'{DATADIR}/clas_grid.def'
pos_ref = ecef2pos(xyz_ref)

sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("EC1C"), rSigRnx("EC5Q"),
        rSigRnx("JC1C"), rSigRnx("JC2L"),
        rSigRnx("GL1C"), rSigRnx("GL2W"), rSigRnx("EL1C"), rSigRnx("EL5Q"),
        rSigRnx("JL1C"), rSigRnx("JL2L"),
        rSigRnx("GS1C"), rSigRnx("GS2W"), rSigRnx("ES1C"), rSigRnx("ES5Q"),
        rSigRnx("JS1C"), rSigRnx("JS2L")]

nav = Nav()
nav = rnxdec().decode_nav(navfile, nav)
atx = atxdec()
atx.readpcv(atxfile)
rnx = rnxdec()
rnx.setSignals(sigs)

cs = cssr()
cs.monlevel = 0
cs.week = time2gpst(time)[0]
cs.read_griddef(griddef)

assert rnx.decode_obsh(obsfile) >= 0
rnx.autoSubstituteSignals()
ppprtk = ppprtkpos(nav, rnx.pos, '/tmp/ppprtk_clas.log')
nav.rcv_ant = searchpcv(atx.pcvr, rnx.ant, rnx.ts)
nav.sat_ant = atx.pcvs
cs.find_grid_index(ecef2pos(rnx.pos))

v = np.genfromtxt(file_l6, dtype=[('wn', 'int'), ('tow', 'int'),
                                  ('prn', 'int'), ('type', 'int'),
                                  ('len', 'int'), ('nav', 'S500')])

best = 9.9
obs = rnx.decode_obs()
while time > obs.t and obs.t.time != 0:
    obs = rnx.decode_obs()
for k in range(nep):
    week, tow = time2gpst(obs.t)
    vi = v[(v['tow'] == tow) & (v['type'] == l6_ch) & (v['prn'] == prn_ref)]
    if len(vi) > 0:
        cs.decode_l6msg(unhexlify(vi['nav'][0]), 0)
        if cs.fcnt == 5:
            cs.decode_cssr(bytes(cs.buff), 0)
    if k == 0:
        nav.t = deepcopy(obs.t)
        t0 = deepcopy(obs.t)
        t0.time = t0.time // 30 * 30
        cs.time = obs.t
        nav.time_p = t0
    if cs.chk_stat():
        ppprtk.process(obs, cs=cs)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu = ecef2enu(pos_ref, sol - xyz_ref)
    h = float(np.hypot(enu[0], enu[1]))
    best = min(best, h)
    if k % 10 == 0 or k == nep - 1:
        print(f"{time2str(obs.t)} ne={k:3d} ENU {enu[0]:+7.3f} {enu[1]:+7.3f} "
              f"{enu[2]:+7.3f}  2D={h:6.3f} mode={nav.smode}")
    obs = rnx.decode_obs()
    if obs.t.time == 0:
        break

print(f"\nbest 2D horizontal error = {best:.3f} m  (mode 4 = ambiguity fixed)")
