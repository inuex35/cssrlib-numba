"""BDS PPP (BeiDou B2b PPP) example (minimal-core cssrlib).

Global PPP with BeiDou B2b PPP corrections (no regional atmosphere grid, so the
troposphere and ionosphere are estimated). Uses the unified PPP engine
(cssrlib.gnssobs) -- the same class RTK/PPP-RTK subclass -- driven by the BDS
B2b decoder cssrlib.cssr_bds.

Float only: the B2b PPP message carries orbit/clock and code bias but NO phase
bias (cssr_bds has no pbias decoder), so undifferenced integer ambiguity
resolution is not possible. Float is the accuracy ceiling for this service.

Data (companion cssrlib-data repo, see examples/ppprtk_clas.py):
  doy2025-233: 233h_rnx.{obs,nav}, 233h_bdsb2b.txt, antex/igs20.atx
"""
import os
from copy import deepcopy
from binascii import unhexlify
import numpy as np

from cssrlib.cssr_bds import cssr_bds
from cssrlib.gnss import (ecef2pos, ecef2enu, Nav, rSigRnx, time2gpst,
                          time2doy, time2str, epoch2time)
from cssrlib.peph import atxdec, searchpcv
from cssrlib.gnssobs import gnssobs
from cssrlib.rinex import rnxdec

DATADIR = os.environ.get(
    'CSSRLIB_DATA',
    os.path.join(os.path.dirname(__file__), '..', 'cssrlib-data', 'data'))
ep = [2025, 8, 21, 7, 0, 0]
xyz_ref = [-3962108.6836, 3381309.5672, 3668678.6720]
pos_ref = ecef2pos(xyz_ref)
prn_ref = 59                      # BeiDou PRN broadcasting B2b PPP
nep = int(os.environ.get('NEP', '300'))

time = epoch2time(ep)
doy = int(time2doy(time))
let = chr(ord('a') + ep[3])
bdir = f'{DATADIR}/doy{ep[0]:04d}-{doy:03d}/'

sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"), rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("GS1C"), rSigRnx("GS2W"),
        rSigRnx("CC1P"), rSigRnx("CC5P"), rSigRnx("CL1P"), rSigRnx("CL5P"),
        rSigRnx("CS1P"), rSigRnx("CS5P")]

rnx = rnxdec()
rnx.setSignals(sigs)
nav = Nav()
nav.pmode = 0
nav = rnx.decode_nav(bdir + f'{doy:03d}{let}_rnx.nav', nav)
v = np.genfromtxt(bdir + f'{doy:03d}{let}_bdsb2b.txt',
                  dtype=[('wn', 'int'), ('tow', 'int'), ('prn', 'int'),
                         ('type', 'int'), ('len', 'int'), ('nav', 'S124')])
cs = cssr_bds()
cs.monlevel = 0
atx = atxdec()
atx.readpcv(f'{DATADIR}/antex/igs20.atx')

assert rnx.decode_obsh(bdir + f'{doy:03d}{let}_rnx.obs') >= 0
rnx.autoSubstituteSignals()
ppp = gnssobs(nav, rnx.pos, '/tmp/ppp_bds.log')   # global PPP: estimate tropo/iono
nav.sat_ant = atx.pcvs
nav.rcv_ant = searchpcv(atx.pcvr, rnx.ant, rnx.ts)

best = 9.9
obs = rnx.decode_obs()
while time > obs.t and obs.t.time != 0:
    obs = rnx.decode_obs()
for k in range(nep):
    week, tow = time2gpst(obs.t)
    cs.week = week
    cs.tow0 = tow // 86400 * 86400
    if k == 0:
        nav.t = deepcopy(obs.t)
        t0 = deepcopy(obs.t)
        t0.time = t0.time // 30 * 30
        nav.time_p = t0
    vi = v[(v['tow'] == tow) & (v['prn'] == prn_ref)]
    if len(vi) > 0:
        cs.decode_cssr(unhexlify(vi['nav'][0]), 0)
    if (cs.lc[0].cstat & 0xf) == 0xf:
        ppp.process(obs, cs=cs)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu = ecef2enu(pos_ref, sol - xyz_ref)
    h = float(np.hypot(enu[0], enu[1]))
    if not np.isnan(h):
        best = min(best, h)
    if k % 30 == 0 or k == nep - 1:
        print(f"{time2str(obs.t)} ne={k:3d} ENU {enu[0]:+7.3f} {enu[1]:+7.3f} "
              f"{enu[2]:+7.3f}  2D={h:6.3f} mode={nav.smode}")
    obs = rnx.decode_obs()
    if obs.t.time == 0:
        break

print(f"\nbest 2D horizontal error = {best:.3f} m  "
      f"(BDS B2b global PPP; convergence is slower than PPP-RTK)")
