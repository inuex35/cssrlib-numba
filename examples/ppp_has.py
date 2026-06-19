"""Galileo HAS PPP example (minimal-core cssrlib).

Global PPP with Galileo High Accuracy Service (HAS) corrections decoded from the
E6-B CNAV pages (Reed-Solomon recovered). No regional atmosphere grid, so the
troposphere and ionosphere are estimated. Uses the unified PPP engine
(cssrlib.gnssobs) driven by the HAS decoder cssrlib.cssr_has.

Float only: HAS broadcasts phase biases (cssr_has decodes them and zdres
applies them), so PPP-AR is mechanically supported, but the bundled sample is
only ~6.5 min of 1 Hz data -- far short of the 20-30 min a global PPP float
needs to converge. Resolving integers before convergence fixes to the WRONG
cycles and degrades the solution, so this sample stays float (set nav.armode
> 0 only with a much longer record).

Data (companion cssrlib-data repo):
  doy2025-233: 233h_rnx.{obs,nav}, 233h_gale6.txt, antex/igs20.atx
  samples/Galileo-HAS-SIS-ICD_1.0_Annex_B_Reed_Solomon_Generator_Matrix.txt
"""
import os
from copy import deepcopy
import numpy as np

from cssrlib.cssr_has import cssr_has, cnav_msg
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
nep = int(os.environ.get('NEP', '300'))

time = epoch2time(ep)
doy = int(time2doy(time))
let = chr(ord('a') + ep[3])
bdir = f'{DATADIR}/doy{ep[0]:04d}-{doy:03d}/'
file_has = bdir + f'{doy:03d}{let}_gale6.txt'
file_gm = f'{DATADIR}/../samples/' \
          'Galileo-HAS-SIS-ICD_1.0_Annex_B_Reed_Solomon_Generator_Matrix.txt'

sigs = [rSigRnx("GC1C"), rSigRnx("GC2L"), rSigRnx("GL1C"), rSigRnx("GL2L"),
        rSigRnx("GS1C"), rSigRnx("GS2L"),
        rSigRnx("EC1C"), rSigRnx("EC5Q"), rSigRnx("EL1C"), rSigRnx("EL5Q"),
        rSigRnx("ES1C"), rSigRnx("ES5Q")]

cnav = cnav_msg()
cnav.load_gmat(file_gm)

rnx = rnxdec()
rnx.setSignals(sigs)
nav = Nav()
nav.pmode = 0
nav = rnx.decode_nav(bdir + f'{doy:03d}{let}_rnx.nav', nav)
v = np.genfromtxt(file_has, dtype=[('wn', 'int'), ('tow', 'int'),
                                   ('prn', 'int'), ('type', 'int'),
                                   ('len', 'int'), ('nav', 'S124')])
cs = cssr_has()
cs.monlevel = 0
atx = atxdec()
atx.readpcv(f'{DATADIR}/antex/igs20.atx')

assert rnx.decode_obsh(bdir + f'{doy:03d}{let}_rnx.obs') >= 0
rnx.autoSubstituteSignals()
ppp = gnssobs(nav, rnx.pos, '/tmp/ppp_has.log')   # global PPP: estimate tropo/iono
nav.elmin = np.deg2rad(5.0)
nav.sat_ant = atx.pcvs
nav.rcv_ant = searchpcv(atx.pcvr, rnx.ant, rnx.ts)

best = 9.9
obs = rnx.decode_obs()
while time > obs.t and obs.t.time != 0:
    obs = rnx.decode_obs()
for k in range(nep):
    week, tow = time2gpst(obs.t)
    cs.week = week
    cs.tow0 = tow // 3600 * 3600
    if k == 0:
        nav.t = deepcopy(obs.t)
        t0 = deepcopy(obs.t)
        t0.time = t0.time // 30 * 30
        nav.time_p = t0
    vi = v[v['tow'] == tow]
    HASmsg = cnav.decode_cnav(tow, vi)          # decode E6-B CNAV pages
    if HASmsg is not None:
        cs.msgtype = cnav.msgtype
        cs.decode_cssr(HASmsg)                   # decode HAS message
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
      f"(Galileo HAS global PPP; convergence is slower than PPP-RTK)")
