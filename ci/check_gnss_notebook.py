"""Assert an executed GNSS notebook produced a healthy final solution.

Usage:
  python ci/check_gnss_notebook.py <executed.ipynb> \
      --fix3d-max 0.05 --min-amb 20 --float3d-max 0.5

Checks:
  * no cell raised an exception
  * "final FIX ... (N SD ambiguities)" line exists, FIX 3D < fix3d-max,
    N >= min-amb
  * "final float" line exists, float 3D < float3d-max
"""
import argparse
import json
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('notebook')
ap.add_argument('--fix3d-max', type=float, required=True)
ap.add_argument('--min-amb', type=int, required=True)
ap.add_argument('--float3d-max', type=float, required=True)
args = ap.parse_args()

nb = json.load(open(args.notebook))

texts = []
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'error':
            sys.exit('FAIL: cell raised %s: %s'
                     % (out.get('ename'), out.get('evalue')))
        if out.get('output_type') == 'stream':
            texts.append(''.join(out.get('text', [])))
        elif out.get('output_type') in ('execute_result', 'display_data'):
            texts.append(''.join(out.get('data', {}).get('text/plain', [])))
blob = '\n'.join(texts)

m_fix = re.search(
    r'final\s+FIX\s+2D=\s*([0-9.]+)\s+3D=\s*([0-9.]+)\s*m\s*'
    r'\((\d+)\s+SD ambiguities\)', blob)
if not m_fix:
    sys.exit('FAIL: no "final FIX ... (N SD ambiguities)" line in outputs')
fix3d = float(m_fix.group(2))
namb = int(m_fix.group(3))

m_flt = re.search(r'final\s+float\s+2D=\s*([0-9.]+)\s+3D=\s*([0-9.]+)\s*m',
                  blob)
if not m_flt:
    sys.exit('FAIL: no "final float" line in outputs')
flt3d = float(m_flt.group(2))

print('final float 3D = %.3f m  (limit %.3f)' % (flt3d, args.float3d_max))
print('final FIX   3D = %.3f m  (limit %.3f, %d SD ambiguities, min %d)'
      % (fix3d, args.fix3d_max, namb, args.min_amb))

if fix3d >= args.fix3d_max:
    sys.exit('FAIL: FIX 3D %.3f m >= %.3f m' % (fix3d, args.fix3d_max))
if namb < args.min_amb:
    sys.exit('FAIL: only %d SD ambiguities fixed (< %d)'
             % (namb, args.min_amb))
if flt3d >= args.float3d_max:
    sys.exit('FAIL: float 3D %.3f m >= %.3f m' % (flt3d, args.float3d_max))
print('PASS')
