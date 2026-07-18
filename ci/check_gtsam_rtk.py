"""Assert the executed gtsam_rtk notebook produced a healthy RTK result.

Usage: python ci/check_gtsam_rtk.py <executed_notebook.ipynb>

Checks:
  * no cell raised an exception
  * the final-result cell reports a FIX solution
  * FIX 3D error < 0.05 m and >= 20 SD ambiguities fixed
  * float 3D error < 0.5 m
"""
import json
import re
import sys

path = sys.argv[1]
nb = json.load(open(path))

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

print('final float 3D = %.3f m' % flt3d)
print('final FIX   3D = %.3f m  (%d SD ambiguities)' % (fix3d, namb))

if fix3d >= 0.05:
    sys.exit('FAIL: FIX 3D %.3f m >= 0.05 m' % fix3d)
if namb < 20:
    sys.exit('FAIL: only %d SD ambiguities fixed (< 20)' % namb)
if flt3d >= 0.5:
    sys.exit('FAIL: float 3D %.3f m >= 0.5 m' % flt3d)
print('PASS')
