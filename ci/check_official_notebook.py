"""Assert the executed official RtkAndPppExample notebook is healthy.

Usage: python ci/check_official_notebook.py <executed.ipynb>

The notebook has two parts, each ending in a line like
  FIXED  2D=0.007  3D=0.015 m  (28 SD ambiguities)
Part 1 is DD-RTK (short baseline), Part 2 is PPP-RTK (CLAS).

Checks:
  * no cell raised an exception
  * two FIXED lines are present
  * Part 1: FIX 3D < 0.05 m, >= 20 SD ambiguities
  * Part 2: FIX 3D < 0.15 m, >= 15 SD ambiguities
"""
import json
import re
import sys

LIMITS = [
    ('Part 1 (RTK)', 0.05, 20),
    ('Part 2 (PPP-RTK)', 0.15, 15),
]

nb = json.load(open(sys.argv[1]))
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

fixed = re.findall(
    r'FIXED\s+2D=\s*([0-9.]+)\s+3D=\s*([0-9.]+)\s*m\s*'
    r'\((\d+)\s+SD ambiguities\)', blob)
if len(fixed) < len(LIMITS):
    sys.exit('FAIL: expected %d FIXED result lines, found %d'
             % (len(LIMITS), len(fixed)))

ok = True
for (label, fix3d_max, min_amb), (d2, d3, n) in zip(LIMITS, fixed):
    d3, n = float(d3), int(n)
    status = 'OK'
    if d3 >= fix3d_max or n < min_amb:
        status = 'FAIL'
        ok = False
    print('%s: FIX 3D=%.3f m (limit %.3f), %d SD ambiguities (min %d) ... %s'
          % (label, d3, fix3d_max, n, min_amb, status))
if not ok:
    sys.exit('FAIL: official notebook metrics out of bounds')
print('PASS')
