"""Assert the executed official DopplerVelocityExample notebook is healthy.

Usage: python ci/check_doppler_notebook.py <executed.ipynb>

The notebook estimates the velocity of a *static* surveyed marker from
Doppler alone, so the truth is zero and the reported velocity error is the
error outright.

It reads a prepared .npz rather than calling cssrlib, so running it as
shipped would test GTSAM and nothing else. The CI job regenerates that .npz
from RINEX with gnss_frontend.load_doppler against the cssrlib revision
under test -- satposs, the RINEX decoder, geodist and satazel all run there
-- so these numbers move when this repository breaks.

Checks:
  * no cell raised an exception
  * enough epochs and Doppler factors survived the front end
  * velocity bias and RMS, and the range-rate residual, are within bounds
"""
import json
import re
import sys

# Reference run: 301 epochs, 5400 factors, horizontal RMS 0.0176 m/s,
# 3D RMS 0.0364 m/s, range-rate residual RMS 0.0290 m/s. The limits sit
# well clear of those so a rebuilt dataset does not make the job flaky,
# while still catching a front end that drops satellites or mis-computes
# satellite velocity.
MIN_EPOCHS = 250
MIN_FACTORS = 4000
MAX_ABS_BIAS = 0.05        # m/s, per axis
MAX_HORIZONTAL_RMS = 0.05  # m/s
MAX_3D_RMS = 0.10          # m/s
MAX_RESIDUAL_RMS = 0.10    # m/s


def notebook_text(path):
    nb = json.load(open(path))
    texts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                sys.exit("FAIL: cell raised %s: %s"
                         % (out.get("ename"), out.get("evalue")))
            if out.get("output_type") == "stream":
                texts.append("".join(out.get("text", [])))
            elif out.get("output_type") in ("execute_result", "display_data"):
                texts.append("".join(out.get("data", {})
                                     .get("text/plain", [])))
    return "\n".join(texts)


def need(pattern, blob, what):
    m = re.search(pattern, blob)
    if not m:
        sys.exit(f"FAIL: could not find {what} in the notebook output")
    return m


def main(path):
    blob = notebook_text(path)

    epochs = int(need(r"(\d+)\s+epochs at", blob, "the epoch count").group(1))
    factors, states = need(r"(\d+)\s+Doppler factors,\s+(\d+)\s+states",
                           blob, "the factor count").groups()
    factors, states = int(factors), int(states)

    bias = [float(v) for v in need(
        r"bias\s+E/N/U\s*=\s*([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*m/s",
        blob, "the velocity bias").groups()]

    hrms, rms3d, worst = (float(v) for v in need(
        r"horizontal RMS\s+([0-9.]+)\s+m/s,\s+3D RMS\s+([0-9.]+)\s+m/s,"
        r"\s+worst epoch\s+([0-9.]+)\s+m/s", blob, "the velocity RMS").groups())

    resid, nobs = need(
        r"range-rate residual:\s+RMS\s+([0-9.]+)\s+m/s over\s+(\d+)\s+obs",
        blob, "the range-rate residual").groups()
    resid, nobs = float(resid), int(nobs)

    checks = [
        ("epochs", epochs, MIN_EPOCHS, "min"),
        ("Doppler factors", factors, MIN_FACTORS, "min"),
        ("|bias| E", abs(bias[0]), MAX_ABS_BIAS, "max"),
        ("|bias| N", abs(bias[1]), MAX_ABS_BIAS, "max"),
        ("|bias| U", abs(bias[2]), MAX_ABS_BIAS, "max"),
        ("horizontal RMS", hrms, MAX_HORIZONTAL_RMS, "max"),
        ("3D RMS", rms3d, MAX_3D_RMS, "max"),
        ("range-rate residual RMS", resid, MAX_RESIDUAL_RMS, "max"),
    ]

    ok = True
    for label, value, limit, sense in checks:
        bad = value < limit if sense == "min" else value > limit
        ok &= not bad
        print("%-24s %10.4f  (%s %g) ... %s"
              % (label, value, sense, limit, "FAIL" if bad else "OK"))

    print("states=%d, observations=%d, worst epoch %.4f m/s"
          % (states, nobs, worst))

    if nobs != factors:
        print("NOTE: %d observations against %d factors" % (nobs, factors))

    if not ok:
        sys.exit("FAIL: Doppler notebook metrics out of bounds")
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1])
