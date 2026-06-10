# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

## [Unreleased]

### Added

- `rinex.auto_detect_signals(sig_map_rov, sig_map_base=None, max_freq=2, ...)`
  builds per-system signal lists straight from the RINEX header(s), so the
  caller no longer hand-crafts them. With a base sig_map it returns matching
  rover/base lists covering the same (sys, typ, band). Also exposed as the
  convenience method `rnxdec.autoSignals(decb=None, max_freq=2)` which detects
  and applies the signals in one call.

### Removed

- Reduced `rtkpos`/`pppos` to a double-difference-only core for external
  estimators. Removed the built-in EKF and undifferenced machinery —
  `process`, `base_process`, `zdres`, `sdres`, `udstate`, `kfupdate`,
  `valpos`, `holdamb`/`holdamb_flags`, and their helpers — keeping only what
  the GTSAM DD workflow uses: `prepare_double_difference_measurements`
  (now always DD-only), `base_process_dd_only`, `qcedit`,
  `manage_ambiguities_external`, `resamb_lambda` (+ `ddidx`/`restamb`) and
  `satposs`. `pppssr.py` shrank from ~2300 to ~870 lines. The state vector
  and its covariance are now owned by the external estimator. Recover the
  EKF path from the `dev` branch if needed.


- Stripped the library down to a minimal broadcast-ephemeris RTK + LAMBDA
  core (11 modules, down from 31). Deleted everything not used by that
  workflow: SSR/CSSR decoders and base (`cssr_bds/has/mdc/pvs`, `cssrlib`,
  `rtcm`), antenna/precise-orbit (`peph`), Earth tides / phase wind-up
  (`ppp`), SBAS & authentication (`sbas`, `osnma`, `qznma`, `ewss`), other
  positioning modes (`dgps`, `pntpos`, `ppprtk`), receiver raw nav
  (`rawnav`) and misc (`ionosphere`, `tlesim`, `plot`, `utils`).
  Correspondingly trimmed `ephemeris.satposs` (broadcast-only; removed
  `satpos`) and `pppssr.zdres` / `qcedit` (dropped SSR-bias, antenna-model,
  phase-wind-up and tide handling). Recover any of these from the `dev`
  branch if needed.

### Changed

- The double-difference RTK workflow
  (`prepare_double_difference_measurements` with
  `dd_only=True, compute_zdres=False` + `manage_ambiguities_external`) now
  depends only on the lightweight core, ideal for embedding in an external
  estimator such as a GTSAM factor graph.
- `uTideModel` moved from `cssrlib.ppp` to `cssrlib.gnss` (grouped with
  `uTropoModel` / `uIonoModel`).
- `rtkpos` now disables solid-Earth-tide correction by default
  (`nav.tidecorr = uTideModel.NONE`): tides cancel in the rover-base double
  difference at RTK baselines. Re-enable via `nav.tidecorr` for long
  baselines.
- `rtkpos.prepare_double_difference_measurements` now returns a
  `DDMeasurements` object: a `dict` subclass (fully backward compatible)
  that also supports attribute access (`dd.rs` as well as `dd['rs']`) and
  documents its fields, for consumption by external estimators such as
  GTSAM factor graphs.

### Fixed

- `mlambda` (LAMBDA AR) now raises `LambdaError` (a `numpy.linalg.LinAlgError`
  subclass) instead of `SystemExit` when the covariance is not positive
  definite, so callers embedding AR in their own solver can catch it with a
  normal `except`.
- `qcedit` no longer raises `IndexError` when a constellation provides fewer
  signal bands than `nav.nf` (e.g. a single-frequency system in a
  dual-frequency setup); the absent bands are edited out per satellite while
  the present bands are still used.

# [1.2.1] 2025-11-03


### Added

- Jupyter notebook for Authentication, EWSS

### Fixed

- Build issue on Windows (removed pysolid)

### Changed

- Updated RTCM SC134 messages

# [1.2.0] 2025-10-14

### Added

- Add SBAS based PPP for PPP via SouthPAN (cssr_pvs)
- Add L1 SBAS and L1/L5(DFMC) SBAS (sbas)
- Add authentication for Galileo OSNMA and QZSS QZNMA (osnma, qznma)
- Add EWSS for QZSS and Galileo (ewss)
- draft RTCM SC134 messages (rtcm)
- Add BDS signals for QZSS MADOCA-PPP
- Decoder for u-blox receiver (on cssrlib-data)
- Improved LAMBDA AR from LAMBDA 4.0 toolbox (mlambda)
- Support for RINEX 4.02 (rinex)
- Add NavIC L1 (rawnav)
- Add doppler for RINEX (@inuex35)

### Fixed

- Fixed GLONASS ephemeris decoder (rawnav)

# [1.1.0] 2024-07-15

### Added

- Add GLONASS FDMA, NavIC, BDS D1/D2, CNAV-2/3 message decoder
- Add GLONASS frequency channel number to apc2com() (@AndreHauschild)
- Add a test workflow (@AndreHauschild)
- Add APC reference corrections for IGS and RTCM3 SSR corrections (@AndreHauschild)
- Decoder for Javad receiver (on cssrlib-data)

### Changed

- Use different APC reference signals for SIS and IDD of Galileo HAS (@AndreHauschild)
- Change MADOCA APC reference (@AndreHauschild)

### Fixed

- Fixed CNAV decoder

# [1.0.0] 2024-01-01

### Added
New integrated class structure (PPPOS) for PPP/PPP-RTK/RTK processing

- Support for RTCM3 (Galileo HAS IDD)
- Decoder for Septentrio receiver (on cssrlib-data)
- Parser for RTCM3, L5 SBAS
- Experimental support for PPP via SouthPAN (PVS)
- New solid Earth tides model using PySolid (2010 IERS Conventions) 

### Changed

- Improved documentation
- Sign of SSR satellite signal code/phase bias align with RTCM 3 convention
- PPPIGS was integrated into PPPOS and removed

### Fixed

- Link for cssrlib-data

### Deprecated
### Removed
Function based PPP/PPP-RTK/RTK processing
### Security

# [0.8.0] 2023-09-09

### Added

- New signal structure
- Support for open PPP services: Galileo HAS (SIS), BDS PPP, QZSS MADOCA-PPP
- Support for IGS (SP3+BIAS)
- Parser for SP3, ANTEX, BIAS files
- Support for PPP-AR
- Jupyter notebook with examples

### Changed

Improved documentation

Added link for Google Colab

### Fixed
### Deprecated
### Removed
### Security

# [0.3.0] 2022-03-01

### Added
Initial version for PPP-RTK (QZSS CLAS) and RTK

### Changed
### Fixed
### Deprecated
### Removed
### Security
