"""GNSS types and units.

Constellations, signals, time scales, coordinates and the data structures
everything above is expressed in.

Named ``domain`` rather than ``types``: a package called ``types`` inside
cssrlib shadows the standard library module of that name whenever
src/cssrlib is on sys.path -- which it is when a module is run directly
from that directory, as several __main__ demo blocks expect. functools
imports types, so the shadowing broke the interpreter before any cssrlib
code ran.
"""
