"""CSSRlib - toolkit for PPP / PPP-RTK / RTK positioning.

The package is organised in layers, bottom to top:

    cssrlib.core         pure numerics and vocabulary
    cssrlib.types        constellations, signals, time, coordinates, structs
    cssrlib.models       orbits, Earth frames, tides, antennas, biases
    cssrlib.fileio       RINEX reading and writing
    cssrlib.ssr          State-Space Representation correction decoding
    cssrlib.estimation   state layout, configuration, observation model, filter
    cssrlib.engine       the composed engine and the modes it is configured into

Dependencies point downwards only; ``test_dependencies.py`` enforces it.

The module names that predate this layout -- ``cssrlib.gnss``,
``cssrlib.rinex``, ``cssrlib.rtk`` and the rest -- remain available at the
top level and forward into the packages above.

Note this file is not decorative: without it ``packages = find:`` discovers
nothing and a non-editable ``pip install .`` produces a distribution
containing no code at all.
"""

__version__ = "1.2.1"
