#!/usr/bin/env python3
"""Where every upstream symbol went, generated from the two trees.

This branch cannot follow upstream by merging. It is 86 commits ahead of
``origin/dev`` with nothing behind it, and ``git diff -M50%`` detects not one
rename between them: the layer split rewrote the files rather than moving
them, and twelve modules were deleted outright. ``git merge`` from upstream
therefore reports every file as an add/delete conflict and is useless.

What is still answerable is the question a person actually has when upstream
changes something: *upstream touched ``pppssr.zdres`` -- where is that now?*
This script answers it by name-matching the symbols of both trees and writing
the result to UPSTREAM_MAP.md.

    python ci/upstream_map.py                 # rewrite UPSTREAM_MAP.md
    python ci/upstream_map.py --check         # exit 1 if it is out of date
    python ci/upstream_map.py --upstream REF  # against another baseline

Name matching is deliberate and its limits are worth stating: a symbol
upstream renamed on its way in looks deleted here, and one whose name is
common lands in "ambiguous". The map is a lookup aid, not a proof, so it says
which of those three it is for every symbol rather than guessing.

The default baseline is ``origin/main``, this fork's mirror of hirokawa's
cssrlib. ``origin/dev`` is the fork's own numba line and shares the flat
upstream layout, so the module map reads the same against either.
"""

from __future__ import annotations

import argparse
import ast
import collections
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(HERE, "UPSTREAM_MAP.md")
# The baseline is a pinned commit, not a moving branch name: against
# "origin/main" the map's content depended on which fork you cloned and
# how far its main had drifted, so --check disagreed between clones of
# the same code. Bump the pin (and regenerate) when a new upstream state
# is actually reviewed, with --upstream for the one-off comparison.
DEFAULT_UPSTREAM = "5e0cfb350032813a657a4cf1a97afa11187158e4"
SRC = "src/cssrlib/"

#: Modules the frontend dropped, and why. A deleted module's symbols are not
#: "missing" -- they were removed on purpose, and an upstream change to them
#: needs no port. Anything deleted without an entry here shows up as
#: UNEXPLAINED in the report.
DELETED = {
    "rtcm.py": "RTCM3 decoding; this branch reads RINEX only.",
    "sbas.py": "SBAS L1 / DFMC corrections; not used by the RTK / PPP-RTK path.",
    "ewss.py": "Emergency warning service; not positioning.",
    "osnma.py": "Galileo OSNMA authentication; not positioning.",
    "qznma.py": "QZSS QZNMA authentication; not positioning.",
    "rawnav.py": "Receiver raw navigation decoders; RINEX only here.",
    "pntpos.py": "Standalone point positioning; the engine is DD / PPP-RTK.",
    "dgps.py": "Code-differential GPS; superseded by the RTK path.",
    "plot.py": "Plotting helpers; presentation, not library.",
    "utils.py": "Grab-bag of helpers; the survivors moved to the layer that "
                "uses them.",
    "tlesim.py": "TLE orbit simulation; no caller.",
    "ionosphere.py": "Klobuchar / NTCM models; the estimator solves for iono. "
                     "(A numba-fork module: on origin/dev, not on "
                     "origin/main.)",
}

#: Modules that are gone as files but whose content lives on. These stay in
#: the map -- they are the ones an upstream change is most likely to touch.
#: pppssr.py in particular is the engine; listing it as deleted would have
#: hidden the mapping that matters most.
SPLIT = {
    "pppssr.py": "The PPP/PPP-RTK engine, split by concern into "
                 "cssrlib.estimation.* and composed by "
                 "cssrlib.engine.gnssobs.",
}

#: Root modules that re-export a whole layer. They define no symbols of their
#: own, so nothing "lands" in them; their location is part of their contract.
FACADES = ("gnss.py", "peph.py", "rinex.py")


def run(*args):
    out = subprocess.run(args, capture_output=True, text=True, cwd=HERE)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def modules(ref):
    """Library modules in ``ref``, excluding tests."""
    names = run("git", "ls-tree", "-r", "--name-only", ref, SRC).split()
    return sorted(n for n in names
                  if n.endswith(".py") and "/test/" not in n
                  and not n.endswith("__init__.py"))


def symbols(ref, path):
    """(kind, name) for every top-level function, class and method."""
    try:
        tree = ast.parse(run("git", "show", f"{ref}:{path}"))
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found.append(("function", node.name))
        elif isinstance(node, ast.ClassDef):
            found.append(("class", node.name))
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    found.append(("method", f"{node.name}.{sub.name}"))
    return found


def index(ref):
    """name -> [module, ...], for lookup across a whole tree.

    Methods are indexed by their bare name as well as ``Class.method``: the
    layer split moved methods between classes wholesale (one god class became
    four mixins), so the class half of the name is exactly what changed.

    Dunders are the exception and are indexed only under ``Class.method``.
    Every class has an ``__init__``; matching one by bare name returns the
    whole tree, which is not an answer.
    """
    by_name = collections.defaultdict(set)
    functions = collections.defaultdict(set)
    per_module = {}
    for path in modules(ref):
        syms = symbols(ref, path)
        per_module[path] = syms
        short = path[len(SRC):]
        for kind, name in syms:
            by_name[name].add(short)
            if kind == "function":
                functions[name].add(short)
            if kind == "method":
                bare = name.split(".", 1)[1]
                if not bare.startswith("__"):
                    by_name[bare].add(short)
    return by_name, functions, per_module


def classify(name, kind, current):
    """Where a symbol went: one place, several candidates, or nowhere."""
    hits = current.get(name)
    if not hits and kind == "method":
        bare = name.split(".", 1)[1]
        if not bare.startswith("__"):
            hits = current.get(bare)
    if not hits:
        return "absent", []
    return ("moved" if len(hits) == 1 else "ambiguous"), sorted(hits)


def build(upstream):
    _, _, up_modules = index(upstream)
    cur_by_name, cur_functions, cur_modules = index("HEAD")
    sha = run("git", "rev-parse", "--short", upstream).strip()
    subject = run("git", "log", "--format=%s", "-1", upstream).strip()

    rows = {}
    for path, syms in up_modules.items():
        short = path[len(SRC):]
        entries = []
        for kind, name in syms:
            verdict, where = classify(name, kind, cur_by_name)
            entries.append((kind, name, verdict, where))
        rows[short] = entries

    live = sorted(set(m[len(SRC):] for m in cur_modules))

    # Top-level functions this tree defines in more than one module. Methods
    # are excluded: a name on a base class and its subclasses is an override,
    # which is how the SSR decoders are meant to work.
    dupes = {n: sorted(m) for n, m in cur_functions.items() if len(m) > 1}

    stale = [d for d in DELETED if d not in rows]
    return sha, subject, rows, live, dupes, stale


def render(sha, subject, rows, live, dupes, stale, full=False):
    out = []
    w = out.append
    w("# Where upstream went")
    w("")
    w("Generated by `python ci/upstream_map.py`. Do not edit by hand.")
    w("")
    w(f"Upstream baseline: `{sha}` -- {subject}")
    w("")
    w("This branch cannot take upstream changes by merging -- it is 86")
    w("commits ahead of `origin/dev` with nothing behind it, and `git diff")
    w("-M50%` finds no renames at all between the two trees, because the")
    w("layer split rewrote the files instead of moving them. Porting an")
    w("upstream change means finding the code by name and applying it by")
    w("hand. This table is that lookup.")
    w("")
    w("Matching is by symbol name, so a symbol renamed on the way in reads")
    w("as `absent` and a common name reads as `ambiguous`. Methods are also")
    w("looked up by their bare name, because the class half is what the")
    w("split changed: one god class became four mixins.")
    w("")

    w("## Deleted upstream modules")
    w("")
    w("An upstream change to these needs no port.")
    w("")
    w("| upstream module | why it is gone |")
    w("| --- | --- |")
    for name in sorted(DELETED):
        if name in rows:
            w(f"| `{name}` | {DELETED[name]} |")
    w("")

    w("## Split modules")
    w("")
    w("Gone as a file, but the code is here. These are the ones an upstream")
    w("change is most likely to touch.")
    w("")
    w("| upstream module | what happened |")
    w("| --- | --- |")
    for name in sorted(SPLIT):
        if name in rows:
            w(f"| `{name}` | {SPLIT[name]} |")
    w("")

    w("## Module map")
    w("")
    w("Where each surviving upstream module's symbols now live, most-used")
    w("destination first.")
    w("")
    w("| upstream module | symbols | destinations |")
    w("| --- | --- | --- |")
    for name in sorted(rows):
        if name in DELETED:
            continue
        entries = rows[name]
        dest = collections.Counter()
        for _, _, verdict, where in entries:
            if verdict == "moved":
                dest[where[0]] += 1
        cells = ", ".join(f"`{d}` ({n})" for d, n in dest.most_common()) or "--"
        w(f"| `{name}` | {len(entries)} | {cells} |")
    w("")

    w("## Symbols with no counterpart")
    w("")
    w("In a surviving upstream module, but not found here. Either dropped")
    w("with the feature, or renamed -- name matching cannot tell the two")
    w("apart, so check before concluding a port is unnecessary.")
    w("")
    for name in sorted(rows):
        if name in DELETED:
            continue
        gone = [f"{k} `{n}`" for k, n, v, _ in rows[name] if v == "absent"]
        if gone:
            w(f"- `{name}`: " + ", ".join(gone))
    w("")

    if full:
        w("## Full symbol table")
        w("")
        for name in sorted(rows):
            if name in DELETED:
                continue
            w(f"### `{name}`")
            w("")
            w("| kind | upstream symbol | now in |")
            w("| --- | --- | --- |")
            for kind, sym, verdict, where in rows[name]:
                if verdict == "moved":
                    cell = f"`{where[0]}`"
                elif verdict == "ambiguous":
                    cell = "? " + ", ".join(f"`{d}`" for d in where)
                else:
                    cell = "--"
                w(f"| {kind} | `{sym}` | {cell} |")
            w("")
    else:
        w("## Per-symbol lookup")
        w("")
        w("The full upstream-symbol -> current-module table is generated")
        w("on demand rather than committed:")
        w("")
        w("```")
        w("python ci/upstream_map.py --full")
        w("```")
        w("")

    w("## Defined in more than one module here")
    w("")
    w("Two definitions of one name. Some are intentional -- `core` holds an")
    w("array kernel and `domain` the adapter that feeds it -- and some are")
    w("duplication inherited from upstream. Either way, porting an upstream")
    w("change to one of these means checking both.")
    w("")
    w("| symbol | modules |")
    w("| --- | --- |")
    for name in sorted(dupes):
        w(f"| `{name}` | " + ", ".join(f"`{m}`" for m in dupes[name]) + " |")
    w("")

    if stale:
        w("## Stale entries in DELETED")
        w("")
        w("Named in `ci/upstream_map.py` but absent from this baseline:")
        w("")
        for name in sorted(stale):
            w(f"- `{name}`")
        w("")

    w("## Modules that received nothing from upstream")
    w("")
    w("No upstream symbol resolves here, so these are this branch's own:")
    w("state layout, configuration, the composition point, the Numba")
    w("kernels. An upstream change never ports into them.")
    w("")
    landed = {d for entries in rows.values()
              for _, _, verdict, where in entries
              if verdict == "moved" for d in where}
    for name in live:
        if name in landed:
            continue
        if name in FACADES:
            w(f"- `{name}` -- re-exports a whole layer; defines nothing "
              f"itself")
        else:
            w(f"- `{name}`")
    w("")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM,
                    help=f"baseline ref (default {DEFAULT_UPSTREAM})")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if UPSTREAM_MAP.md is out of date")
    ap.add_argument("--full", action="store_true",
                    help="include the per-symbol table (large; for local "
                         "lookups, not for committing)")
    args = ap.parse_args(argv)

    if subprocess.run(["git", "rev-parse", "--verify", "--quiet",
                       args.upstream + "^{commit}"],
                      capture_output=True).returncode:
        print(f"baseline {args.upstream} is not in this clone; fetch it "
              f"first:\n  git fetch https://github.com/hirokawa/cssrlib.git "
              f"{args.upstream}")
        return 1

    text = render(*build(args.upstream), full=args.full)

    if args.check:
        current = open(OUTPUT).read() if os.path.exists(OUTPUT) else ""
        if current != text:
            print(f"{OUTPUT} is out of date; rerun python ci/upstream_map.py")
            return 1
        print(f"{OUTPUT} is up to date")
        return 0

    with open(OUTPUT, "w") as fh:
        fh.write(text)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
