"""UPSTREAM_MAP.md must name every module, or it is quietly wrong.

The map says where each upstream symbol went, and it is generated --
``python ci/upstream_map.py``. Generated files rot when the thing they
describe changes and nobody reruns them, and this one rots silently: a
module added after the last run simply is not mentioned, and a reader
looking for it concludes upstream has nothing that belongs there.

Regenerating needs ``origin/main``, which a shallow CI checkout does not
have, so this test does not do that. It checks the property that can be
checked from the working tree alone: every library module is named
somewhere in the map, either as a destination or in the list of modules
that received nothing.
"""

import os
import re

import pytest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(SRC))
MAP = os.path.join(ROOT, "UPSTREAM_MAP.md")
REGEN = "python ci/upstream_map.py"


def library_modules():
    """Every module the map should account for, as ``layer/name.py``."""
    out = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "test", "data")]
        for fn in filenames:
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), SRC)
            out.append(rel.replace(os.sep, "/"))
    return sorted(out)


@pytest.fixture(scope="module")
def mentioned():
    if not os.path.exists(MAP):
        pytest.fail(f"{MAP} is missing; generate it with {REGEN}")
    return set(re.findall(r"`([\w/]+\.py)`", open(MAP).read()))


def test_the_map_records_which_upstream_it_was_built_against(mentioned):
    """Without a baseline the map cannot be checked or regenerated."""
    text = open(MAP).read()
    assert re.search(r"Upstream baseline: `\S+` at `[0-9a-f]{7,}`", text), (
        "the map does not name the upstream commit it describes")


@pytest.mark.parametrize("module", library_modules())
def test_every_module_is_accounted_for(module, mentioned):
    assert module in mentioned, (
        f"{module} is not mentioned in UPSTREAM_MAP.md -- it was added or "
        f"renamed after the map was last generated. Rerun {REGEN}")
