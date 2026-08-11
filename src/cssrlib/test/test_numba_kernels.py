"""Assert the Numba kernels are actually compiled, not plain Python.

Commit 20c0df1 ("port full PPP-RTK (CLAS) onto the minimal core") overwrote
mlambda.py with upstream's pure-Python version and deleted pppssr.py in
favour of a hand-merged gnssobs.py that carried none of its kernels. Twenty
njit kernels vanished from a repository named cssrlib-numba and no test
noticed, because a reverted kernel still computes the right answer -- just
tens to thousands of times slower.

Checking that the functions exist is not enough: the failure mode is a
kernel silently reverting to an ordinary function. So this asserts each one
is a real Numba dispatcher.
"""

import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher

import cssrlib.core.atmosphere as atmosphere
import cssrlib.core.geometry as geometry
import cssrlib.models.glonass as glonass
import cssrlib.engine.gnssobs as gnssobs
import cssrlib.core.mlambda as mlambda
import cssrlib.core.orbit as orbit

# module -> kernels that must stay JIT-compiled.
EXPECTED = {
    mlambda: ("_round_to_int", "_signed_step", "_sr_boost", "_ldldecom",
              "_reduction", "_msearch", "_estimILS"),
    gnssobs: ("_ddidx_core", "_tropmapf_dispatch_ppp", "_sdres_variance",
              "_sdres_core"),
}

# Modules that must keep at least this many kernels, from the counts that
# ship today. Named individually so a wholesale revert of any one of them is
# caught even if an individual kernel gets renamed.
MINIMUM_KERNELS = {
    geometry: 6,
    atmosphere: 4,
    orbit: 1,
    glonass: 3,
}


def _dispatchers(module):
    return {name: obj for name, obj in vars(module).items()
            if isinstance(obj, Dispatcher)}


@pytest.mark.parametrize(
    "module,name",
    [(m, n) for m, names in EXPECTED.items() for n in names],
    ids=lambda v: v if isinstance(v, str) else v.__name__.split(".")[-1])
def test_kernel_is_jit_compiled(module, name):
    obj = getattr(module, name, None)
    assert obj is not None, (
        f"{module.__name__}.{name} is gone -- a merge may have reverted the "
        f"Numba work, as 20c0df1 did")
    assert isinstance(obj, Dispatcher), (
        f"{module.__name__}.{name} is a plain Python function; the @njit "
        f"decorator was lost")


@pytest.mark.parametrize("module,least", list(MINIMUM_KERNELS.items()),
                         ids=lambda v: (v.__name__.split(".")[-1]
                                        if hasattr(v, "__name__") else str(v)))
def test_module_keeps_its_kernels(module, least):
    found = _dispatchers(module)
    assert len(found) >= least, (
        f"{module.__name__} has {len(found)} Numba kernels, expected at "
        f"least {least}: {sorted(found)}")


def test_total_kernel_count():
    """A floor across the library, so a broad revert cannot slip through.

    Counted over distinct dispatcher objects: gnssobs re-exports
    atmosphere's tropmapf_niell, and counting names would double it.
    """
    modules = set(EXPECTED) | set(MINIMUM_KERNELS)
    distinct = {id(obj) for m in modules for obj in _dispatchers(m).values()}
    assert len(distinct) >= 25, (
        f"only {len(distinct)} distinct Numba kernels across "
        f"{len(modules)} modules")


def test_lambda_error_is_catchable():
    """ldldecom must raise a catchable exception, never SystemExit.

    It raised SystemExit before a5a4300, which 20c0df1 then reverted: an
    external estimator doing AR had its process torn down by a
    non positive-definite covariance.
    """
    not_pos_def = np.array([[1.0, 0.0], [0.0, -1.0]])

    with pytest.raises(mlambda.LambdaError) as excinfo:
        mlambda.ldldecom(not_pos_def)

    assert isinstance(excinfo.value, np.linalg.LinAlgError)
    assert not isinstance(excinfo.value, SystemExit)


def test_scipy_backs_numba_linalg():
    """Numba's np.linalg binds to SciPy's BLAS/LAPACK at runtime.

    Nothing imports scipy, so a dependency audit by grep calls it unused;
    dropping it makes the njit'd geodist abort the interpreter with
    "Specified LAPACK function could not be found". Keep the coupling
    visible.
    """
    pytest.importorskip("scipy")

    rs = np.array([20.0e6, 10.0e6, 15.0e6])
    rr = np.array([-3962108.7, 3381309.5, 3668678.6])
    r, e = geometry.geodist(rs, rr)

    assert np.isfinite(r) and r > 0.0
    assert np.isclose(np.linalg.norm(e), 1.0)
