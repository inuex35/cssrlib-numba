"""Regression for review finding 1: solve_dd_ambiguities must survive
parsearch's failure branch (default parmode=2, weak epoch).

Before the b.ndim guard this raised IndexError at ``b[:, 0]`` — a
production crash on any epoch weak enough that partial AR needs to
drop more than exclmax candidates.
"""
import numpy as np

from cssrlib.estimation.ambiguity import solve_dd_ambiguities


def _weak_problem(seed=7, n_amb=9, na=3):
    rng = np.random.default_rng(seed)
    nx = na + n_amb
    x = np.zeros(nx)
    x[na:] = rng.integers(-300, 300, n_amb) + rng.normal(0, 0.9, n_amb)
    P = np.zeros((nx, nx))
    P[:na, :na] = np.eye(na) * 0.01
    A = rng.normal(size=(n_amb, n_amb))
    P[na:, na:] = A @ A.T * 0.5 + np.eye(n_amb) * 0.3
    ref = na
    ix = np.array([[na + 1 + i, ref] for i in range(n_amb - 1)], dtype=int)
    return x, P, ix, na, nx


def test_parsearch_failure_does_not_crash():
    x, P, ix, na, nx = _weak_problem()
    sol = solve_dd_ambiguities(x, P, ix, na, nx,
                               parmode=2, P0=0.995, thresar=3.0)
    assert isinstance(sol.accepted, bool)
    bias = np.asarray(sol.bias)
    assert bias.ndim == 1 and bias.shape[0] == ix.shape[0]


def test_strong_problem_still_fixes():
    rng = np.random.default_rng(3)
    na, n_amb = 3, 6
    nx = na + n_amb
    truth = rng.integers(-200, 200, n_amb).astype(float)
    x = np.zeros(nx)
    x[na:] = truth + rng.normal(0, 0.005, n_amb)
    P = np.eye(nx) * 1e-4
    ix = np.array([[na + 1 + i, na] for i in range(n_amb - 1)], dtype=int)
    sol = solve_dd_ambiguities(x, P, ix, na, nx,
                               parmode=2, P0=0.995, thresar=3.0)
    assert sol.accepted
    dd_truth = truth[1 + np.arange(n_amb - 1)] - truth[0]
    assert np.allclose(np.round(sol.bias), dd_truth)
