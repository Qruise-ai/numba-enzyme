"""
End-to-end correctness tests for driver.py: build the Enzyme driver for a
lowered kernel (via build.py + runtime.py) and check the resulting
gradients/JVPs against hand-derived analytic derivatives, across several
functions and arities, both AD modes.
"""

import math

import pytest

from numba_enzyme.build import build
from numba_enzyme.runtime import load
from numba_enzyme.types import Float64


def f1(x: Float64) -> Float64:
    return math.sin(x) * x + x * x


def f1_grad(x):
    return (math.cos(x) * x + math.sin(x) + 2 * x,)


def f2(x: Float64, y: Float64) -> Float64:
    return math.sin(x) * y + x * y * y


def f2_grad(x, y):
    return (math.cos(x) * y + y * y, math.sin(x) + 2 * x * y)


def f3(x: Float64, y: Float64, z: Float64) -> Float64:
    return x * y + y * z + math.exp(z)


def f3_grad(x, y, z):
    return (y, x + z, y + math.exp(z))


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUMBA_ENZYME_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)])
def test_grad_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    xs = tuple(1.0 + 0.3 * i for i in range(diff.n_args))
    expected = analytic_grad(*xs)
    got = diff.grad(*xs)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize("func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)])
def test_jvp_matches_analytic(func, analytic_grad):
    diff = load(build(func))
    n = diff.n_args
    xs = tuple(1.0 + 0.3 * i for i in range(n))
    expected = analytic_grad(*xs)
    for i in range(n):
        seed = tuple(1.0 if k == i else 0.0 for k in range(n))
        got = diff.jvp(xs, seed)
        assert got == pytest.approx(expected[i], abs=1e-9)
