"""
End-to-end correctness tests for driver.py: build the Enzyme driver for a
lowered kernel (via build.py) and check the resulting gradients/JVPs
against hand-derived analytic derivatives, across several functions and
arities, both AD modes.
"""

import ctypes
import math

import pytest

from numba_enzyme.build import build
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


def _load(func):
    result = build(func)
    lib = ctypes.CDLL(str(result.path))
    n = result.n_args

    # grad_<entry> is void and writes through an explicit `out` pointer,
    # uniformly for every n -- sidesteps needing to replicate the
    # platform's small-vs-large-aggregate return-value ABI ourselves.
    grad_fn = getattr(lib, result.grad_symbol)
    grad_fn.restype = None
    grad_fn.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_double] * n

    jvp_fn = getattr(lib, result.jvp_symbol)
    jvp_fn.restype = ctypes.c_double
    jvp_fn.argtypes = [ctypes.c_double] * (2 * n)

    def grad(*xs):
        out = (ctypes.c_double * n)()
        grad_fn(out, *xs)
        return tuple(out)

    def jvp(xs, seed):
        interleaved = [v for pair in zip(xs, seed) for v in pair]
        return jvp_fn(*interleaved)

    return grad, jvp, n


@pytest.mark.parametrize("func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)])
def test_grad_matches_analytic(func, analytic_grad):
    grad, _jvp, n = _load(func)
    xs = tuple(1.0 + 0.3 * i for i in range(n))
    expected = analytic_grad(*xs)
    got = grad(*xs)
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize("func,analytic_grad", [(f1, f1_grad), (f2, f2_grad), (f3, f3_grad)])
def test_jvp_matches_analytic(func, analytic_grad):
    _grad, jvp, n = _load(func)
    xs = tuple(1.0 + 0.3 * i for i in range(n))
    expected = analytic_grad(*xs)
    for i in range(n):
        seed = tuple(1.0 if k == i else 0.0 for k in range(n))
        got = jvp(xs, seed)
        assert got == pytest.approx(expected[i], abs=1e-9)
