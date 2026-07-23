"""
Loads a built shared object via ctypes and exposes its gradient/JVP entry
points as plain Python callables. grad_<entry> is void and writes through
an explicit `out` pointer uniformly for every arity (see driver.py); jvp_
<entry> always returns a bare double regardless of arity.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable

from numba_enzyme.build import BuiltKernel


@dataclass(frozen=True)
class Differentiable:
    grad: Callable[..., tuple[float, ...]]
    jvp: Callable[[tuple[float, ...], tuple[float, ...]], float]
    n_args: int


def load(built: BuiltKernel) -> Differentiable:
    """
    ctypes-load `built.path` and wrap its grad_<entry>/jvp_<entry>
    symbols as plain Python callables.
    """
    lib = ctypes.CDLL(str(built.path))
    n = built.n_args

    grad_fn = getattr(lib, built.grad_symbol)
    grad_fn.restype = None
    grad_fn.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.c_double] * n

    jvp_fn = getattr(lib, built.jvp_symbol)
    jvp_fn.restype = ctypes.c_double
    jvp_fn.argtypes = [ctypes.c_double] * (2 * n)

    def grad(*xs: float) -> tuple[float, ...]:
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        out = (ctypes.c_double * n)()
        grad_fn(out, *xs)
        return tuple(out)

    def jvp(xs: tuple[float, ...], seed: tuple[float, ...]) -> float:
        if len(xs) != n or len(seed) != n:
            raise TypeError(f"expected {n} values for both xs and seed")
        interleaved = [v for pair in zip(xs, seed) for v in pair]
        return jvp_fn(*interleaved)

    return Differentiable(grad=grad, jvp=jvp, n_args=n)
