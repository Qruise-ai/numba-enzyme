"""
Public API: the @differentiable decorator and grad()/jvp() functions
tying lowering, driver synthesis, build, and runtime together.
"""

from __future__ import annotations

import functools
from typing import Callable

from numba_enzyme.build import build
from numba_enzyme.runtime import load


def grad(func: Callable) -> Callable[..., tuple[float, ...]]:
    """
    Return a callable computing the gradient of `func` (a Python function
    annotated with numba_enzyme.types) with respect to all its
    arguments, via reverse-mode automatic differentiation.
    """
    return load(build(func)).grad


def jvp(func: Callable) -> Callable[[tuple, tuple], float]:
    """
    Return a callable computing the Jacobian-vector product of `func`:
    (primals, tangents) -> float, via forward-mode automatic
    differentiation.
    """
    return load(build(func)).jvp


class Differentiable:
    """
    Wraps a Python function: calling it runs the original Python code
    directly, while `.grad`/`.jvp` are built lazily (and cached) on first
    access -- so decorating a function costs nothing until you actually
    differentiate it.
    """

    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    @functools.cached_property
    def grad(self) -> Callable[..., tuple[float, ...]]:
        return grad(self._func)

    @functools.cached_property
    def jvp(self) -> Callable[[tuple, tuple], float]:
        return jvp(self._func)


def differentiable(func: Callable) -> Differentiable:
    """
    Mark `func` as differentiable, exposing `.grad`/`.jvp` alongside
    normal Python calls to `func` itself.
    """
    return Differentiable(func)
