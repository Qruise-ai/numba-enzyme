"""
Load a built shared object and expose it as plain Python callables.

Wraps a `~numba_enzyme.build.BuiltKernel`'s ``grad_<entry>``/
``jvp_<entry>`` symbols with :mod:`ctypes`. ``grad_<entry>`` is void and
writes through an explicit output pointer uniformly for every arity (see
`numba_enzyme.driver`); ``jvp_<entry>`` always returns a bare `float`
regardless of arity.

See Also
--------
numba_enzyme.build.build : Produces the `BuiltKernel` this module loads.
numba_enzyme.core.grad : Public API built on top of this module.

Examples
--------
>>> from numba_enzyme.build import build
>>> from numba_enzyme.runtime import load
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> load(build(f)).grad(2.0)  # doctest: +SKIP
(4.0,)
"""

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

from numba_enzyme.build import BuiltKernel


@dataclass(frozen=True)
class Differentiable:
    """
    Plain Python callables wrapping a built kernel's grad/JVP symbols.

    Attributes
    ----------
    grad : callable
        Computes the reverse-mode gradient. Takes `n_args` positional
        arguments and returns a `tuple` of `n_args`; raises `TypeError`
        if called with the wrong number of arguments.
    jvp : callable
        Computes the forward-mode Jacobian-vector product. Takes a
        `tuple` of `n_args` primal values and a `tuple` of `n_args`
        tangent values, and returns a single `float`; raises
        `TypeError` if either tuple has the wrong length.
    n_args : int
        Number of scalar arguments the underlying function takes.

    See Also
    --------
    load : Builds a `Differentiable` instance from a `BuiltKernel`.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.runtime import load
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> load(build(f)).n_args  # doctest: +SKIP
    1
    """

    grad: Callable[..., tuple[float, ...]]
    jvp: Callable[[tuple[float, ...], tuple[float, ...]], float]
    n_args: int


def load(built: BuiltKernel) -> Differentiable:
    """
    Load a built kernel via ctypes and wrap it as plain Python callables.

    Parameters
    ----------
    built : numba_enzyme.build.BuiltKernel
        The compiled shared object to load.

    Returns
    -------
    Differentiable
        Plain Python callables wrapping `built`'s grad/JVP symbols.

    See Also
    --------
    Differentiable : The result this function returns.
    numba_enzyme.build.build : Produces the `built` this function
        consumes.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.runtime import load
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> load(build(f)).grad(2.0)  # doctest: +SKIP
    (4.0,)
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
        """
        Compute the reverse-mode gradient.

        Parameters
        ----------
        *xs : float
            The point(s) to differentiate at.

        Returns
        -------
        tuple of float
            The gradient with respect to each argument.

        Raises
        ------
        TypeError
            If the number of arguments given doesn't match `n`.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).grad(2.0)  # doctest: +SKIP
        (4.0,)
        """
        if len(xs) != n:
            raise TypeError(f"expected {n} arguments, got {len(xs)}")
        out = (ctypes.c_double * n)()
        grad_fn(out, *xs)
        return tuple(out)

    def jvp(xs: tuple[float, ...], seed: tuple[float, ...]) -> float:
        """
        Compute the forward-mode Jacobian-vector product.

        Parameters
        ----------
        xs : tuple of float
            The point to differentiate at, one value per argument.
        seed : tuple of float
            The tangent direction, one value per argument.

        Returns
        -------
        float
            The directional derivative of the underlying function at
            `xs` in direction `seed`.

        Raises
        ------
        TypeError
            If `xs` or `seed` doesn't have exactly `n` values.

        Examples
        --------
        >>> from numba_enzyme.build import build
        >>> from numba_enzyme.runtime import load
        >>> from numba_enzyme.types import Float64
        >>> def f(x: Float64) -> Float64:
        ...     return x * x
        >>> load(build(f)).jvp((2.0,), (1.0,))  # doctest: +SKIP
        4.0
        """
        if len(xs) != n or len(seed) != n:
            raise TypeError(f"expected {n} values for both xs and seed")
        interleaved = [v for pair in zip(xs, seed) for v in pair]
        return jvp_fn(*interleaved)

    return Differentiable(grad=grad, jvp=jvp, n_args=n)
