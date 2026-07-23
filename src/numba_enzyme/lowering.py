"""
Lowers a Python function to LLVM IR via Numba, using the function's own
type annotations (numba_enzyme.types classes) to build the Numba
signature, and recovers the mangled entry-point symbol (the
retptr/excinfo-ABI inner kernel, not the `cfunc.` wrapper) along with its
exact parameter types for use by driver synthesis.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Callable

import llvmlite.binding as llvm_binding
import numba as nb

_CFUNC_PREFIX = "cfunc."

# Numba's fixed internal exception-info representation. Stable across
# signatures/arities in principle, but re-checked on every lowering
# rather than assumed -- this is the ABI this whole package leans on.
_EXPECTED_EXCINFO_TYPE = "{ i8*, i32, i8*, i8*, i32 }**"

# LLVM textual type for each scalar numba type this package accepts as
# an annotation (see numba_enzyme.types).
_LLVM_SCALAR_TYPE = {
    nb.types.float64: "double",
    nb.types.float32: "float",
    nb.types.int32: "i32",
    nb.types.int64: "i64",
}

llvm_binding.initialize()


class LoweringError(RuntimeError):
    """
    Raised when Numba's emitted IR doesn't match the retptr/excinfo
    entry-point ABI this package relies on.
    """


@dataclass(frozen=True)
class LoweredKernel:
    ir: str
    entry_symbol: str
    n_args: int
    arg_types: tuple[str, ...]  # retptr, excinfo, then n_args scalar types


def _numba_type_of(annotation) -> nb.types.Type:
    try:
        return annotation()
    except TypeError as exc:
        raise LoweringError(
            f"annotation {annotation!r} is not a numba_enzyme.types type"
        ) from exc


def _llvm_scalar_type(numba_type: nb.types.Type) -> str:
    try:
        return _LLVM_SCALAR_TYPE[numba_type]
    except KeyError:
        raise LoweringError(
            f"unsupported scalar type {numba_type!r}; supported: "
            f"{sorted(str(t) for t in _LLVM_SCALAR_TYPE)}"
        ) from None


def lower(func: Callable) -> LoweredKernel:
    """
    Compile `func` to LLVM IR via Numba, using its parameter/return type
    annotations (numba_enzyme.types classes) to build the signature, then
    locate and validate its retptr/excinfo entry point.
    """
    hints = typing.get_type_hints(func)
    params = inspect.signature(func).parameters

    try:
        arg_numba_types = [_numba_type_of(hints[name]) for name in params]
        ret_numba_type = _numba_type_of(hints["return"])
    except KeyError as exc:
        raise LoweringError(
            f"{func!r} is missing a type annotation for {exc.args[0]!r}"
        ) from exc

    sig = ret_numba_type(*arg_numba_types)
    compiled = nb.cfunc(sig, error_model="numpy")(func)
    ir_text = compiled.inspect_llvm()

    native_name = compiled.native_name
    if not native_name.startswith(_CFUNC_PREFIX):
        raise LoweringError(
            f"expected native_name to start with {_CFUNC_PREFIX!r}, got {native_name!r}"
        )
    entry_symbol = native_name[len(_CFUNC_PREFIX) :]

    mod = llvm_binding.parse_assembly(ir_text)
    mod.verify()
    fn = next((f for f in mod.functions if f.name == entry_symbol), None)
    if fn is None:
        raise LoweringError(f"parsed IR has no function named {entry_symbol!r}")

    args = list(fn.arguments)
    n_args = len(arg_numba_types)
    expected_n_params = n_args + 2
    if len(args) != expected_n_params:
        raise LoweringError(
            f"expected {expected_n_params} parameters (retptr, excinfo, "
            f"{n_args} scalar args) but {entry_symbol!r} has {len(args)}"
        )

    retptr, excinfo, *scalar_args = args

    expected_retptr_type = _llvm_scalar_type(ret_numba_type) + "*"
    if retptr.name != "retptr" or str(retptr.type) != expected_retptr_type:
        raise LoweringError(
            f"expected first parameter 'retptr: {expected_retptr_type}', "
            f"got {retptr.name!r}: {retptr.type}"
        )
    if excinfo.name != "excinfo" or str(excinfo.type) != _EXPECTED_EXCINFO_TYPE:
        raise LoweringError(
            f"expected second parameter 'excinfo: {_EXPECTED_EXCINFO_TYPE}', "
            f"got {excinfo.name!r}: {excinfo.type}"
        )
    for arg, numba_type in zip(scalar_args, arg_numba_types):
        expected = _llvm_scalar_type(numba_type)
        if str(arg.type) != expected:
            raise LoweringError(
                f"expected parameter {arg.name!r} to be {expected}, got {arg.type}"
            )

    arg_types = tuple(str(a.type) for a in args)
    return LoweredKernel(
        ir=ir_text, entry_symbol=entry_symbol, n_args=n_args, arg_types=arg_types
    )
