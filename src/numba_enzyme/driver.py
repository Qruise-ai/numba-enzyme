"""
Synthesizes the Enzyme driver module via llvmlite.ir -- declares the
Numba kernel with its exact discovered type (no bitcast needed) and builds
the reverse-mode (__enzyme_autodiff) and forward-mode (__enzyme_fwddiff)
entry points with the correct per-argument activity markers.

Differentiates the kernel's retptr/excinfo entry point DIRECTLY (no
intermediate wrapper function performs a call), matching the Phase 0
lesson: an intermediate wrapper making a real call through a mismatched
bitcast crashes Enzyme's forward-mode generator. Here there is no
mismatch at all -- the kernel is declared with its exact real type, taken
straight from the validated LoweredKernel, so the only bitcast anywhere is
the universal "(void*)KERNEL" address-of pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from llvmlite import ir

from numba_enzyme.lowering import LoweredKernel

_EXCINFO_STRUCT = ir.LiteralStructType(
    [
        ir.IntType(8).as_pointer(),
        ir.IntType(32),
        ir.IntType(8).as_pointer(),
        ir.IntType(8).as_pointer(),
        ir.IntType(32),
    ]
)

_SCALAR_IR_TYPE = {
    "double": ir.DoubleType(),
    "float": ir.FloatType(),
    "i32": ir.IntType(32),
    "i64": ir.IntType(64),
}


@dataclass(frozen=True)
class SynthesizedDriver:
    ir: str
    grad_symbol: str
    jvp_symbol: str


def _target_lines(kernel_ir: str) -> tuple[str, str]:
    """Mirror Numba's own target triple/datalayout so llvm-link doesn't
    even emit its harmless mismatched-triple warning."""
    triple = datalayout = None
    for line in kernel_ir.splitlines():
        if line.startswith("target triple"):
            triple = line.split('"')[1]
        elif line.startswith("target datalayout"):
            datalayout = line.split('"')[1]
    return triple, datalayout


def synthesize(kernel: LoweredKernel) -> SynthesizedDriver:
    """
    Build the Enzyme driver module for `kernel`: a `grad_<entry>`
    (reverse-mode) and `jvp_<entry>` (forward-mode) function.
    """
    retptr_str, _excinfo_str, *scalar_strs = kernel.arg_types
    ret_scalar_type = _SCALAR_IR_TYPE[retptr_str[:-1]]
    arg_scalar_types = [_SCALAR_IR_TYPE[s] for s in scalar_strs]
    n_args = len(arg_scalar_types)

    module = ir.Module(name="numba_enzyme_driver")
    module.triple, module.data_layout = _target_lines(kernel.ir)

    excinfo_ptr_type = _EXCINFO_STRUCT.as_pointer()
    kernel_func_type = ir.FunctionType(
        ir.IntType(32),
        [ret_scalar_type.as_pointer(), excinfo_ptr_type.as_pointer(), *arg_scalar_types],
    )
    kernel_fn = ir.Function(module, kernel_func_type, name=kernel.entry_symbol)

    enzyme_dup = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_dup")
    enzyme_dup.linkage = "external"
    enzyme_const = ir.GlobalVariable(module, ir.IntType(32), name="enzyme_const")
    enzyme_const.linkage = "external"

    i8p = ir.IntType(8).as_pointer()
    grad_symbol = f"grad_{kernel.entry_symbol}"
    jvp_symbol = f"jvp_{kernel.entry_symbol}"

    # ---- reverse mode: grad_<entry> ----
    # Enzyme packs the gradients of multiple active by-value args into a
    # literal struct internally; a single active arg gets a bare scalar.
    # That struct never crosses an external ABI boundary: grad_<entry>
    # itself is void and takes an explicit `out` pointer, uniformly for
    # every n_args, so we never have to replicate the platform's small-
    # vs-large-aggregate return classification ourselves. (A first draft
    # returned the struct directly from grad_<entry> -- silently wrong
    # for n_args=3, a 24-byte struct exceeding x86-64 SysV's 16-byte
    # register-return threshold, since our hand-built IR never lowered
    # it to the required hidden-pointer/sret convention the way a real C
    # frontend would.)
    grad_ret_type = (
        ret_scalar_type if n_args == 1 else ir.LiteralStructType([ret_scalar_type] * n_args)
    )
    autodiff_fn = ir.Function(
        module, ir.FunctionType(grad_ret_type, [i8p], var_arg=True), name="__enzyme_autodiff"
    )

    out_ptr_type = ret_scalar_type.as_pointer()
    grad_fn = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), [out_ptr_type, *arg_scalar_types]),
        name=grad_symbol,
    )
    grad_fn.args[0].name = "out"
    for i, arg in enumerate(grad_fn.args[1:]):
        arg.name = f"x{i}"
    out_ptr, *grad_inputs = grad_fn.args
    b = ir.IRBuilder(grad_fn.append_basic_block("entry"))

    result_ptr = b.alloca(ret_scalar_type, name="result")
    b.store(ir.Constant(ret_scalar_type, 0.0), result_ptr)
    d_result_ptr = b.alloca(ret_scalar_type, name="d_result")
    b.store(ir.Constant(ret_scalar_type, 1.0), d_result_ptr)
    excinfo_local = b.alloca(excinfo_ptr_type, name="excinfo")
    b.store(ir.Constant(excinfo_ptr_type, None), excinfo_local)

    kernel_i8p = b.bitcast(kernel_fn, i8p)
    call_args = [
        kernel_i8p,
        b.load(enzyme_dup),
        result_ptr,
        d_result_ptr,
        b.load(enzyme_const),
        excinfo_local,
        *grad_inputs,
    ]
    grad_result = b.call(autodiff_fn, call_args)
    if n_args == 1:
        b.store(grad_result, out_ptr)
    else:
        for i in range(n_args):
            elem = b.extract_value(grad_result, i)
            elem_ptr = b.gep(out_ptr, [ir.Constant(ir.IntType(32), i)], inbounds=True)
            b.store(elem, elem_ptr)
    b.ret_void()

    # ---- forward mode: jvp_<entry> ----
    # Every active arg is a (primal, tangent) dup pair; the JVP always
    # lands in d_result's shadow regardless of n_args, matching Phase 0.
    jvp_arg_types = []
    for t in arg_scalar_types:
        jvp_arg_types.extend([t, t])
    fwddiff_fn = ir.Function(
        module, ir.FunctionType(ret_scalar_type, [i8p], var_arg=True), name="__enzyme_fwddiff"
    )

    jvp_fn = ir.Function(module, ir.FunctionType(ret_scalar_type, jvp_arg_types), name=jvp_symbol)
    for i in range(n_args):
        jvp_fn.args[2 * i].name = f"x{i}"
        jvp_fn.args[2 * i + 1].name = f"dx{i}"
    b2 = ir.IRBuilder(jvp_fn.append_basic_block("entry"))

    result_ptr2 = b2.alloca(ret_scalar_type, name="result")
    b2.store(ir.Constant(ret_scalar_type, 0.0), result_ptr2)
    d_result_ptr2 = b2.alloca(ret_scalar_type, name="d_result")
    b2.store(ir.Constant(ret_scalar_type, 0.0), d_result_ptr2)
    excinfo_local2 = b2.alloca(excinfo_ptr_type, name="excinfo")
    b2.store(ir.Constant(excinfo_ptr_type, None), excinfo_local2)

    kernel_i8p2 = b2.bitcast(kernel_fn, i8p)
    call_args2 = [
        kernel_i8p2,
        b2.load(enzyme_dup),
        result_ptr2,
        d_result_ptr2,
        b2.load(enzyme_const),
        excinfo_local2,
    ]
    for i in range(n_args):
        call_args2.append(b2.load(enzyme_dup))
        call_args2.append(jvp_fn.args[2 * i])
        call_args2.append(jvp_fn.args[2 * i + 1])
    b2.call(fwddiff_fn, call_args2)
    b2.ret(b2.load(d_result_ptr2))

    return SynthesizedDriver(ir=str(module), grad_symbol=grad_symbol, jvp_symbol=jvp_symbol)
