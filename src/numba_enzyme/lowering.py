"""
Lowers a Python function of N float64 scalar arguments to LLVM IR via
Numba, and recovers the mangled entry-point symbol (the retptr/excinfo-ABI
inner kernel, not the `cfunc.` wrapper) along with its exact parameter
types for use by driver synthesis.
"""
