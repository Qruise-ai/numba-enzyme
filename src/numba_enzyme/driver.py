"""
Synthesizes the Enzyme driver module via llvmlite.ir -- declares the
Numba kernel with its exact discovered type (no bitcast needed) and builds
the reverse-mode (__enzyme_autodiff) and forward-mode (__enzyme_fwddiff)
entry points with the correct per-argument activity markers.
"""
