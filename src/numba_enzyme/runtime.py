"""
Loads a built shared object via ctypes and exposes its gradient/JVP
entry points as plain Python callables, marshalling argtypes/restype
(including the multi-argument reverse-mode struct return) from the
function's signature.
"""
