"""
Locates the external build tools this package drives: clang-15,
llvm-link-15, opt-15, and the standalone Enzyme LLVM pass plugin
(LLVMEnzyme-15.so). Raises a clear error naming whatever is missing
rather than failing deep inside a subprocess call.
"""
