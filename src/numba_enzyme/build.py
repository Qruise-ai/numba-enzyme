"""
Orchestrates the full pipeline -- lowering, driver synthesis, llvm-link,
the Enzyme opt pass, and the final shared-object compile -- and caches
built .so files keyed on function source, signature, and toolchain
versions.
"""
