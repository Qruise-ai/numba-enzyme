"""
Orchestrate and cache the full lowering-to-shared-object pipeline.

Wires lowering, driver synthesis, ``llvm-link``, the Enzyme ``opt``
pass, and the final shared-object compile into a single call, and
caches built ``.so`` files on disk, keyed on the function's source text
and a fingerprint of the toolchain used to build it.

See Also
--------
numba_enzyme.runtime.load : Loads the `BuiltKernel` this module produces.
numba_enzyme.core.grad : Public API built on top of this module.

Examples
--------
>>> from numba_enzyme.build import build
>>> from numba_enzyme.types import Float64
>>> def f(x: Float64) -> Float64:
...     return x * x
>>> build(f).n_args  # doctest: +SKIP
1
"""

import hashlib
import inspect
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from numba_enzyme.driver import synthesise
from numba_enzyme.lowering import lower
from numba_enzyme.toolchain import get_toolchain

_CACHE_DIR_ENV_VAR = "NUMBA_ENZYME_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "numba_enzyme"


@dataclass(frozen=True)
class BuiltKernel:
    """
    A compiled, cacheable shared object exposing grad/JVP entry points.

    Attributes
    ----------
    path : pathlib.Path
        Path to the built ``.so`` file.
    grad_symbol : str
        Name of the reverse-mode entry point exported by `path`.
    jvp_symbol : str
        Name of the forward-mode entry point exported by `path`.
    n_args : int
        Number of scalar arguments the original function takes.
    from_cache : bool
        Whether this result was served from the on-disk cache rather
        than freshly compiled.

    See Also
    --------
    build : Builds and validates a `BuiltKernel` instance.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> build(f).from_cache  # doctest: +SKIP
    False
    """

    path: Path
    grad_symbol: str
    jvp_symbol: str
    n_args: int
    from_cache: bool


def _cache_dir() -> Path:
    """
    Return the directory `build` caches compiled shared objects in.

    Defaults to ``~/.cache/numba_enzyme``, overridable via the
    ``NUMBA_ENZYME_CACHE_DIR`` environment variable.

    Returns
    -------
    pathlib.Path
        The cache root directory.

    Examples
    --------
    >>> from numba_enzyme.build import _cache_dir
    >>> _cache_dir()  # doctest: +SKIP
    PosixPath('/home/user/.cache/numba_enzyme')
    """
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_CACHE_DIR


def _toolchain_fingerprint() -> str:
    """
    Fingerprint the resolved toolchain by each tool's mtime and size.

    Used as part of the cache key so rebuilding any tool -- most
    notably the Enzyme plugin itself -- invalidates every cache entry
    automatically.

    Returns
    -------
    str
        A string encoding each tool's path, modification time, and
        size.

    See Also
    --------
    numba_enzyme.toolchain.get_toolchain : Resolves the tools fingerprinted
        here.

    Examples
    --------
    >>> from numba_enzyme.build import _toolchain_fingerprint
    >>> _toolchain_fingerprint()  # doctest: +SKIP
    '/usr/bin/clang-15:...|/usr/bin/llvm-link-15:...'
    """
    tc = get_toolchain()
    parts = []
    for path in (tc.clang, tc.llvm_link, tc.opt, tc.enzyme_plugin):
        stat = path.stat()
        parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _cache_key(func: Callable) -> str:
    """
    Compute `build`'s cache key for a function.

    Parameters
    ----------
    func : callable
        The function to key on.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest of `func`'s source text combined
        with the current `_toolchain_fingerprint`.

    Examples
    --------
    >>> from numba_enzyme.build import _cache_key
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> len(_cache_key(f))  # doctest: +SKIP
    64
    """
    digest_input = inspect.getsource(func) + "\n" + _toolchain_fingerprint()
    return hashlib.sha256(digest_input.encode()).hexdigest()


def build(func: Callable) -> BuiltKernel:
    """
    Build, or fetch from cache, the shared object for a function.

    Lowers `func` with Numba, synthesises its Enzyme driver, links the
    two with ``llvm-link``, runs the standalone Enzyme ``opt`` pass, and
    compiles the result to a shared object with ``clang`` -- unless an
    identical build (same source text and toolchain fingerprint) is
    already cached on disk, in which case that result is returned
    directly with no subprocess calls.

    Parameters
    ----------
    func : callable
        A Python function whose parameters and return value are each
        annotated with a `numba_enzyme.types` class.

    Returns
    -------
    BuiltKernel
        The compiled (or cached) shared object and its entry-point
        names.

    See Also
    --------
    BuiltKernel : The result this function returns.
    numba_enzyme.runtime.load : Loads the result into Python callables.

    Examples
    --------
    >>> from numba_enzyme.build import build
    >>> from numba_enzyme.types import Float64
    >>> def f(x: Float64) -> Float64:
    ...     return x * x
    >>> build(f).path.suffix  # doctest: +SKIP
    '.so'
    """
    entry_dir = _cache_dir() / _cache_key(func)
    so_path = entry_dir / "kernel.so"
    meta_path = entry_dir / "meta.json"

    if so_path.is_file() and meta_path.is_file():
        # Don't call lower() again here: Numba embeds an internal version
        # counter in the mangled symbol name that increments every time
        # nb.cfunc compiles "the same" function again in the same process
        # (e.g. ...B2v1... vs ...B2v2...), even with identical source.
        # Recomputing symbol names on a cache hit would silently drift from
        # what's actually embedded into the already-built .so. Persist them
        # from the original build instead.
        meta = json.loads(meta_path.read_text())
        return BuiltKernel(path=so_path, from_cache=True, **meta)

    kernel = lower(func)
    drv = synthesise(kernel)
    tc = get_toolchain()

    entry_dir.mkdir(parents=True, exist_ok=True)
    kernel_ll = entry_dir / "kernel.ll"
    driver_ll = entry_dir / "driver.ll"
    combined_ll = entry_dir / "combined.ll"
    enzyme_out_ll = entry_dir / "enzyme_out.ll"

    kernel_ll.write_text(kernel.ir)
    driver_ll.write_text(drv.ir)

    subprocess.run(
        [
            str(tc.llvm_link),
            str(kernel_ll),
            str(driver_ll),
            "-S",
            "-o",
            str(combined_ll),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(tc.opt),
            f"-load-pass-plugin={tc.enzyme_plugin}",
            "-passes=enzyme",
            "-S",
            str(combined_ll),
            "-o",
            str(enzyme_out_ll),
        ],
        check=True,
    )
    # `clang -shared` shells out to a separate linker executable for the
    # final link step. A vendored toolchain (see toolchain.py) ships its
    # own `ld.lld` right next to clang specifically so this works with no
    # system linker present at all -- confirmed necessary by a real
    # install on a bare-minimum target (python:3.11-slim has no system
    # `ld`/binutils: "clang: error: unable to execute command: Executable
    # 'ld' doesn't exist!"). The system/dev-mode toolchain doesn't need
    # this -- a normal dev machine already has a system `ld` -- so only
    # pass it when the vendored linker actually exists next to clang.
    vendored_lld = tc.clang.parent / "ld.lld"
    extra_link_args = [f"-fuse-ld={vendored_lld}"] if vendored_lld.is_file() else []

    # TODO: consider `-O3` at some point
    subprocess.run(
        [
            str(tc.clang),
            "-x",
            "ir",
            "-O2",
            "-fPIC",
            "-shared",
            *extra_link_args,
            str(enzyme_out_ll),
            "-o",
            str(so_path),
        ],
        check=True,
    )

    meta = {
        "grad_symbol": drv.grad_symbol,
        "jvp_symbol": drv.jvp_symbol,
        "n_args": kernel.n_args,
    }
    meta_path.write_text(json.dumps(meta))

    return BuiltKernel(path=so_path, from_cache=False, **meta)
