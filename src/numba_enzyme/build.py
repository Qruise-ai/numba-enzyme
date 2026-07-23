"""
Orchestrates the full pipeline -- lowering, driver synthesis, llvm-link,
the Enzyme opt pass, and the final shared-object compile -- and caches
built .so files keyed on function source and toolchain fingerprint.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from numba_enzyme.driver import synthesize
from numba_enzyme.lowering import lower
from numba_enzyme.toolchain import get_toolchain

_CACHE_DIR_ENV_VAR = "NUMBA_ENZYME_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "numba_enzyme"


@dataclass(frozen=True)
class BuiltKernel:
    path: Path
    grad_symbol: str
    jvp_symbol: str
    n_args: int
    from_cache: bool


def _cache_dir() -> Path:
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    return Path(override) if override else _DEFAULT_CACHE_DIR


def _toolchain_fingerprint() -> str:
    tc = get_toolchain()
    parts = []
    for path in (tc.clang, tc.llvm_link, tc.opt, tc.enzyme_plugin):
        stat = path.stat()
        parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _cache_key(func: Callable) -> str:
    digest_input = inspect.getsource(func) + "\n" + _toolchain_fingerprint()
    return hashlib.sha256(digest_input.encode()).hexdigest()


def build(func: Callable) -> BuiltKernel:
    """
    Build (or fetch from cache) the shared object exposing grad_<entry>
    and jvp_<entry> for `func`.
    """
    entry_dir = _cache_dir() / _cache_key(func)
    so_path = entry_dir / "kernel.so"
    meta_path = entry_dir / "meta.json"

    if so_path.is_file() and meta_path.is_file():
        # Don't call lower() again here: Numba embeds an internal version
        # counter in the mangled symbol name that increments every time
        # nb.cfunc compiles "the same" function again in the same process
        # (confirmed empirically -- e.g. ...B2v1... vs ...B2v2...), even
        # with identical source. Recomputing symbol names on a cache hit
        # would silently drift from what's actually baked into the
        # already-built .so. Persist them from the original build instead.
        meta = json.loads(meta_path.read_text())
        return BuiltKernel(path=so_path, from_cache=True, **meta)

    kernel = lower(func)
    drv = synthesize(kernel)
    tc = get_toolchain()

    entry_dir.mkdir(parents=True, exist_ok=True)
    kernel_ll = entry_dir / "kernel.ll"
    driver_ll = entry_dir / "driver.ll"
    combined_ll = entry_dir / "combined.ll"
    enzyme_out_ll = entry_dir / "enzyme_out.ll"

    kernel_ll.write_text(kernel.ir)
    driver_ll.write_text(drv.ir)

    subprocess.run(
        [str(tc.llvm_link), str(kernel_ll), str(driver_ll), "-S", "-o", str(combined_ll)],
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
    subprocess.run(
        [str(tc.clang), "-x", "ir", "-O2", "-fPIC", "-shared", str(enzyme_out_ll), "-o", str(so_path)],
        check=True,
    )

    meta = {
        "grad_symbol": drv.grad_symbol,
        "jvp_symbol": drv.jvp_symbol,
        "n_args": kernel.n_args,
    }
    meta_path.write_text(json.dumps(meta))

    return BuiltKernel(path=so_path, from_cache=False, **meta)
