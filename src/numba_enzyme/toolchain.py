"""
Locates the external build tools this package drives: clang-15,
llvm-link-15, opt-15, and the standalone Enzyme LLVM pass plugin
(LLVMEnzyme-15.so). Raises a clear error naming whatever is missing
rather than failing deep inside a subprocess call.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLUGIN_PATH = _REPO_ROOT / "enzyme-build" / "Enzyme" / "LLVMEnzyme-15.so"
_PLUGIN_PATH_ENV_VAR = "NUMBA_ENZYME_PLUGIN_PATH"


class ToolchainError(RuntimeError):
    """
    Raised when a required build tool cannot be located or used.
    """


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    llvm_link: Path
    opt: Path
    enzyme_plugin: Path


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


@lru_cache(maxsize=1)
def get_toolchain() -> Toolchain:
    """
    Resolve and validate all build tools.

    Cached after the first call -- use get_toolchain.cache_clear()
    to force re-resolution.
    """
    clang = _which("clang-15")
    llvm_link = _which("llvm-link-15")
    opt = _which("opt-15")

    plugin_override = os.environ.get(_PLUGIN_PATH_ENV_VAR)
    enzyme_plugin = Path(plugin_override) if plugin_override else _DEFAULT_PLUGIN_PATH

    missing = []
    if clang is None:
        missing.append("clang-15 (not found on PATH)")
    if llvm_link is None:
        missing.append("llvm-link-15 (not found on PATH)")
    if opt is None:
        missing.append("opt-15 (not found on PATH)")
    if not enzyme_plugin.is_file():
        missing.append(
            f"Enzyme plugin (not found at {enzyme_plugin}; "
            f"override with the {_PLUGIN_PATH_ENV_VAR} env var)"
        )
    if missing:
        raise ToolchainError(
            "missing required build tool(s):\n  - " + "\n  - ".join(missing)
        )

    for tool in (clang, llvm_link, opt):
        if not os.access(tool, os.X_OK):
            raise ToolchainError(f"{tool} exists but is not executable")
    if not os.access(enzyme_plugin, os.R_OK):
        raise ToolchainError(f"{enzyme_plugin} exists but is not readable")

    return Toolchain(
        clang=clang, llvm_link=llvm_link, opt=opt, enzyme_plugin=enzyme_plugin
    )
