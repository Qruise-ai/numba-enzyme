"""
Locate the required external build tools.

Resolves the ``clang``/``llvm-link``/``opt`` binaries
and the standalone Enzyme LLVM pass plugin.

See Also
--------
numba_enzyme.build.build : Uses these tools to compile
    a differentiated kernel.

Examples
--------
>>> from numba_enzyme.toolchain import get_toolchain
>>> toolchain = get_toolchain()  # doctest: +SKIP
>>> toolchain.clang.name  # doctest: +SKIP
'clang-15'
"""

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
    Raise when a required build tool cannot be located or used.

    This covers both a missing tool (not found on ``PATH``, or the
    Enzyme plugin not found at its configured path) and a tool that
    exists on disk but is not executable or readable.

    See Also
    --------
    get_toolchain : Raises this error when tool resolution fails.

    Examples
    --------
    >>> from numba_enzyme.toolchain import ToolchainError, get_toolchain
    >>> try:
    ...     get_toolchain()
    ... except ToolchainError as exc:
    ...     print(exc)  # doctest: +SKIP
    """


@dataclass(frozen=True)
class Toolchain:
    """
    Resolved, validated paths to the build tools.

    Attributes
    ----------
    clang : pathlib.Path
        Path to the ``clang-15`` executable.
    llvm_link : pathlib.Path
        Path to the ``llvm-link-15`` executable.
    opt : pathlib.Path
        Path to the ``opt-15`` executable.
    enzyme_plugin : pathlib.Path
        Path to the standalone Enzyme LLVM pass plugin (``LLVMEnzyme-15.so``).

    See Also
    --------
    get_toolchain : Builds and validates a `Toolchain` instance.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain().opt.name  # doctest: +SKIP
    'opt-15'
    """

    clang: Path
    llvm_link: Path
    opt: Path
    enzyme_plugin: Path


def _which(name: str) -> Path | None:
    """
    Resolve an executable's absolute path in ``PATH``.

    Parameters
    ----------
    name : str
        Name of the executable to look up, e.g. ``"clang-15"``.

    Returns
    -------
    pathlib.Path or None
        The resolved absolute path, or `None` if `name` is not found in
        ``PATH``.

    Examples
    --------
    >>> from numba_enzyme.toolchain import _which
    >>> _which("nonexistent-tool-xyz") is None
    True
    """
    found = shutil.which(name)
    return Path(found) if found else None


@lru_cache(maxsize=1)
def get_toolchain() -> Toolchain:
    """
    Resolve and validate every build tool.

    Looks up ``clang-15``, ``llvm-link-15``, and ``opt-15`` in ``PATH``,
    and the standalone Enzyme LLVM pass plugin at a path relative to the
    repository root (overridable via the ``NUMBA_ENZYME_PLUGIN_PATH``
    environment variable). The result is cached after the first
    successful call.

    Returns
    -------
    Toolchain
        The resolved, validated paths of the build tools.

    Raises
    ------
    ToolchainError
        If any tool is missing, or exists but is not
        executable/readable.

    See Also
    --------
    Toolchain : The return resolved paths.

    Examples
    --------
    >>> from numba_enzyme.toolchain import get_toolchain
    >>> get_toolchain()  # doctest: +SKIP
    Toolchain(clang=..., llvm_link=..., opt=..., enzyme_plugin=...)

    Force re-resolution (e.g. in tests that change the environment)::

    >>> get_toolchain.cache_clear()
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
