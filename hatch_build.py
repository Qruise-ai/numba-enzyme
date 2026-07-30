"""
Conditionally vendor pre-built binaries into the wheel.

`src/numba_enzyme/_vendor/` is populated at wheel-build time by
`packaging/cibw_before_build.sh` and is absent for a normal dev-mode
editable install. A static `force-include` in pyproject.toml can't
express "include only if present" -- hatchling requires force-included
files to exist unconditionally -- so this hook adds them dynamically,
only when `_vendor/` is actually there.
"""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VendorBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        vendor_dir = Path(self.root) / "src" / "numba_enzyme" / "_vendor"
        if not vendor_dir.is_dir():
            return

        force_include = build_data.setdefault("force_include", {})
        src_root = Path(self.root) / "src"
        for path in sorted(vendor_dir.rglob("*")):
            if path.is_file():
                force_include[str(path)] = str(path.relative_to(src_root))
