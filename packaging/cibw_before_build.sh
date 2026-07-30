#!/usr/bin/env bash
# CIBW_BEFORE_BUILD: runs once per Python version, before that
# version's wheel is built. Copies the binaries staged once by
# cibw_before_all.sh into src/numba_enzyme/_vendor/, where
# pyproject.toml's hatchling force-include picks them up for this
# specific wheel build.
#
# UNVERIFIED AS WRITTEN -- see packaging/Dockerfile.manylinux's header.
#
# Usage: cibw_before_build.sh <project-dir>
# (cibuildwheel substitutes {project} for <project-dir> automatically)

set -euo pipefail

PROJECT_DIR="${1:?usage: cibw_before_build.sh <project-dir>}"
STAGING_DIR="/tmp/numba_enzyme_vendor_staging"
VENDOR_DIR="$PROJECT_DIR/src/numba_enzyme/_vendor"

if [[ ! -d "$STAGING_DIR" ]]; then
    echo "error: $STAGING_DIR not found -- did cibw_before_all.sh run first?" >&2
    exit 1
fi

rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
cp -r "$STAGING_DIR/bin" "$VENDOR_DIR/bin"
cp -r "$STAGING_DIR/enzyme" "$VENDOR_DIR/enzyme"
chmod +x "$VENDOR_DIR/bin/"*

echo "== populated $VENDOR_DIR =="
find "$VENDOR_DIR" -type f -exec ls -lh {} \;
