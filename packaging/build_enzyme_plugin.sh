#!/usr/bin/env bash
# Clones the pinned Enzyme release and builds the standalone
# LLVMEnzyme-15.so pass plugin against system LLVM 15.
#
# Shared between packaging/cibw_before_all.sh (wheel-build vendoring)
# and .github/workflows/tests.yml (running the test suite in CI) so
# the pinned tag/CMake flags can't silently drift between the two.
#
# Usage: build_enzyme_plugin.sh <output-dir>
# Produces <output-dir>/Enzyme/LLVMEnzyme-15.so.

set -euo pipefail

OUTPUT_DIR="${1:?usage: build_enzyme_plugin.sh <output-dir>}"
ENZYME_TAG="v0.0.289"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "== cloning Enzyme @ $ENZYME_TAG =="
git clone --depth 1 --branch "$ENZYME_TAG" https://github.com/EnzymeAD/Enzyme.git "$WORK_DIR/enzyme-src"

echo "== configuring Enzyme against LLVM 15 =="
cmake -S "$WORK_DIR/enzyme-src/enzyme" -B "$OUTPUT_DIR" \
    -DLLVM_DIR=/usr/lib/llvm-15/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DENZYME_CLANG=OFF \
    -DENZYME_MLIR=OFF \
    -DENZYME_FORTRAN=OFF \
    -DENZYME_FLANG=OFF \
    -DENZYME_ENABLE_REACTANT=OFF \
    -DENZYME_ENABLE_BENCHMARKS=OFF

echo "== building LLVMEnzyme-15.so =="
cmake --build "$OUTPUT_DIR" --target LLVMEnzyme-15 -j"$(nproc)"
