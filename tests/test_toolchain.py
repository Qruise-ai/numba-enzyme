import os

import pytest

from numba_enzyme.toolchain import Toolchain, ToolchainError, get_toolchain


def test_get_toolchain_resolves_existing_executables():
    get_toolchain.cache_clear()
    tc = get_toolchain()
    assert isinstance(tc, Toolchain)
    for path in (tc.clang, tc.llvm_link, tc.opt):
        assert path.is_file()
        assert os.access(path, os.X_OK)
    assert tc.enzyme_plugin.is_file()
    get_toolchain.cache_clear()


def test_missing_tool_raises_toolchain_error(monkeypatch):
    get_toolchain.cache_clear()
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ToolchainError, match="clang-15"):
        get_toolchain()
    get_toolchain.cache_clear()


def test_plugin_path_override(monkeypatch, tmp_path):
    get_toolchain.cache_clear()
    fake_plugin = tmp_path / "LLVMEnzyme-15.so"
    fake_plugin.write_bytes(b"not a real plugin, just here for a stat check")
    monkeypatch.setenv("NUMBA_ENZYME_PLUGIN_PATH", str(fake_plugin))
    tc = get_toolchain()
    assert tc.enzyme_plugin == fake_plugin
    get_toolchain.cache_clear()
