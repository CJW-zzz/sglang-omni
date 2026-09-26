# SPDX-License-Identifier: Apache-2.0
"""Regression tests for repeated torch profiler starts."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import ModuleType

import pytest


class FakeProfiler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def torch_profiler_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_platforms = types.ModuleType("sglang_omni.platforms")
    fake_platforms.current_platform = types.SimpleNamespace(is_npu=lambda: False)
    monkeypatch.setitem(sys.modules, "sglang_omni.platforms", fake_platforms)
    sys.modules.pop("sglang_omni.profiler.torch_profiler", None)
    module = importlib.import_module("sglang_omni.profiler.torch_profiler")
    module.TorchProfiler.profiler = None
    module.TorchProfiler.active_run_id = None
    module.TorchProfiler.trace_template = ""
    yield module
    module.TorchProfiler.profiler = None
    module.TorchProfiler.active_run_id = None
    module.TorchProfiler.trace_template = ""


def test_start_same_run_id_returns_existing_trace(
    torch_profiler_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RANK", "3")
    profiler = torch_profiler_module.TorchProfiler
    profiler.profiler = FakeProfiler()
    profiler.active_run_id = "run-1"
    profiler.trace_template = str(tmp_path / "trace")

    assert profiler.start(str(tmp_path / "new-trace"), run_id="run-1") == (
        f"{tmp_path / 'trace'}_rank3.trace.json.gz"
    )


def test_start_different_run_id_restarts_existing_profiler(
    torch_profiler_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RANK", "3")
    profiler = torch_profiler_module.TorchProfiler
    existing_profiler = FakeProfiler()
    replacement_profiler = FakeProfiler()
    profiler.profiler = existing_profiler
    profiler.active_run_id = "run-1"
    profiler.trace_template = str(tmp_path / "old-trace")
    monkeypatch.setattr(
        torch_profiler_module,
        "profile",
        lambda **_: replacement_profiler,
    )

    trace_path = profiler.start(str(tmp_path / "new-trace"), run_id="run-2")

    assert existing_profiler.stopped
    assert replacement_profiler.started
    assert trace_path == f"{tmp_path / 'new-trace'}_rank3.trace.json.gz"
