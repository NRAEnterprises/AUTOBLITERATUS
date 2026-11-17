from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import torch

from autobliteratus.tuning._44_memory_footprint_auto_tuning import _resident_bytes, run


class _Model(torch.nn.Module):
    def __init__(self, device_map=None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.hf_device_map = device_map or {}


class _Adapter:
    def __init__(self, model) -> None:
        self.handle = SimpleNamespace(model=model)


def test_resident_bytes_reads_proc_status() -> None:
    assert isinstance(_resident_bytes(), int)
    assert _resident_bytes() > 0


def test_resident_bytes_falls_back_to_psutil(monkeypatch) -> None:
    real_open = builtins.open

    def fail_proc_status(*args, **kwargs):
        if args and args[0] == "/proc/self/status":
            raise OSError("proc unavailable")
        return real_open(*args, **kwargs)

    psutil = ModuleType("psutil")
    psutil.Process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1234))
    monkeypatch.setattr(builtins, "open", fail_proc_status)
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    assert _resident_bytes() == 1234


def test_memory_stage_reports_cpu_measurement_with_shared_contract() -> None:
    context = {}
    report = run(_Adapter(_Model()), context)

    assert report["stage_id"] == "tuning.memory_footprint_auto"
    assert report["schema_version"] == 1
    assert report["status"] == "measured"
    assert report["accelerator_allocated_bytes"] == {}
    assert report["optimization_performed"] is False
    assert report["model_modified"] is False
    assert context["memory_report"] is report


def test_memory_stage_reports_cuda_allocation(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 4096)
    report = run(_Adapter(_Model({"": "cuda:0"})), {})

    assert report["accelerator_allocated_bytes"] == {"cuda:0": 4096}


def test_memory_stage_reports_allocation_failure_as_unknown(monkeypatch) -> None:
    def fail_allocation(device):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(torch.cuda, "memory_allocated", fail_allocation)
    report = run(_Adapter(_Model({"": "cuda:0"})), {})

    assert report["accelerator_allocated_bytes"] == {"cuda:0": None}


def test_memory_stage_supports_mps_no_argument_allocator(monkeypatch) -> None:
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 2048, raising=False)
    monkeypatch.setattr(torch.mps, "memory_allocated", None, raising=False)
    report = run(_Adapter(_Model({"": "mps"})), {})

    assert report["accelerator_allocated_bytes"] == {"mps": 2048}


def test_resident_bytes_returns_none_when_all_sources_fail(monkeypatch) -> None:
    def fail_open(*args, **kwargs):
        raise OSError("proc unavailable")

    monkeypatch.setattr(builtins, "open", fail_open)
    psutil = ModuleType("psutil")
    psutil.Process = lambda: (_ for _ in ()).throw(OSError("psutil unavailable"))
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    assert _resident_bytes() is None
