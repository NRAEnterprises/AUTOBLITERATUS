from __future__ import annotations

import pytest

from autobliteratus.tuning._06_auto_ablated_capability import AXES, run
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Adapter:
    def __init__(self, scores):
        self.scores = scores

    def capability(self, axes):
        return {axis: self.scores[axis] for axis in axes}


def test_phase_a_capability_backend_is_callable_and_opt_in() -> None:
    assert TOGGLES["tuning.auto_ablated_capability"]["available"] is True
    assert TOGGLES["tuning.auto_ablated_capability"]["default"] is False


def test_capability_battery_validates_and_namespaces_scores() -> None:
    context = {"capability_axes": ["coherence", "math"]}
    report = run(_Adapter({"coherence": 0.75, "math": 0.5}), context)
    assert report["status"] == "measured"
    assert report["scores"] == {"coherence": 0.75, "math": 0.5}
    assert report["model_modified"] is False
    assert context["capability_battery_report"] is report


def test_capability_battery_rejects_unknown_duplicate_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="Unsupported capability"):
        run(_Adapter({}), {"capability_axes": ["unknown"]})
    with pytest.raises(ValueError, match="duplicates"):
        run(_Adapter({"math": 0.5}), {"capability_axes": ["math", "math"]})
    with pytest.raises(ValueError, match="finite"):
        run(_Adapter({"math": float("nan")}), {"capability_axes": ["math"]})


def test_capability_battery_default_axis_contract_is_explicit() -> None:
    scores = {axis: 0.0 for axis in AXES}
    report = run(_Adapter(scores), {})
    assert report["axes"] == list(AXES)
    assert report["sample_count"] == len(AXES)
