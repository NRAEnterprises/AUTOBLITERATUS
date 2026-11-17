from __future__ import annotations

import pytest

from autobliteratus.core.loader import FAMILY_TABLE
from autobliteratus.core.model_mapping import (
    AREA_CATEGORIES,
    MODEL_MAPPINGS,
    get_model_mapping,
    mapping_is_complete,
)


def test_every_loader_supported_model_type_has_a_labeled_mapping() -> None:
    assert set(FAMILY_TABLE) == set(MODEL_MAPPINGS)
    for model_type, family in FAMILY_TABLE.items():
        mapping = get_model_mapping(model_type)
        assert mapping is not None
        assert mapping.family == family
        assert mapping_is_complete(mapping)
        assert set(mapping.areas) == set(AREA_CATEGORIES)
        assert all(area.status in {"verified", "conditional", "unsupported"} for area in mapping.areas.values())


@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2", "gpt2", "opt", "mixtral", "mamba", "rwkv"])
def test_mapping_exposes_explicit_cache_or_recurrent_state_boundary(model_type: str) -> None:
    mapping = get_model_mapping(model_type)
    assert mapping is not None
    cache = mapping.areas["kv_cache"]
    recurrent = mapping.areas["position_or_recurrent_state"]
    assert cache.status in {"verified", "conditional", "unsupported"}
    assert recurrent.status in {"verified", "conditional", "unsupported"}
    assert cache.paths or recurrent.paths


def test_unknown_model_type_fails_closed() -> None:
    assert get_model_mapping("unknown-model-family") is None
