from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


class DeterministicKVCacheTokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation
        values = [texts] if isinstance(texts, str) else list(texts)
        token_rows = [[(len(token) % 11) + 1 for token in text.split()] or [1] for text in values]
        if max_length is not None:
            token_rows = [row[:max_length] or [1] for row in token_rows]
        width = max(len(row) for row in token_rows)
        if padding:
            token_rows = [row + [0] * (width - len(row)) for row in token_rows]
        input_ids = torch.tensor(token_rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).to(dtype=torch.long),
        }


class DeterministicKVCacheModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls: list[bool] = []

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int,
        do_sample: bool,
        use_cache: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        del attention_mask
        assert do_sample is False
        self.generate_calls.append(use_cache)
        suffix = torch.full(
            (input_ids.shape[0], max_new_tokens),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, suffix), dim=-1)


def make_kv_cache_fixture_adapter() -> Any:
    model = DeterministicKVCacheModel()
    tokenizer = DeterministicKVCacheTokenizer()
    return SimpleNamespace(handle=SimpleNamespace(model=model, tokenizer=tokenizer))


def projected_kv_cache_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    use_cache: bool,
    num_layers: int = 2,
    num_heads: int = 2,
    head_dim: int = 4,
    bytes_per_value: int = 4,
) -> int:
    if not use_cache:
        return 0
    return batch_size * sequence_length * num_layers * num_heads * head_dim * bytes_per_value * 2
