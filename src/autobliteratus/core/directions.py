"""Refusal direction computation from paired harmful/harmless activations."""

from __future__ import annotations

import torch


def compute_refusal_directions(harmful, harmless):
    directions = {}
    norms = {}

    num_layers = min(harmful.num_layers, harmless.num_layers)
    for i in range(num_layers):
        h_mean = harmful.mean(i)
        l_mean = harmless.mean(i)
        if h_mean is None or l_mean is None:
            continue
        diff = h_mean - l_mean
        n = float(diff.norm().item())
        if n < 1e-8:
            continue
        directions[i] = diff / n
        norms[i] = n

    return {"directions": directions, "norms": norms}
