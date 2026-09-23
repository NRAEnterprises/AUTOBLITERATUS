"""Linear CKA between two activation matrices."""

from __future__ import annotations

import torch


def linear_cka(a, b):
    if a is None or b is None:
        return 0.0
    a = a.float()
    b = b.float()
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cross = (b.T @ a).norm() ** 2
    denom = (a.T @ a).norm() * (b.T @ b).norm()
    if denom == 0:
        return 0.0
    return float((cross / denom).item())
