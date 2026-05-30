"""Runtime support checks for the chunked gated-delta-rule kernels (Windows stub)."""

from __future__ import annotations

import torch


def chunk_gated_delta_rule_available(device: torch.device | None = None) -> bool:
    return False


def require_chunk_gated_delta_rule_runtime_support(*tensors: torch.Tensor | None) -> None:
    raise RuntimeError("chunk_gated_delta_rule requires tilelang which is not available on Windows")


__all__ = [
    "chunk_gated_delta_rule_available",
    "require_chunk_gated_delta_rule_runtime_support",
]
