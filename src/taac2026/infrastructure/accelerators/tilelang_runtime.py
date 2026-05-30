"""Shared TileLang runtime discovery and compatibility helpers (Windows stub)."""

from __future__ import annotations

import torch

T = None
tl = None


def tilelang_available() -> bool:
    return False


def cuda_multiprocessor_count(device: torch.device | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None
    resolved_device = device
    if resolved_device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(resolved_device).multi_processor_count)
    except Exception:
        return None


def _ensure_tilelang_cuda_fp8_compatibility(*, tilelang_header=None, cuda_header_paths=None) -> bool:
    return False


def tilelang_dtype(dtype: torch.dtype):
    raise RuntimeError("tilelang is not available on Windows")


__all__ = [
    "T",
    "cuda_multiprocessor_count",
    "tilelang_available",
    "tilelang_dtype",
    "tl",
]
