"""Shared Triton runtime discovery helpers (Windows stub)."""

from __future__ import annotations

import torch

tl = None
triton = None


def triton_available() -> bool:
    return False


def triton_next_power_of_2(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def triton_num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    return 1


def triton_supported_floating_dtype(dtype: torch.dtype) -> bool:
    return dtype in {torch.float16, torch.bfloat16, torch.float32}


__all__ = [
    "tl",
    "triton",
    "triton_available",
    "triton_next_power_of_2",
    "triton_num_warps",
    "triton_supported_floating_dtype",
]
