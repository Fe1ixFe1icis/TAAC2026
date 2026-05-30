"""RMSNorm operator boundary (Windows — torch backend only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch


RMSNormKernel = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
RMSNormBackend = Literal["torch"]

_rms_norm_kernel: RMSNormKernel | None = None


@dataclass(frozen=True, slots=True)
class RMSNormKernelKey:
    rows: int
    cols: int
    dtype: torch.dtype
    eps: float
    block_rows: int


def clear_rms_norm_kernel_cache() -> None:
    pass


def register_rms_norm_kernel(kernel: RMSNormKernel) -> None:
    global _rms_norm_kernel
    _rms_norm_kernel = kernel


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * scale * weight


def _normalize_rms_norm_inputs(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("rms_norm expects input with at least 2 dimensions")
    if weight.ndim != 1:
        raise ValueError("rms_norm weight must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"last dimension {x.shape[-1]} does not match weight size {weight.shape[0]}")
    original_shape = x.shape
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    normalized_weight = weight.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    return matrix, normalized_weight, original_shape


def _resolve_rms_norm_backend(x: torch.Tensor, backend: RMSNormBackend) -> Literal["torch"]:
    if backend == "torch":
        return "torch"
    raise ValueError(f"unsupported rms_norm backend on Windows: {backend}")


def _rms_norm_registered_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _rms_norm_kernel is None:
        raise RuntimeError("no registered rms_norm kernel is available")
    return _rms_norm_kernel(x, weight, eps)


def _run_torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _rms_norm_kernel is not None:
        return _rms_norm_registered_kernel(x, weight, eps)
    return _torch_rms_norm(x, weight, eps)


def resolved_rms_norm_backend(
    x: torch.Tensor,
    backend: RMSNormBackend,
    *,
    eps: float = 1e-6,
    block_rows: int | None = None,
) -> Literal["torch"]:
    del eps, block_rows
    matrix = x.reshape(-1, x.shape[-1]).contiguous() if x.ndim >= 2 else x
    return _resolve_rms_norm_backend(matrix, backend)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    backend: RMSNormBackend,
    block_rows: int | None = None,
) -> torch.Tensor:
    del block_rows
    matrix, normalized_weight, original_shape = _normalize_rms_norm_inputs(x, weight)
    resolved_backend = resolved_rms_norm_backend(matrix, backend, eps=eps)
    if resolved_backend == "torch":
        return _run_torch_rms_norm(matrix, normalized_weight, eps).reshape(original_shape)
    raise RuntimeError(f"unsupported rms_norm backend on Windows: {resolved_backend}")


__all__ = [
    "RMSNormBackend",
    "RMSNormKernel",
    "RMSNormKernelKey",
    "clear_rms_norm_kernel_cache",
    "register_rms_norm_kernel",
    "resolved_rms_norm_backend",
    "rms_norm",
]
