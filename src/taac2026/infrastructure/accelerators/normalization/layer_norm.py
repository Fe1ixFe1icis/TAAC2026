"""LayerNorm operator boundary (Windows — torch backend only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


LayerNormKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
LayerNormBackend = Literal["torch"]

_layer_norm_kernel: LayerNormKernel | None = None


@dataclass(frozen=True, slots=True)
class LayerNormKernelKey:
    rows: int
    cols: int
    dtype: torch.dtype
    eps: float
    block_rows: int


def clear_layer_norm_kernel_cache() -> None:
    pass


def register_layer_norm_kernel(kernel: LayerNormKernel) -> None:
    global _layer_norm_kernel
    _layer_norm_kernel = kernel


def _torch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _normalize_layer_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("layer_norm expects input with at least 2 dimensions")
    if weight.ndim != 1:
        raise ValueError("layer_norm weight must be a 1D tensor")
    if bias.ndim != 1:
        raise ValueError("layer_norm bias must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"last dimension {x.shape[-1]} does not match weight size {weight.shape[0]}")
    if weight.shape != bias.shape:
        raise ValueError(f"layer_norm weight shape {tuple(weight.shape)} does not match bias shape {tuple(bias.shape)}")
    original_shape = x.shape
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    normalized_weight = weight.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    normalized_bias = bias.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    return matrix, normalized_weight, normalized_bias, original_shape


def _resolve_layer_norm_backend(x: torch.Tensor, backend: LayerNormBackend) -> Literal["torch"]:
    if backend == "torch":
        return "torch"
    raise ValueError(f"unsupported layer_norm backend on Windows: {backend}")


def _layer_norm_registered_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if _layer_norm_kernel is None:
        raise RuntimeError("no registered layer_norm kernel is available")
    return _layer_norm_kernel(x, weight, bias, eps)


def _run_torch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    if _layer_norm_kernel is not None:
        return _layer_norm_registered_kernel(x, weight, bias, eps)
    return _torch_layer_norm(x, weight, bias, eps)


def resolved_layer_norm_backend(
    x: torch.Tensor,
    backend: LayerNormBackend,
    *,
    eps: float = 1e-5,
    block_rows: int | None = None,
) -> Literal["torch"]:
    del eps, block_rows
    matrix = x.reshape(-1, x.shape[-1]).contiguous() if x.ndim >= 2 else x
    return _resolve_layer_norm_backend(matrix, backend)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    backend: LayerNormBackend,
    block_rows: int | None = None,
) -> torch.Tensor:
    del block_rows
    matrix, normalized_weight, normalized_bias, original_shape = _normalize_layer_norm_inputs(x, weight, bias)
    resolved_backend = resolved_layer_norm_backend(matrix, backend, eps=eps)
    if resolved_backend == "torch":
        return _run_torch_layer_norm(matrix, normalized_weight, normalized_bias, eps).reshape(original_shape)
    raise RuntimeError(f"unsupported layer_norm backend on Windows: {resolved_backend}")


__all__ = [
    "LayerNormBackend",
    "LayerNormKernel",
    "LayerNormKernelKey",
    "clear_layer_norm_kernel_cache",
    "layer_norm",
    "register_layer_norm_kernel",
    "resolved_layer_norm_backend",
]
