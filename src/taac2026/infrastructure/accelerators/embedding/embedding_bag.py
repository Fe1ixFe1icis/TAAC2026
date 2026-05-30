"""Embedding fused operator boundary (Windows — torch backend only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from taac2026.infrastructure.accelerators.tensor_validation import require_same_device


EmbeddingBagMeanKernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EmbeddingBagMeanBackend = Literal["torch"]

_embedding_bag_mean_kernel: EmbeddingBagMeanKernel | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBagMeanKernelKey:
	batch: int
	bag_size: int
	num_embeddings: int
	emb_dim: int
	dtype: torch.dtype
	block_rows: int
	block_cols: int


def clear_embedding_bag_mean_kernel_cache() -> None:
	pass


def register_embedding_bag_mean_kernel(kernel: EmbeddingBagMeanKernel) -> None:
	global _embedding_bag_mean_kernel
	_embedding_bag_mean_kernel = kernel


def _torch_embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
	return F.embedding_bag(values, embedding_weight, mode="mean", padding_idx=0)


def _normalize_embedding_bag_mean_inputs(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
	if embedding_weight.ndim != 2:
		raise ValueError("embedding_bag_mean weight must be a 2D tensor")
	if values.ndim != 2:
		raise ValueError("embedding_bag_mean values must be a 2D tensor")
	require_same_device("embedding_bag_mean", embedding_weight, values)
	original_shape = values.shape
	normalized_weight = embedding_weight.contiguous()
	normalized_values = values.to(device=normalized_weight.device).contiguous()
	return normalized_weight, normalized_values, original_shape


def _resolve_embedding_bag_mean_backend(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	backend: EmbeddingBagMeanBackend,
) -> Literal["torch"]:
	if backend == "torch":
		return "torch"
	raise ValueError(f"unsupported embedding_bag_mean backend on Windows: {backend}")


def resolved_embedding_bag_mean_backend(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	backend: EmbeddingBagMeanBackend = "torch",
	*,
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> Literal["torch"]:
	del block_rows, block_cols
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	return _resolve_embedding_bag_mean_backend(normalized_weight, normalized_values, backend)


def _run_torch_embedding_bag_mean(embedding_weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
	if _embedding_bag_mean_kernel is not None:
		return _embedding_bag_mean_kernel(embedding_weight, values)
	return _torch_embedding_bag_mean(embedding_weight, values)


def embedding_bag_mean(
	embedding_weight: torch.Tensor,
	values: torch.Tensor,
	*,
	backend: EmbeddingBagMeanBackend = "torch",
	block_rows: int | None = None,
	block_cols: int | None = None,
) -> torch.Tensor:
	del block_rows, block_cols
	normalized_weight, normalized_values, _original_shape = _normalize_embedding_bag_mean_inputs(embedding_weight, values)
	resolved_backend = _resolve_embedding_bag_mean_backend(normalized_weight, normalized_values, backend)
	if resolved_backend == "torch":
		return _run_torch_embedding_bag_mean(normalized_weight, normalized_values)
	raise RuntimeError(f"unsupported embedding_bag_mean backend on Windows: {resolved_backend}")


__all__ = [
	"EmbeddingBagMeanBackend",
	"EmbeddingBagMeanKernel",
	"EmbeddingBagMeanKernelKey",
	"clear_embedding_bag_mean_kernel_cache",
	"embedding_bag_mean",
	"register_embedding_bag_mean_kernel",
	"resolved_embedding_bag_mean_backend",
]
