"""Accelerator-backed operator boundaries for PCVR kernels (Windows — torch backend only)."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.attention.flash_attention import (
    FlashAttentionBackend,
    FlashAttentionKernel,
    FlashAttentionKernelKey,
    FlashAttentionMaskPlan,
    _resolve_flash_attention_backend,
    clear_flash_attention_kernel_cache,
    flash_attention,
    register_flash_attention_kernel,
    resolved_flash_attention_backend,
)
from taac2026.infrastructure.accelerators.attention.gated_delta_rule import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_available,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)
from taac2026.infrastructure.accelerators.attention.mla import multi_latent_attention
from taac2026.infrastructure.accelerators.embedding.embedding_bag import (
    EmbeddingBagMeanBackend,
    EmbeddingBagMeanKernel,
    EmbeddingBagMeanKernelKey,
    clear_embedding_bag_mean_kernel_cache,
    embedding_bag_mean,
    register_embedding_bag_mean_kernel,
    resolved_embedding_bag_mean_backend,
)
from taac2026.infrastructure.accelerators.normalization.rms_norm import (
    RMSNormBackend,
    RMSNormKernel,
    RMSNormKernelKey,
    clear_rms_norm_kernel_cache,
    register_rms_norm_kernel,
    resolved_rms_norm_backend,
    rms_norm,
)
from taac2026.infrastructure.accelerators.normalization.layer_norm import (
    LayerNormBackend,
    LayerNormKernel,
    LayerNormKernelKey,
    clear_layer_norm_kernel_cache,
    layer_norm,
    register_layer_norm_kernel,
    resolved_layer_norm_backend,
)
from taac2026.infrastructure.accelerators.triton_runtime import triton_available
from taac2026.infrastructure.accelerators.tilelang_runtime import (
    cuda_multiprocessor_count,
    tilelang_available,
)
from taac2026.infrastructure.accelerators.tensor_validation import (
    require_cuda_tensors,
    require_same_device,
    require_same_dtype,
)


def clear_tilelang_kernel_cache() -> None:
    clear_embedding_bag_mean_kernel_cache()
    clear_flash_attention_kernel_cache()
    clear_layer_norm_kernel_cache()
    clear_rms_norm_kernel_cache()


__all__ = [
    "EmbeddingBagMeanBackend",
    "EmbeddingBagMeanKernel",
    "EmbeddingBagMeanKernelKey",
    "FlashAttentionBackend",
    "FlashAttentionKernel",
    "FlashAttentionKernelKey",
    "FlashAttentionMaskPlan",
    "LayerNormBackend",
    "LayerNormKernel",
    "LayerNormKernelKey",
    "RMSNormBackend",
    "RMSNormKernel",
    "RMSNormKernelKey",
    "_resolve_flash_attention_backend",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_available",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
    "clear_embedding_bag_mean_kernel_cache",
    "clear_flash_attention_kernel_cache",
    "clear_layer_norm_kernel_cache",
    "clear_rms_norm_kernel_cache",
    "clear_tilelang_kernel_cache",
    "cuda_multiprocessor_count",
    "embedding_bag_mean",
    "flash_attention",
    "layer_norm",
    "multi_latent_attention",
    "register_embedding_bag_mean_kernel",
    "register_flash_attention_kernel",
    "register_layer_norm_kernel",
    "register_rms_norm_kernel",
    "require_cuda_tensors",
    "require_same_device",
    "require_same_dtype",
    "resolved_embedding_bag_mean_backend",
    "resolved_flash_attention_backend",
    "resolved_layer_norm_backend",
    "resolved_rms_norm_backend",
    "rms_norm",
    "tilelang_available",
    "triton_available",
]
