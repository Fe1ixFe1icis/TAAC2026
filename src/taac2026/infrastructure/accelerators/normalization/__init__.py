"""Normalization accelerator operator boundaries (Windows — torch backend only)."""

from __future__ import annotations

from taac2026.infrastructure.accelerators.normalization.layer_norm import (
	LayerNormBackend,
	LayerNormKernel,
	LayerNormKernelKey,
	clear_layer_norm_kernel_cache,
	layer_norm,
	register_layer_norm_kernel,
	resolved_layer_norm_backend,
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

__all__ = [
	"LayerNormBackend",
	"LayerNormKernel",
	"LayerNormKernelKey",
	"RMSNormBackend",
	"RMSNormKernel",
	"RMSNormKernelKey",
	"clear_layer_norm_kernel_cache",
	"clear_rms_norm_kernel_cache",
	"layer_norm",
	"register_layer_norm_kernel",
	"register_rms_norm_kernel",
	"resolved_layer_norm_backend",
	"resolved_rms_norm_backend",
	"rms_norm",
]
