"""Chunked gated-delta-rule attention operator boundary (Windows stub)."""

from __future__ import annotations


def chunk_gated_delta_rule_available() -> bool:
    return False


def chunk_gated_delta_rule(*args, **kwargs):
    raise RuntimeError("chunk_gated_delta_rule is not available on Windows")


def chunk_gated_delta_rule_fwd(*args, **kwargs):
    raise RuntimeError("chunk_gated_delta_rule_fwd is not available on Windows")


def chunk_gated_delta_rule_bwd(*args, **kwargs):
    raise RuntimeError("chunk_gated_delta_rule_bwd is not available on Windows")


__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_available",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule_fwd",
]
