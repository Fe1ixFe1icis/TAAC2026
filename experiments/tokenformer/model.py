"""TokenFormer: unified token stream for multi-field + sequential recommendation.

Implements the architecture from "TokenFormer: Unify the Multi-Field and
Sequential Recommendation Worlds" (Tencent, arXiv 2604.13737):
  - Unified token stream  X = [F | <SEP> | T | <SEP> | V]
  - UIB with BFTS attention, NLIR, SwiGLU FFN with RMSNorm

This is a drop-in replacement for PCVRHyFormer — accepts the same
ModelInput contract and returns the same (logits) / (logits, extras) signatures.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from baseline model (relative import within experiments package)
import sys
from pathlib import Path
_baseline_path = Path(__file__).resolve().parent.parent / "baseline"
if str(_baseline_path) not in sys.path:
    sys.path.insert(0, str(_baseline_path))

from model import (
    ModelInput,
    PCVRHyFormer,
    RankMixerNSTokenizer,
    GroupNSTokenizer,
    RotaryEmbedding,
    SwiGLU,
    apply_rope_to_tensor,
    rotate_half,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Llama-style RMSNorm.

    Normalizes by RMS along the last dimension and applies a learnable per-channel
    gain. No bias, no mean-subtraction.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (Eq. 19): H = (Swish(I~ W1) ⊙ (I~ W2)) W3."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PerFieldResSwiGLUTokenizer(nn.Module):
    """Per-field tokenization with Residual SwiGLU + Gating per token.

    Each fid gets 1 token:
      emb = emb_layer[fid](value)              # (B, emb_dim)
      I   = optional Linear(emb)               # (B, d_model)
      gate = sigmoid(Linear(emb))              # (B, 1) per-field gate
      tok = gate * I + (1 - gate) * fallback   # gated mixing
      tok = tok + SwiGLU(RMSNorm(tok))         # residual LLM-style block

    Skipped fids (vs ≤ 0 or vs > emb_skip_threshold) → shared learnable fallback token.
    """

    def __init__(
        self,
        ns_tokenizer: "RankMixerNSTokenizer",
        d_model: int,
        hidden_mult: int = 4,
        use_gating: bool = True,
    ) -> None:
        super().__init__()
        # Share emb tables (sparse params compatible)
        self.embs: nn.ModuleList = ns_tokenizer.embs
        self._emb_index: List[int] = list(ns_tokenizer._emb_index)
        self.feature_specs: List[Tuple[int, int, int]] = list(ns_tokenizer.feature_specs)
        self.d_model = d_model
        self.use_gating = use_gating
        emb_dim = ns_tokenizer.emb_dim
        self.proj: Optional[nn.Linear] = (
            nn.Linear(emb_dim, d_model, bias=False) if emb_dim != d_model else None
        )
        # Shared ResSwiGLU block across all fids
        self.res_norm = RMSNorm(d_model)
        self.res_ffn = SwiGLUFFN(d_model, hidden_mult=hidden_mult)
        # Fallback token for skipped fids
        self.fallback = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.fallback, std=0.02)
        # Per-field gating: gate = sigmoid(Linear(emb))
        # Initialize bias to 2.0 so sigmoid(2.0) ≈ 0.88, gate starts near "pass-through"
        if use_gating:
            self.gate_proj = nn.Linear(emb_dim, 1, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)
        else:
            self.gate_proj = None

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            int_feats: (B, total_int_dim)
        Returns:
            (B, num_fids, d_model)
        """
        B = int_feats.shape[0]
        tokens: List[torch.Tensor] = []
        for fid_idx, (vs, offset, length) in enumerate(self.feature_specs):
            real_idx = self._emb_index[fid_idx]
            if real_idx < 0:
                tok = self.fallback.to(int_feats.dtype).expand(B, 1, -1)
            else:
                values = int_feats[:, offset:offset + length].long().clamp(min=0, max=int(vs))
                emb_out = self.embs[real_idx](values)         # (B, length, emb_dim)
                if length > 1:
                    emb_out = emb_out.mean(dim=1, keepdim=True)  # (B, 1, emb_dim)

                # Gating: gate = sigmoid(Linear(emb))
                if self.gate_proj is not None and self.use_gating:
                    gate = torch.sigmoid(self.gate_proj(emb_out))  # (B, 1, 1)
                    # Project embedding to d_model
                    if self.proj is not None:
                        proj_out = self.proj(emb_out)              # (B, 1, d_model)
                    else:
                        proj_out = emb_out
                    # Gated mixing: gate * proj + (1 - gate) * fallback
                    tok = gate * proj_out + (1 - gate) * self.fallback.to(int_feats.dtype)
                else:
                    if self.proj is not None:
                        tok = self.proj(emb_out)                   # (B, 1, d_model)
                    else:
                        tok = emb_out
            tokens.append(tok)
        I = torch.cat(tokens, dim=1)                          # (B, num_fids, d_model)
        # ResSwiGLU: token = I + SwiGLU(RMSNorm(I))
        out = I + self.res_ffn(self.res_norm(I))
        return out


def apply_rope_per_sample(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation with per-sample positions.

    Args:
        x: (B, H, N, HD)
        cos: (B, N, HD)  per-sample, per-position cosine values.
        sin: (B, N, HD)

    Returns:
        Rotated tensor (B, H, N, HD).
    """
    cos_ = cos.unsqueeze(1)
    sin_ = sin.unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# TokenFormer Block
# ═══════════════════════════════════════════════════════════════════════════════


class TokenFormerBlock(nn.Module):
    """Unified Interaction Block with optional OneTrans mixed parameters.

    Layout (Pre-Norm):
        I_tilde = sigmoid(X W_G) * Attn(RMSNorm(X), mask=BFTS)
        I       = X + I_tilde
        H       = SwiGLU(RMSNorm(I))
        X(l+1)  = I + H

    OneTrans mixed params (optional):
        - F-tokens: per-field independent Q/K/V (heterogeneous)
        - T-tokens: shared Q/K/V (homogeneous)
        - V-token: independent Q/K/V
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        window_size: Optional[int] = None,
        discard_F: bool = False,
        mixed_params: bool = False,
        num_F: int = 0,
        num_V: int = 1,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.discard_F = discard_F
        self.dropout = dropout
        self.mixed_params = mixed_params
        self.num_F = num_F
        self.num_V = num_V

        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)

        # Shared Q/K/V for T-tokens (homogeneous)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # OneTrans: independent Q/K/V for F-tokens (heterogeneous)
        if mixed_params and num_F > 0:
            self.F_W_q = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_F)])
            self.F_W_k = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_F)])
            self.F_W_v = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(num_F)])
        else:
            self.F_W_q = None
            self.F_W_k = None
            self.F_W_v = None

        # OneTrans: independent Q/K/V for V-token
        if mixed_params and num_V > 0:
            self.V_W_q = nn.Linear(d_model, d_model, bias=False)
            self.V_W_k = nn.Linear(d_model, d_model, bias=False)
            self.V_W_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.V_W_q = None
            self.V_W_k = None
            self.V_W_v = None

        # NLIR gate projection (Eq. 16)
        self.W_g = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

        self.ffn = SwiGLUFFN(d_model, hidden_mult)

    def _build_attn_mask(
        self,
        N: int,
        num_F: int,
        padding_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Constructs the additive float mask for SDPA.

        Components:
          1. Causal (j <= i).
          2. SWA window: i - j < window_size.
          3. Non-Sequence Token Discarding: discard_F=True, F cols masked for non-F rows.
          4. Padding (KV-side): masked positions cannot be attended.
        """
        B = padding_mask.shape[0]

        # Causal + SWA mask (N, N)
        i_idx = torch.arange(N, device=device).unsqueeze(1)
        j_idx = torch.arange(N, device=device).unsqueeze(0)
        causal_ok = (j_idx <= i_idx)
        if self.window_size is not None:
            window_ok = ((i_idx - j_idx) < self.window_size)
            visible = causal_ok & window_ok
        else:
            visible = causal_ok

        # Non-sequence token discarding (Eq. 15)
        if self.discard_F and num_F > 0:
            row_is_non_F = (i_idx >= num_F)
            col_is_F = (j_idx < num_F)
            discard = row_is_non_F & col_is_F
            visible = visible & (~discard)

        # F-F full attention (bidirectional)
        if num_F > 0:
            both_F = (i_idx < num_F) & (j_idx < num_F)
            visible = visible | both_F

        # Convert to additive float
        mask_2d = torch.zeros((N, N), device=device, dtype=dtype)
        mask_2d = mask_2d.masked_fill(~visible, float('-inf'))
        mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N).clone()

        # KV-side padding mask
        invalid_kv = (~padding_mask).unsqueeze(1).unsqueeze(2)
        mask = mask.masked_fill(invalid_kv, float('-inf'))

        return mask

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        num_F: int,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        v_start: int = 0,
        v_end: int = 0,
    ) -> torch.Tensor:
        """One UIB pass.

        Args:
            x: (B, N, D)
            padding_mask: (B, N) bool, True = valid token.
            num_F: number of static field tokens at the prefix.
            rope_cos: (B, N, head_dim) per-sample cos values.
            rope_sin: (B, N, head_dim) per-sample sin values.
            v_start: start index of V-token(s).
            v_end: end index of V-token(s).
        """
        B, N, D = x.shape
        residual_x = x

        # Attention branch (Pre-Norm)
        x_normed = self.norm_attn(x)

        # OneTrans mixed parameter routing
        if self.mixed_params and (self.F_W_q is not None or self.V_W_q is not None):
            Q_parts, K_parts, V_parts = [], [], []
            # F-tokens: per-field independent projections
            if self.F_W_q is not None and num_F > 0:
                for i in range(min(num_F, len(self.F_W_q))):
                    q_i = self.F_W_q[i](x_normed[:, i:i+1, :])
                    k_i = self.F_W_k[i](x_normed[:, i:i+1, :])
                    v_i = self.F_W_v[i](x_normed[:, i:i+1, :])
                    Q_parts.append(q_i)
                    K_parts.append(k_i)
                    V_parts.append(v_i)
                # Remaining F-tokens use shared projection
                if num_F > len(self.F_W_q):
                    rem_q = self.W_q(x_normed[:, len(self.F_W_q):num_F, :])
                    rem_k = self.W_k(x_normed[:, len(self.F_W_q):num_F, :])
                    rem_v = self.W_v(x_normed[:, len(self.F_W_q):num_F, :])
                    Q_parts.append(rem_q)
                    K_parts.append(rem_k)
                    V_parts.append(rem_v)
            else:
                f_q = self.W_q(x_normed[:, :num_F, :])
                f_k = self.W_k(x_normed[:, :num_F, :])
                f_v = self.W_v(x_normed[:, :num_F, :])
                Q_parts.append(f_q)
                K_parts.append(f_k)
                V_parts.append(f_v)

            # Middle tokens (SEP + T + SEP): shared projection
            mid_start = num_F
            mid_end = v_start
            if mid_end > mid_start:
                mid_q = self.W_q(x_normed[:, mid_start:mid_end, :])
                mid_k = self.W_k(x_normed[:, mid_start:mid_end, :])
                mid_v = self.W_v(x_normed[:, mid_start:mid_end, :])
                Q_parts.append(mid_q)
                K_parts.append(mid_k)
                V_parts.append(mid_v)

            # V-tokens: independent projection
            if self.V_W_q is not None and v_end > v_start:
                v_q = self.V_W_q(x_normed[:, v_start:v_end, :])
                v_k = self.V_W_k(x_normed[:, v_start:v_end, :])
                v_v = self.V_W_v(x_normed[:, v_start:v_end, :])
                Q_parts.append(v_q)
                K_parts.append(v_k)
                V_parts.append(v_v)
            elif v_end > v_start:
                v_q = self.W_q(x_normed[:, v_start:v_end, :])
                v_k = self.W_k(x_normed[:, v_start:v_end, :])
                v_v = self.W_v(x_normed[:, v_start:v_end, :])
                Q_parts.append(v_q)
                K_parts.append(v_k)
                V_parts.append(v_v)

            Q = torch.cat(Q_parts, dim=1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = torch.cat(K_parts, dim=1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = torch.cat(V_parts, dim=1).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            Q = self.W_q(x_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.W_k(x_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.W_v(x_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            Q = apply_rope_per_sample(Q, rope_cos, rope_sin)
            K = apply_rope_per_sample(K, rope_cos, rope_sin)

        attn_mask = self._build_attn_mask(
            N=N, num_F=num_F, padding_mask=padding_mask,
            device=x.device, dtype=Q.dtype,
        )

        dropout_p = self.dropout if self.training else 0.0
        A = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        A = torch.nan_to_num(A, nan=0.0)
        A = A.transpose(1, 2).contiguous().view(B, N, D)
        A = self.W_o(A)

        # NLIR (Eq. 16-18)
        G = torch.sigmoid(self.W_g(x_normed))
        I_tilde = G * A
        I = residual_x + I_tilde

        # FFN branch (Pre-Norm)
        H = self.ffn(self.norm_ffn(I))
        return I + H


# ═══════════════════════════════════════════════════════════════════════════════
# TokenFormer Model
# ═══════════════════════════════════════════════════════════════════════════════


class TokenFormerModel(nn.Module):
    """Unified-token-stream model. Drop-in replacement for PCVRHyFormer.

    Internally composes a `PCVRHyFormer` to reuse all the well-tested
    feature-embedding code paths.
    """

    def __init__(
        self,
        # --- same data-schema args as PCVRHyFormer ---
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "Dict[str, List[int]]",
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # --- TokenFormer-specific hyperparameters ---
        d_model: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        num_full_attn_layers: int = 2,
        swa_windows: Optional[List[int]] = None,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        max_position: int = 4096,
        rope_base: float = 10000.0,
        mixed_params: bool = False,
        # --- reuse-from-PCVRHyFormer args ---
        emb_dim: int = 64,
        num_queries: int = 1,  # Ignored by TokenFormer (always uses unified stream)
        num_time_buckets: int = 65,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        seq_sideinfo_fids: Optional[Dict[str, List[int]]] = None,
        semantic_id_codes: Optional[Dict[str, Dict[str, Any]]] = None,
        user_dense_feature_dims: Optional[List[Tuple[int, int]]] = None,
        use_seq_gating: bool = False,
        unk_id_mask_rate: "Union[float, Dict[str, float]]" = 0.0,
        user_int_fids: Optional[List[int]] = None,
        item_int_fids: Optional[List[int]] = None,
        per_field: bool = False,
        # --- additional standard args that build_pcvr_model may pass ---
        seq_encoder_type: str = 'swiglu',
        seq_top_k: int = 50,
        seq_causal: bool = False,
        use_time_buckets: bool = True,
        rank_mixer_mode: str = 'none',
        use_rope: bool = True,
        gradient_checkpointing: bool = False,
        drop_path_rate: float = 0.0,
        flash_attention_backend: str = 'torch',
        rms_norm_backend: str = 'torch',
        rms_norm_block_rows: int = 1,
    ) -> None:
        super().__init__()

        if swa_windows is None:
            swa_windows = [32, 16]

        num_swa_layers = num_blocks - num_full_attn_layers
        if num_swa_layers < 0:
            raise ValueError(
                f"num_full_attn_layers ({num_full_attn_layers}) must be <= "
                f"num_blocks ({num_blocks})"
            )
        if num_swa_layers != len(swa_windows):
            raise ValueError(
                f"swa_windows ({swa_windows}) must have length "
                f"num_blocks - num_full_attn_layers = {num_swa_layers}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_blocks = num_blocks
        self.num_full_attn_layers = num_full_attn_layers
        self.swa_windows = swa_windows
        self.action_num = action_num
        self.max_position = max_position
        self.mixed_params = mixed_params

        # Compose PCVRHyFormer for embedding paths
        self._inner = PCVRHyFormer(
            user_int_feature_specs=user_int_feature_specs,
            item_int_feature_specs=item_int_feature_specs,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            seq_vocab_sizes=seq_vocab_sizes,
            user_ns_groups=user_ns_groups,
            item_ns_groups=item_ns_groups,
            d_model=d_model,
            emb_dim=emb_dim,
            num_queries=1,
            num_blocks=1,  # TokenFormer handles blocks internally
            num_heads=num_heads,
            seq_encoder_type='swiglu',
            hidden_mult=hidden_mult,
            dropout_rate=dropout_rate,
            seq_top_k=seq_top_k,
            seq_causal=seq_causal,
            action_num=action_num,
            num_time_buckets=num_time_buckets,
            rank_mixer_mode='none',
            use_rope=False,
            emb_skip_threshold=emb_skip_threshold,
            seq_id_threshold=seq_id_threshold,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path_rate,
            ns_tokenizer_type=ns_tokenizer_type,
            user_ns_tokens=user_ns_tokens,
            item_ns_tokens=item_ns_tokens,
        )

        self.seq_domains: List[str] = list(self._inner.seq_domains)
        self.num_sequences: int = self._inner.num_sequences
        self.num_time_buckets: int = num_time_buckets

        # Per-field tokenizer with ResSwiGLU
        self.per_field = per_field
        if per_field:
            self.user_per_field = PerFieldResSwiGLUTokenizer(
                self._inner.user_ns_tokenizer, d_model, hidden_mult=hidden_mult)
            self.item_per_field = PerFieldResSwiGLUTokenizer(
                self._inner.item_ns_tokenizer, d_model, hidden_mult=hidden_mult)
            self.user_dense_res_norm = RMSNorm(d_model)
            self.user_dense_res_ffn = SwiGLUFFN(d_model, hidden_mult=hidden_mult)
            _emb_dim = self._inner.user_ns_tokenizer.emb_dim
            self.cp_dense_ln = nn.LayerNorm(_emb_dim)
            self.cp_int_ln = nn.LayerNorm(_emb_dim)
            self.f_segment_emb = nn.Embedding(3, d_model)
            nn.init.normal_(self.f_segment_emb.weight, mean=0.0, std=0.02)

            num_user_int_fids = len(self.user_per_field.feature_specs)
            num_item_int_fids = len(self.item_per_field.feature_specs)
            num_user_dense_toks = 1  # Simplified: baseline PCVRHyFormer doesn't have ue_split

            self.v_ui_proj = nn.Sequential(
                nn.Linear(num_user_int_fids * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.v_ud_proj = nn.Sequential(
                nn.Linear(num_user_dense_toks * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.v_ii_proj = nn.Sequential(
                nn.Linear(num_item_int_fids * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.v_cat_weights = nn.Parameter(
                torch.tensor([0.0, 0.0, math.log(2.0)]))
            self.v_res_norm = RMSNorm(d_model)
            self.v_res_ffn = SwiGLUFFN(d_model, hidden_mult=hidden_mult)

            logging.info(
                f"[TokenFormer] per_field=True: user_int {num_user_int_fids} fids, "
                f"item_int {num_item_int_fids} fids"
            )
        else:
            self.user_per_field = None
            self.item_per_field = None
            self.user_dense_res_norm = None
            self.user_dense_res_ffn = None
            self.cp_dense_ln = None
            self.cp_int_ln = None
            self.f_segment_emb = None
            self.v_cat_weights = None
            self.v_res_norm = None
            self.v_res_ffn = None
            self.v_ui_proj = None
            self.v_ud_proj = None
            self.v_ii_proj = None

        # Segment embedding for T tokens
        self.segment_emb = nn.Embedding(self.num_sequences, d_model)
        nn.init.normal_(self.segment_emb.weight, mean=0.0, std=0.02)
        self._domain_to_segment = {d: i for i, d in enumerate(self.seq_domains)}

        # SEP tokens
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.sep_token, mean=0.0, std=0.02)

        # Position assignments
        self.sep_t_position = num_time_buckets
        self.sep_v_position = num_time_buckets + 1
        self.v_position = num_time_buckets + 2

        # RoPE cache
        rope_max = max(max_position, self.v_position + 1)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, max_seq_len=rope_max, base=rope_base,
        )

        # Pre-compute num_F and num_V for mixed_params
        if mixed_params:
            if per_field:
                num_user_int_fids_mp = len(user_int_feature_specs)
                num_item_int_fids_mp = len(item_int_feature_specs)
                num_user_dense_toks_mp = 1 if user_dense_dim > 0 else 0
                self._mp_num_F = num_user_int_fids_mp + num_user_dense_toks_mp + num_item_int_fids_mp
            else:
                self._mp_num_F = len(user_ns_groups) + len(item_ns_groups) + (1 if user_dense_dim > 0 else 0)
            self._mp_num_V = 1
        else:
            self._mp_num_F = 0
            self._mp_num_V = 0

        # Block stack with BFTS
        blocks = []
        for layer_idx in range(num_blocks):
            if layer_idx < num_full_attn_layers:
                ws = None
                disc = False
            else:
                ws = swa_windows[layer_idx - num_full_attn_layers]
                disc = True
            blocks.append(TokenFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                window_size=ws,
                discard_F=disc,
                mixed_params=mixed_params,
                num_F=self._mp_num_F,
                num_V=self._mp_num_V,
            ))
        self.blocks = nn.ModuleList(blocks)

        # Output head
        self.output_norm = RMSNorm(d_model)
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        # MLM-compat schema attributes (not available in baseline PCVRHyFormer)
        self._schema_user_int = None
        self._schema_item_int = None
        self._mlm_seq_sideinfo_fids = None
        self.unk_id_mask_rate = unk_id_mask_rate

        logging.info(
            f"[TokenFormer] depth={num_blocks} (L_f={num_full_attn_layers}, "
            f"swa={swa_windows}), d_model={d_model}, heads={num_heads}"
        )

    # Param-group helpers
    def get_sparse_params(self) -> List[nn.Parameter]:
        sparse_set = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_set.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_set]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 1000) -> "set":
        return self._inner.reinit_high_cardinality_params(cardinality_threshold)

    # Stream construction helpers
    def _build_user_dense_tokens_split(
        self, user_dense_feats: torch.Tensor, user_int_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Build user_dense token with ResSwiGLU residual."""
        inner = self._inner
        tok = inner._build_user_dense_tok(user_dense_feats, user_int_feats)
        return tok + self.user_dense_res_ffn(self.user_dense_res_norm(tok))

    def _build_F_tokens(
        self,
        inputs: ModelInput,
        apply_mask: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Builds F = [user_ns | user_dense | item_ns]."""
        inner = self._inner

        user_int = inputs.user_int_feats
        item_int = inputs.item_int_feats
        user_unk, item_unk = {}, {}

        if self.per_field:
            user_ns = self.user_per_field(user_int)
            item_ns = self.item_per_field(item_int)
        else:
            user_ns = inner.user_ns_tokenizer(user_int)
            item_ns = inner.item_ns_tokenizer(item_int)

        F_parts: List[torch.Tensor] = [user_ns]
        user_dense_tok = None
        if inner.has_user_dense:
            if self.per_field:
                user_dense_tok = self._build_user_dense_tokens_split(
                    inputs.user_dense_feats, user_int)
            else:
                user_dense_tok = inner._build_user_dense_tok(
                    inputs.user_dense_feats, user_int)
            F_parts.append(user_dense_tok)

        F_parts.append(item_ns)
        F_tokens = torch.cat(F_parts, dim=1)

        if self.per_field and self.f_segment_emb is not None:
            B, num_F, _D = F_tokens.shape
            num_user_int = user_ns.shape[1]
            num_user_dense = (user_dense_tok.shape[1] if user_dense_tok is not None else 0)
            num_item_int = item_ns.shape[1]
            seg_ids = torch.cat([
                F_tokens.new_zeros(B, num_user_int, dtype=torch.long),
                F_tokens.new_full((B, num_user_dense), 1, dtype=torch.long),
                F_tokens.new_full((B, num_item_int), 2, dtype=torch.long),
            ], dim=1)
            F_tokens = F_tokens + self.f_segment_emb(seg_ids)

        ns_unk_masks = {**user_unk, **item_unk}
        return F_tokens, ns_unk_masks, item_int

    def _build_V_tokens(
        self, inputs: ModelInput, item_int: torch.Tensor,
        F_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Builds target token V (1 token)."""
        inner = self._inner

        if (self.per_field and F_tokens is not None
                and self.v_cat_weights is not None):
            B = F_tokens.shape[0]
            D = F_tokens.shape[-1]
            num_user_int = len(self.user_per_field.feature_specs)
            num_item_int = len(self.item_per_field.feature_specs)
            num_user_dense_toks = max(0, F_tokens.shape[1] - num_user_int - num_item_int)

            # UI pool
            ui_flat = F_tokens[:, 0:num_user_int, :].reshape(B, num_user_int * D)
            ui_pool = self.v_ui_proj(ui_flat).unsqueeze(1)

            # UD pool (only if user_dense exists)
            if num_user_dense_toks > 0:
                ud_flat = F_tokens[:, num_user_int:num_user_int + num_user_dense_toks, :].reshape(B, num_user_dense_toks * D)
                ud_pool = self.v_ud_proj(ud_flat).unsqueeze(1)
            else:
                ud_pool = torch.zeros_like(ui_pool)

            # II pool
            ii_flat = F_tokens[:, num_user_int + num_user_dense_toks:, :].reshape(B, num_item_int * D)
            ii_pool = self.v_ii_proj(ii_flat).unsqueeze(1)

            w = torch.softmax(self.v_cat_weights, dim=0)
            fused = w[0] * ui_pool + w[1] * ud_pool + w[2] * ii_pool

            V = fused + self.v_res_ffn(self.v_res_norm(fused))
            return V

        if inner.has_item_dense:
            V = F.silu(inner.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            return V
        item_ns = inner.item_ns_tokenizer(item_int)
        return item_ns.mean(dim=1, keepdim=True)

    def _build_T_tokens(
        self, inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict[int, torch.Tensor]]]:
        """Embeds each sequence domain and concatenates into unified T stream."""
        inner = self._inner
        token_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        pos_chunks: List[torch.Tensor] = []
        seg_chunks: List[torch.Tensor] = []
        unk_masks_per_domain: Dict[str, Dict[int, torch.Tensor]] = {}

        for domain in self.seq_domains:
            tokens = inner._embed_seq_domain(
                inputs.seq_data[domain],
                inner._seq_embs[domain],
                inner._seq_proj[domain],
                inner._seq_is_id[domain],
                inner._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
            )
            unk_masks = {}

            B, L_i, D = tokens.shape
            device = tokens.device

            mask = ~inner._make_padding_mask(inputs.seq_lens[domain], L_i)
            pos = inputs.seq_time_buckets[domain].long().clamp(
                min=0, max=max(self.num_time_buckets - 1, 0)
            )
            seg_id = torch.full((B, L_i), self._domain_to_segment[domain],
                                dtype=torch.long, device=device)

            token_chunks.append(tokens)
            mask_chunks.append(mask)
            pos_chunks.append(pos)
            seg_chunks.append(seg_id)
            if unk_masks:
                unk_masks_per_domain[domain] = unk_masks

        T_tokens = torch.cat(token_chunks, dim=1)
        T_padding = torch.cat(mask_chunks, dim=1)
        T_positions = torch.cat(pos_chunks, dim=1)
        T_segments = torch.cat(seg_chunks, dim=1)

        # Time-aligned interleave: sort by time_bucket, padding first
        sort_key = T_positions.clone()
        sort_key = torch.where(T_padding, sort_key, torch.full_like(sort_key, -1))
        sort_idx = torch.argsort(sort_key, dim=1, stable=True)

        d_idx = sort_idx.unsqueeze(-1).expand(-1, -1, T_tokens.shape[-1])
        T_tokens = torch.gather(T_tokens, 1, d_idx)
        T_padding = torch.gather(T_padding, 1, sort_idx)
        T_positions = torch.gather(T_positions, 1, sort_idx)
        T_segments = torch.gather(T_segments, 1, sort_idx)

        T_tokens = T_tokens + self.segment_emb(T_segments)

        return T_tokens, T_padding, T_positions, unk_masks_per_domain

    def _assemble_stream(
        self,
        F_tokens: torch.Tensor,
        T_tokens: torch.Tensor,
        T_padding: torch.Tensor,
        T_positions: torch.Tensor,
        V_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """Builds X = [F | SEP | T | SEP | V]."""
        B = F_tokens.shape[0]
        device = F_tokens.device
        num_F = F_tokens.shape[1]
        num_T = T_tokens.shape[1]
        num_V = V_tokens.shape[1]

        sep_tok = self.sep_token.to(F_tokens.dtype).expand(B, 1, -1)

        X = torch.cat([F_tokens, sep_tok, T_tokens, sep_tok, V_tokens], dim=1)
        N = X.shape[1]

        F_mask = torch.ones((B, num_F), dtype=torch.bool, device=device)
        sep_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        V_mask = torch.ones((B, num_V), dtype=torch.bool, device=device)
        padding_mask = torch.cat(
            [F_mask, sep_mask, T_padding, sep_mask, V_mask], dim=1
        )

        F_pos = torch.zeros((B, num_F), dtype=torch.long, device=device)
        sep1_pos = torch.full((B, 1), self.sep_t_position, dtype=torch.long, device=device)
        sep2_pos = torch.full((B, 1), self.sep_v_position, dtype=torch.long, device=device)
        V_pos = torch.full((B, num_V), self.v_position, dtype=torch.long, device=device)
        positions = torch.cat(
            [F_pos, sep1_pos, T_positions, sep2_pos, V_pos], dim=1
        )

        v_start = num_F + 1 + num_T + 1
        v_end = v_start + num_V
        return X, padding_mask, positions, num_F, v_start, v_end

    def _gather_rope(
        self, positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gathers RoPE cos/sin at per-sample positions."""
        B, N = positions.shape
        cos_full = self.rotary_emb.cos_cached.to(positions.device)
        sin_full = self.rotary_emb.sin_cached.to(positions.device)
        HD = cos_full.shape[-1]
        cos_exp = cos_full.expand(B, -1, -1)
        sin_exp = sin_full.expand(B, -1, -1)
        idx = positions.unsqueeze(-1).expand(-1, -1, HD)
        cos = torch.gather(cos_exp, 1, idx)
        sin = torch.gather(sin_exp, 1, idx)
        return cos, sin

    def _run(
        self, inputs: ModelInput, apply_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Shared body for forward/predict."""
        F_tokens, ns_unk_masks, item_int_masked = self._build_F_tokens(inputs, apply_mask=apply_mask)
        T_tokens, T_padding, T_positions, unk_masks_per_domain = self._build_T_tokens(inputs)
        V_tokens = self._build_V_tokens(inputs, item_int_masked, F_tokens=F_tokens)

        X, padding_mask, positions, num_F, v_start, v_end = self._assemble_stream(
            F_tokens, T_tokens, T_padding, T_positions, V_tokens,
        )

        rope_cos, rope_sin = self._gather_rope(positions)

        for block in self.blocks:
            X = block(
                X,
                padding_mask=padding_mask,
                num_F=num_F,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                v_start=v_start,
                v_end=v_end,
            )

        V_slice = X[:, v_start:v_end, :]
        pooled = V_slice.mean(dim=1)
        pooled = self.output_norm(pooled)

        logits = self.clsfier(pooled)

        extras: Dict[str, Any] = {
            '_output': pooled,
            '_unk_masks_per_domain': unk_masks_per_domain,
            '_ns_unk_masks': ns_unk_masks,
            '_curr_seqs': None,
            '_curr_masks': None,
            '_raw_parts': None,
        }
        return logits, pooled, extras

    def forward(self, inputs: ModelInput):
        """Training: returns logits. Eval: returns logits only."""
        logits, _pooled, extras = self._run(inputs, apply_mask=self.training)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: returns (logits, pooled_embedding)."""
        logits, pooled, _ = self._run(inputs, apply_mask=False)
        return logits, pooled


# Export
__all__ = ["TokenFormerModel", "ModelInput"]
