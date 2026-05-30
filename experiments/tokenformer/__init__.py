"""TokenFormer experiment: unified token stream for multi-dataset training."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from taac2026.api import (
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRFeatureMaskConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
    PCVRTrainConfig,
    PCVREMAConfig,
)
from taac2026.api import create_pcvr_experiment
from taac2026.api import RuntimeExecutionConfig
from taac2026.application.training.workflow import (
    PCVRTrainContext,
    PCVRTrainDataBundle,
    PCVRTrainHooks,
    default_build_train_model,
)
from taac2026.infrastructure.modeling.model_contract import build_pcvr_model, load_ns_groups
from taac2026.infrastructure.logging import logger


# TokenFormer-specific hyperparameters (not in standard PCVRModelConfig)
TOKENFORMER_EXTRA_KWARGS = {
    "num_full_attn_layers": 3,   # P2: Increased from 2 to 3 (for num_blocks=6)
    'swa_windows': [64, 32, 16], # L_s: top 3 layers SWA (shrinking windows)
    "per_field": True,           # Per-field tokenization with ResSwiGLU + Gating
    "max_position": 4096,        # RoPE cache size
    "mixed_params": False,       # OneTrans: 关闭 (E4 性价比不高)
    "small_init": True,          # P0: Down-matrix small initialization
}


TRAIN_DEFAULTS = PCVRTrainConfig(
    data=PCVRDataConfig(
        batch_size=512,
        num_workers=8,
        buffer_batches=1,
        train_ratio=1.0,
        valid_ratio=0.1,
        sampling_strategy="step_random",
        eval_every_n_steps=5_000,
        seq_max_lens="seq_a:256",
    ),
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="none", max_batches=0),
        transforms=(
            PCVRFeatureMaskConfig(enabled=True, probability=0.1),
            PCVRNonSequentialSparseDropoutConfig(enabled=True, probability=0.05),
        ),
        seed=None,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=100_000,
        patience_steps=25_000,
        seed=42,
        device=None,
        dense_optimizer_type="adamw",
        scheduler_type="none",
        warmup_steps=0,
        min_lr_ratio=0.0,
    ),
    ema=PCVREMAConfig(
        enabled=False,
        decay=0.999,
        start_step=1000,
        update_every_n_steps=1,
    ),
    runtime=RuntimeExecutionConfig(
        amp=True,
        amp_dtype="bfloat16",
        compile=False,
        progress_log_interval_steps=100,
    ),
    loss=PCVRLossConfig(terms=(PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.05,
        sparse_weight_decay=0.0,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=PCVRModelConfig(
        d_model=64,
        emb_dim=64,
        num_queries=1,
        num_blocks=6,               # P2: Increased from 4 to 6
        num_heads=4,
        seq_encoder_type="swiglu",
        hidden_mult=4,
        dropout_rate=0.01,
        seq_top_k=50,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="none",
        use_rope=True,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
    ),
    ns=PCVRNSConfig(
        grouping_strategy="explicit",
        user_groups={
            "U1": [1],
            "U2": [2, 3, 4, 5],
            "U3": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "U4": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            "U5": [26, 27, 28, 29, 30],
            "U6": [31],
        },
        item_groups={
            "I1": [1],
            "I2": [2, 3, 4, 5, 6, 7, 8],
            "I3": [9, 10, 11, 12, 13],
            "I4": [14, 15],
        },
        tokenizer_type="rankmixer",
        user_tokens=6,
        item_tokens=4,
    ),
)


def _tokenformer_build_train_model(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
) -> torch.nn.Module:
    """Custom build_model hook that injects TokenFormer-specific kwargs."""
    user_ns_groups, item_ns_groups = load_ns_groups(
        data_bundle.dataset,
        context.config,
        context.package_dir,
        context.ckpt_dir,
    )
    logger.info("User NS groups: {}", user_ns_groups)
    logger.info("Item NS groups: {}", item_ns_groups)

    from taac2026.infrastructure.modeling.sequence import configure_flash_attention_runtime
    configure_flash_attention_runtime(
        backend=str(context.config.get("flash_attention_backend", "torch")),
    )

    # Merge standard config with TokenFormer extra kwargs
    standard_keys = {f.name for f in fields(PCVRModelConfig)}
    extra_kwargs = {
        k: v for k, v in context.config.items()
        if k not in standard_keys and not k.startswith(("data_", "optimizer_", "runtime_", "sparse_", "ns_", "ema_", "validation_", "loss_"))
    }
    # Always inject TokenFormer defaults (can be overridden by CLI)
    for key, default_value in TOKENFORMER_EXTRA_KWARGS.items():
        if key not in extra_kwargs:
            extra_kwargs[key] = default_value

    logger.info("TokenFormer extra kwargs: {}", extra_kwargs)

    model = build_pcvr_model(
        model_module=context.model_module,
        model_class_name=context.model_class_name,
        data_module=data_bundle.data_module,
        dataset=data_bundle.dataset,
        config=context.config,
        package_dir=context.package_dir,
        checkpoint_dir=context.ckpt_dir,
        extra_model_kwargs=extra_kwargs if extra_kwargs else None,
    ).to(context.args.device)

    num_sequences = len(data_bundle.dataset.seq_domains)
    num_ns = model.num_ns if hasattr(model, 'num_ns') else 0
    token_count = context.args.num_queries * num_sequences + num_ns
    logger.info(
        "TokenFormer model created: class={}, num_ns={}, T={}, d_model={}",
        context.model_class_name,
        num_ns,
        token_count,
        context.args.d_model,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Total parameters: {}", f"{total_params:,}")
    return model


EXPERIMENT = create_pcvr_experiment(
    name="tokenformer_multi_dataset",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="TokenFormerModel",
    train_defaults=TRAIN_DEFAULTS,
    train_hook_overrides={"build_model": _tokenformer_build_train_model},
)
TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks

__all__ = ["EXPERIMENT", "PREDICTION_HOOKS", "RUNTIME_HOOKS", "TRAIN_DEFAULTS", "TRAIN_HOOKS"]
