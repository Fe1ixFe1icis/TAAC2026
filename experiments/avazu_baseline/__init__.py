"""Avazu CTR baseline experiment package (singleton NS groups)."""

from __future__ import annotations

from pathlib import Path

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

TRAIN_DEFAULTS = PCVRTrainConfig(
    data=PCVRDataConfig(
        batch_size=256,
        num_workers=8,
        buffer_batches=1,
        train_ratio=1.0,
        valid_ratio=0.1,
        sampling_strategy="step_random",
        eval_every_n_steps=5_000,
        seq_max_lens="a:50",
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
        dense_optimizer_type="muon",
        scheduler_type="cosine",
        warmup_steps=200,
        min_lr_ratio=0.1,
    ),
    ema=PCVREMAConfig(
        enabled=True,
        decay=0.999,
        start_step=1000,
        update_every_n_steps=1,
    ),
    runtime=RuntimeExecutionConfig(
        amp=False,
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
        num_blocks=2,
        num_heads=4,
        seq_encoder_type="transformer",
        hidden_mult=4,
        dropout_rate=0.01,
        seq_top_k=50,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
        drop_path_rate=0.0,
    ),
    ns=PCVRNSConfig(
        grouping_strategy="singleton",
        user_groups={},
        item_groups={},
        tokenizer_type="rankmixer",
        user_tokens=5,
        item_tokens=2,
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="avazu_baseline",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRHyFormer",
    train_defaults=TRAIN_DEFAULTS,
)
TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks

__all__ = ["EXPERIMENT", "PREDICTION_HOOKS", "RUNTIME_HOOKS", "TRAIN_DEFAULTS", "TRAIN_HOOKS"]
