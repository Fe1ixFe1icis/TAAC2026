"""Multi-dataset PCVR experiment: Amazon + Avazu + Criteo."""

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
        d_model=48,
        emb_dim=48,
        num_queries=2,
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
    ),
    ns=PCVRNSConfig(
        # NS token groups for unified multi-dataset schema.
        # Unified schema has:
        #   user_int: fid 1-31 (31 dims)
        #   item_int: fid 1-15 (15 dims)
        #   seq: domain_a with features [0, 38]
        #
        # Grouping strategy: group related features together
        # User groups: Amazon/Avazu shared (1-5), Criteo-specific (6-30), Avazu-specific (31)
        # Item groups: Amazon/Avazu shared (1), Avazu-specific (2-15), Criteo-specific (2-13)
        grouping_strategy="explicit",
        user_groups={
            "U1": [1],           # Shared user feature (Amazon/Avazu/Criteo)
            "U2": [2, 3, 4, 5],  # Avazu-specific user features
            "U3": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Criteo user features (part 1)
            "U4": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],  # Criteo user features (part 2)
            "U5": [26, 27, 28, 29, 30],  # Criteo user features (part 3)
            "U6": [31],          # Avazu user id (large vocab)
        },
        item_groups={
            "I1": [1],           # Shared item feature (Amazon/Avazu/Criteo)
            "I2": [2, 3, 4, 5, 6, 7, 8],   # Avazu/Criteo shared item features
            "I3": [9, 10, 11, 12, 13],     # Avazu/Criteo shared item features (cont.)
            "I4": [14, 15],      # Avazu-specific item features
        },
        tokenizer_type="rankmixer",
        user_tokens=6,
        item_tokens=4,
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="multi_dataset_pcvr",
    package_dir=Path(__file__).resolve().parent,
    model_class_name="PCVRHyFormer",
    train_defaults=TRAIN_DEFAULTS,
)
TRAIN_HOOKS = EXPERIMENT.train_hooks
PREDICTION_HOOKS = EXPERIMENT.prediction_hooks
RUNTIME_HOOKS = EXPERIMENT.runtime_hooks

__all__ = ["EXPERIMENT", "PREDICTION_HOOKS", "RUNTIME_HOOKS", "TRAIN_DEFAULTS", "TRAIN_HOOKS"]
