"""Re-run E1-E7 experiments with 5000 steps for fair comparison."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE_CMD = [
    sys.executable, "-m", "taac2026.application.training.cli",
    "--experiment", "experiments/avazu_baseline",
    "--dataset-path", "data/avazu-ctr-prediction/pcvr/avazu.parquet",
    "--schema-path", "data/avazu-ctr-prediction/pcvr/schema.json",
    "--device", "cuda",
    "--batch_size", "256",
    "--max_steps", "5000",
    "--eval_every_n_steps", "1000",
    "--seq_encoder_type", "longer",
    "--seq_causal",
    "--rank_mixer_mode", "none",
    "--ns_tokenizer_type", "rankmixer",
    "--user_ns_tokens", "1",
    "--item_ns_tokens", "2",
]

EXPERIMENTS = [
    # E1: Sequence Encoder Ablation
    {"name": "E1-Transformer", "run_dir": "outputs/e1_e7_5000/E1-Transformer", "extra_args": ["--seq_encoder_type", "transformer", "--seq_causal", "False"]},
    {"name": "E1-SwiGLU", "run_dir": "outputs/e1_e7_5000/E1-SwiGLU", "extra_args": ["--seq_encoder_type", "swiglu", "--seq_causal", "False"]},
    {"name": "E1-Longer", "run_dir": "outputs/e1_e7_5000/E1-Longer", "extra_args": ["--seq_encoder_type", "longer", "--seq_causal", "False"]},
    {"name": "E1-Longer-Causal", "run_dir": "outputs/e1_e7_5000/E1-Longer-Causal", "extra_args": ["--seq_encoder_type", "longer", "--seq_causal", "True"]},

    # E2: NS Tokenizer Comparison
    {"name": "E2-RankMixer-1-2", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-1-2", "extra_args": ["--user_ns_tokens", "1", "--item_ns_tokens", "2"]},
    {"name": "E2-RankMixer-5-2", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-5-2", "extra_args": ["--user_ns_tokens", "5", "--item_ns_tokens", "2"]},
    {"name": "E2-RankMixer-5-10", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-5-10", "extra_args": ["--user_ns_tokens", "5", "--item_ns_tokens", "10"]},

    # E3: RankMixer Mode Ablation
    {"name": "E3-Full", "run_dir": "outputs/e1_e7_5000/E3-Full", "extra_args": ["--rank_mixer_mode", "full"]},
    {"name": "E3-FFNOnly", "run_dir": "outputs/e1_e7_5000/E3-FFNOnly", "extra_args": ["--rank_mixer_mode", "ffn_only"]},
    {"name": "E3-None", "run_dir": "outputs/e1_e7_5000/E3-None", "extra_args": ["--rank_mixer_mode", "none"]},

    # E4: RoPE Impact
    {"name": "E4-NoRoPE", "run_dir": "outputs/e1_e7_5000/E4-NoRoPE", "extra_args": ["--use_rope", "False"]},
    {"name": "E4-RoPE-10k", "run_dir": "outputs/e1_e7_5000/E4-RoPE-10k", "extra_args": ["--use_rope", "True", "--rope_base", "10000"]},

    # E5: Embedding Dimension Scaling
    {"name": "E5-d32", "run_dir": "outputs/e1_e7_5000/E5-d32", "extra_args": ["--d_model", "32", "--emb_dim", "32"]},
    {"name": "E5-d64", "run_dir": "outputs/e1_e7_5000/E5-d64", "extra_args": ["--d_model", "64", "--emb_dim", "64"]},
    {"name": "E5-d128", "run_dir": "outputs/e1_e7_5000/E5-d128", "extra_args": ["--d_model", "128", "--emb_dim", "128"]},
    {"name": "E5-d256", "run_dir": "outputs/e1_e7_5000/E5-d256", "extra_args": ["--d_model", "256", "--emb_dim", "256"]},

    # E6: Model Depth Scaling
    {"name": "E6-b1", "run_dir": "outputs/e1_e7_5000/E6-b1", "extra_args": ["--num_blocks", "1"]},
    {"name": "E6-b2", "run_dir": "outputs/e1_e7_5000/E6-b2", "extra_args": ["--num_blocks", "2"]},
    {"name": "E6-b4", "run_dir": "outputs/e1_e7_5000/E6-b4", "extra_args": ["--num_blocks", "4"]},
    {"name": "E6-b8", "run_dir": "outputs/e1_e7_5000/E6-b8", "extra_args": ["--num_blocks", "8"]},

    # E7: Query Count Ablation (only q=1 is valid)
    {"name": "E7-q1", "run_dir": "outputs/e1_e7_5000/E7-q1", "extra_args": ["--num_queries", "1"]},
]

RESULTS = []


def extract_auc_from_checkpoint(run_dir: str) -> float | None:
    """Extract final AUC from checkpoint directory name."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    ckpt_dirs = list(run_path.glob("global_step*"))
    if not ckpt_dirs:
        return None

    ckpt_dirs.sort(key=lambda p: int(p.name.split(".")[0].replace("global_step", "")))
    last_ckpt = ckpt_dirs[-1]

    name = last_ckpt.name
    if "AUC=" in name:
        try:
            return float(name.split("AUC=")[1])
        except ValueError:
            pass

    return None


def run_experiment(exp: dict) -> dict:
    cmd = BASE_CMD + ["--run-dir", exp["run_dir"]] + exp["extra_args"]

    print(f"\n{'='*60}")
    print(f"Running {exp['name']}")
    print(f"{'='*60}")

    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)

    auc = extract_auc_from_checkpoint(exp["run_dir"])
    print(f"{exp['name']} | AUC: {auc}")

    return {
        "name": exp["name"],
        "auc": auc,
        "run_dir": exp["run_dir"],
    }


def main():
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        RESULTS.append(result)

    # Save results
    output_path = Path("outputs/e1_e7_5000/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("E1-E7 5000 STEPS SUMMARY")
    print("="*60)

    for r in RESULTS:
        print(f"{r['name']}: AUC={r['auc']}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
