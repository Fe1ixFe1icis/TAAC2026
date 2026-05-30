"""Re-run E1-E7 experiments with 5000 steps (resume from interruption)."""
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
    "--batch-size", "128",
    "--max-steps", "5000",
    "--eval-every-n-steps", "1000",
    "--seq_encoder_type", "longer",
    "--seq_causal",
    "--rank_mixer_mode", "none",
    "--ns_tokenizer_type", "rankmixer",
    "--user_ns_tokens", "1",
    "--item_ns_tokens", "2",
]

EXPERIMENTS = [
    # E1: Sequence Encoder Ablation
    {"name": "E1-Transformer", "run_dir": "outputs/e1_e7_5000/E1-Transformer", "extra_args": ["--seq-encoder-type", "transformer"]},
    {"name": "E1-SwiGLU", "run_dir": "outputs/e1_e7_5000/E1-SwiGLU", "extra_args": ["--seq-encoder-type", "swiglu"]},
    {"name": "E1-Longer", "run_dir": "outputs/e1_e7_5000/E1-Longer", "extra_args": ["--seq-encoder-type", "longer"]},
    {"name": "E1-Longer-Causal", "run_dir": "outputs/e1_e7_5000/E1-Longer-Causal", "extra_args": ["--seq-encoder-type", "longer", "--seq-causal"]},

    # E2: NS Tokenizer Comparison (skip completed ones)
    {"name": "E2-RankMixer-1-2", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-1-2", "extra_args": ["--user-ns-tokens", "1", "--item-ns-tokens", "2"]},
    {"name": "E2-RankMixer-5-2", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-5-2", "extra_args": ["--user-ns-tokens", "5", "--item-ns-tokens", "2"]},
    {"name": "E2-RankMixer-5-10", "run_dir": "outputs/e1_e7_5000/E2-RankMixer-5-10", "extra_args": ["--user-ns-tokens", "5", "--item-ns-tokens", "10"]},

    # E3: RankMixer Mode Ablation
    {"name": "E3-Full", "run_dir": "outputs/e1_e7_5000/E3-Full", "extra_args": ["--rank-mixer-mode", "full"]},
    {"name": "E3-FFNOnly", "run_dir": "outputs/e1_e7_5000/E3-FFNOnly", "extra_args": ["--rank-mixer-mode", "ffn_only"]},
    {"name": "E3-None", "run_dir": "outputs/e1_e7_5000/E3-None", "extra_args": ["--rank-mixer-mode", "none"]},

    # E4: RoPE Impact
    {"name": "E4-NoRoPE", "run_dir": "outputs/e1_e7_5000/E4-NoRoPE", "extra_args": []},
    {"name": "E4-RoPE-10k", "run_dir": "outputs/e1_e7_5000/E4-RoPE-10k", "extra_args": ["--use-rope", "--rope-base", "10000"]},

    # E5: Embedding Dimension Scaling
    {"name": "E5-d32", "run_dir": "outputs/e1_e7_5000/E5-d32", "extra_args": ["--d-model", "32", "--emb-dim", "32"]},
    {"name": "E5-d64", "run_dir": "outputs/e1_e7_5000/E5-d64", "extra_args": ["--d-model", "64", "--emb-dim", "64"]},
    {"name": "E5-d128", "run_dir": "outputs/e1_e7_5000/E5-d128", "extra_args": ["--d-model", "128", "--emb-dim", "128"]},
    {"name": "E5-d256", "run_dir": "outputs/e1_e7_5000/E5-d256", "extra_args": ["--d-model", "256", "--emb-dim", "256"]},

    # E6: Model Depth Scaling
    {"name": "E6-b1", "run_dir": "outputs/e1_e7_5000/E6-b1", "extra_args": ["--num-blocks", "1"]},
    {"name": "E6-b2", "run_dir": "outputs/e1_e7_5000/E6-b2", "extra_args": ["--num-blocks", "2"]},
    {"name": "E6-b4", "run_dir": "outputs/e1_e7_5000/E6-b4", "extra_args": ["--num-blocks", "4"]},
    {"name": "E6-b8", "run_dir": "outputs/e1_e7_5000/E6-b8", "extra_args": ["--num-blocks", "8"]},

    # E7: Query Count Ablation (only q=1 is valid)
    {"name": "E7-q1", "run_dir": "outputs/e1_e7_5000/E7-q1", "extra_args": ["--num-queries", "1"]},
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
    # Skip if already completed with 5000 steps
    existing_auc = extract_auc_from_checkpoint(exp["run_dir"])
    if existing_auc is not None:
        ckpt_dirs = list(Path(exp["run_dir"]).glob("global_step*"))
        ckpt_dirs.sort(key=lambda p: int(p.name.split(".")[0].replace("global_step", "")))
        last_step = int(ckpt_dirs[-1].name.split(".")[0].replace("global_step", ""))
        if last_step >= 5000:
            print(f"\n{'='*60}")
            print(f"Skipping {exp['name']} (already completed: step {last_step}, AUC={existing_auc})")
            print(f"{'='*60}")
            return {
                "name": exp["name"],
                "auc": existing_auc,
                "run_dir": exp["run_dir"],
                "status": "skipped",
            }

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
        "status": "completed",
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
