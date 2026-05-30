"""Phase 2 optimization experiments.

Tests:
- E11: DropPath (b=4, rate=0.1)
- E12: FM feature cross (rank_mixer_mode=fm)
- E13: SENET + FM combination
- E14: Comprehensive best config (Phase 1 + Phase 2)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Use the project virtual environment (venv_taac with CUDA)
PYTHON = r"D:\trae_proj\taac2026\venv_taac\Scripts\python.exe"

# Set PYTHONPATH to include src directory
env = os.environ.copy()
env["PYTHONPATH"] = r"D:\trae_proj\taac2026\win_version\src"

BASE_CMD = [
    PYTHON, "-m", "taac2026.application.training.cli",
    "--experiment", "experiments/avazu_baseline",
    "--dataset-path", "data/avazu-ctr-prediction/pcvr/avazu.parquet",
    "--schema-path", "data/avazu-ctr-prediction/pcvr/schema.json",
    "--device", "cuda",
    "--batch-size", "128",
    "--max-steps", "5000",
    "--eval-every-n-steps", "1000",
    "--seq_encoder_type", "swiglu",
    "--ns_tokenizer_type", "rankmixer",
    "--user_ns_tokens", "1",
    "--item_ns_tokens", "2",
    # Use AdamW baseline (Phase 1 optimizations hurt performance)
    "--dense_optimizer_type", "adamw",
    "--scheduler_type", "none",
    "--warmup_steps", "0",
    "--min_lr_ratio", "0.0",
]

EXPERIMENTS = [
    {
        "name": "E11-DropPath-b4",
        "run_dir": "outputs/phase2/E11-DropPath-b4",
        "extra_args": [
            "--num_blocks", "4",
            "--drop-path-rate", "0.1",
            "--rank_mixer_mode", "none",
        ],
        "description": "DropPath with b=4, rate=0.1",
    },
    {
        "name": "E12-FM-Cross",
        "run_dir": "outputs/phase2/E12-FM-Cross",
        "extra_args": [
            "--num_blocks", "2",
            "--rank_mixer_mode", "fm",
            "--drop-path-rate", "0.0",
        ],
        "description": "Explicit FM feature cross layer",
    },
    {
        "name": "E13-FM-DropPath",
        "run_dir": "outputs/phase2/E13-FM-DropPath",
        "extra_args": [
            "--num_blocks", "4",
            "--rank_mixer_mode", "fm",
            "--drop-path-rate", "0.1",
        ],
        "description": "FM + DropPath combined",
    },
    {
        "name": "E14-Comprehensive",
        "run_dir": "outputs/phase2/E14-Comprehensive",
        "extra_args": [
            "--num_blocks", "4",
            "--rank_mixer_mode", "fm",
            "--drop-path-rate", "0.1",
            "--seq-encoder-type", "swiglu",
        ],
        "description": "Comprehensive: FM + DropPath + SwiGLU + AdamW",
    },
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
                "description": exp["description"],
            }

    cmd = BASE_CMD + ["--run-dir", exp["run_dir"]] + exp["extra_args"]

    print(f"\n{'='*60}")
    print(f"Running {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"{'='*60}")

    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, env=env)

    auc = extract_auc_from_checkpoint(exp["run_dir"])
    print(f"{exp['name']} | AUC: {auc}")

    return {
        "name": exp["name"],
        "auc": auc,
        "run_dir": exp["run_dir"],
        "status": "completed",
        "description": exp["description"],
    }


def main():
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        RESULTS.append(result)

    # Save results
    output_path = Path("outputs/phase2/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PHASE 2 EXPERIMENT SUMMARY")
    print("="*60)

    for r in RESULTS:
        print(f"{r['name']}: AUC={r['auc']}")
        print(f"  {r['description']}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
