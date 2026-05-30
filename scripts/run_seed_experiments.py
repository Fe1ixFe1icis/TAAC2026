"""Batch run seed robustness experiments and collect results."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# All seeds: 5 fixed + 5 random
SEEDS = [42, 3507, 114514, 4615, 2026, 455711, 734613, 734087, 544634, 978088]

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

RESULTS = []


def extract_auc_from_checkpoint(run_dir: str) -> tuple[float | None, float | None]:
    """Extract final AUC from checkpoint directory name."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None, None

    # Find the last checkpoint directory
    ckpt_dirs = list(run_path.glob("global_step*"))
    if not ckpt_dirs:
        return None, None

    # Sort by step number
    ckpt_dirs.sort(key=lambda p: int(p.name.split(".")[0].replace("global_step", "")))
    last_ckpt = ckpt_dirs[-1]

    # Parse AUC from directory name: global_step5000.AUC=0.740601
    auc = None
    logloss = None
    name = last_ckpt.name
    if "AUC=" in name:
        auc_str = name.split("AUC=")[1]
        try:
            auc = float(auc_str)
        except ValueError:
            pass

    return auc, logloss


def run_experiment(seed: int) -> dict:
    run_dir = f"outputs/seed_exp/seed_{seed}"
    cmd = BASE_CMD + [
        "--run-dir", run_dir,
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Running seed={seed}")
    print(f"{'='*60}")

    # Run training
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)

    # Extract results from checkpoint
    auc, logloss = extract_auc_from_checkpoint(run_dir)

    print(f"Seed {seed} | AUC: {auc} | LogLoss: {logloss}")

    return {
        "seed": seed,
        "auc": auc,
        "logloss": logloss,
        "run_dir": run_dir,
    }


def main():
    for seed in SEEDS:
        result = run_experiment(seed)
        RESULTS.append(result)

    # Save results
    output_path = Path("outputs/seed_exp/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SEED ROBUSTNESS SUMMARY")
    print("="*60)

    aucs = [r["auc"] for r in RESULTS if r["auc"] is not None]
    if aucs:
        import statistics
        mean_auc = statistics.mean(aucs)
        std_auc = statistics.stdev(aucs) if len(aucs) > 1 else 0
        cv = std_auc / mean_auc * 100 if mean_auc > 0 else 0

        print(f"Seeds tested: {len(aucs)}")
        print(f"Mean AUC: {mean_auc:.6f} ± {std_auc:.6f}")
        print(f"Min AUC: {min(aucs):.6f}")
        print(f"Max AUC: {max(aucs):.6f}")
        print(f"CV: {cv:.3f}%")

        if cv < 0.5:
            print("\n结论: 模型对随机种子不敏感，训练稳定 (CV < 0.5%)")
        elif cv < 1.0:
            print("\n结论: 模型对随机种子中等敏感 (CV < 1.0%)")
        else:
            print("\n结论: 模型对随机种子敏感，可能需要更多正则化 (CV > 1.0%)")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
