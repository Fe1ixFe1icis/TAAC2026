"""Phase 1 optimization smoke test."""
import subprocess
import sys
from pathlib import Path

# Use the working Python environment
python = r"D:\PYenvs\py3119\python.exe"

# Check python exists
if not Path(python).exists():
    print(f"ERROR: {python} not found")
    sys.exit(1)

# Set PYTHONPATH to include src directory
env = {
    "PYTHONPATH": r"D:\trae_proj\taac2026\win_version\src",
}

cmd = [
    python, "-m", "taac2026.application.training.workflow",
    "--experiment", "experiments/baseline",
    "--run-dir", "outputs/phase1_smoke",
    "--dataset-path", "data/avazu_pcvr",
    "--device", "cpu",
    "--num_workers", "0",
    "--batch_size", "8",
    "--max_steps", "100",
    "--eval_every_n_steps", "50",
    "--d_model", "64",
    "--emb_dim", "64",
    "--num_blocks", "2",
    "--num_queries", "1",
    "--seq_encoder_type", "swiglu",
    "--rank_mixer_mode", "none",
    "--ns_tokenizer_type", "rankmixer",
    "--user_ns_tokens", "1",
    "--item_ns_tokens", "2",
    "--ns_grouping_strategy", "singleton",
    "--dropout_rate", "0.01",
    "--dense_optimizer_type", "muon",
    "--scheduler_type", "cosine",
    "--warmup_steps", "10",
    "--min_lr_ratio", "0.1",
    "--ema_enabled",
    "--ema_decay", "0.999",
    "--ema_start_step", "50",
]

print("Running Phase 1 smoke test...")
print(f"Command: {' '.join(cmd)}")
print(f"PYTHONPATH: {env['PYTHONPATH']}")
print()

result = subprocess.run(
    cmd, 
    cwd=r"D:\trae_proj\taac2026\win_version", 
    env={**subprocess.os.environ, **env},
    capture_output=False, 
    text=True
)
print(f"\nExit code: {result.returncode}")
