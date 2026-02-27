#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_on_cluster.sh /absolute/path/to/FakeImageDetector
# If no arg is provided, current directory is used.

ROOT_DIR="${1:-$(pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"

echo "Project root: $ROOT_DIR"
cd "$ROOT_DIR"

if [[ ! -f "ml/train_transfer.py" ]]; then
  echo "Error: ml/train_transfer.py not found in $ROOT_DIR"
  exit 1
fi

echo "Python: $(python3 --version)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<not set>}"

python3 - <<'PY'
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}] =", torch.cuda.get_device_name(i))
PY

mkdir -p ml/artifacts

echo "Preparing merged dataset CSVs..."
python3 ml/prepare_combined_dataset.py

mkdir -p ml/artifacts/gpu_logs

common_args=(
  --data-root "$ROOT_DIR"
  --train-csv "$ROOT_DIR/datasets/combined_faces/train.csv"
  --valid-csv "$ROOT_DIR/datasets/combined_faces/valid.csv"
  --train-per-class 0
  --valid-per-class 0
  --num-workers 8
)

echo "Starting 3 training runs in parallel..."

CUDA_VISIBLE_DEVICES=0 python3 ml/train_transfer.py \
  "${common_args[@]}" \
  --out-dir "$ROOT_DIR/ml/artifacts/gpu_run_a" \
  --epochs 20 --batch-size 128 --lr 2e-4 --weight-decay 1e-4 --freeze-backbone \
  > "$ROOT_DIR/ml/artifacts/gpu_logs/run_a.log" 2>&1 &
pid_a=$!

CUDA_VISIBLE_DEVICES=1 python3 ml/train_transfer.py \
  "${common_args[@]}" \
  --out-dir "$ROOT_DIR/ml/artifacts/gpu_run_b" \
  --epochs 20 --batch-size 96 --lr 1e-4 --weight-decay 1e-4 --unfreeze-backbone \
  > "$ROOT_DIR/ml/artifacts/gpu_logs/run_b.log" 2>&1 &
pid_b=$!

CUDA_VISIBLE_DEVICES=2 python3 ml/train_transfer.py \
  "${common_args[@]}" \
  --out-dir "$ROOT_DIR/ml/artifacts/gpu_run_c" \
  --epochs 24 --batch-size 128 --lr 3e-4 --weight-decay 3e-4 --freeze-backbone \
  > "$ROOT_DIR/ml/artifacts/gpu_logs/run_c.log" 2>&1 &
pid_c=$!

echo "PIDs: run_a=$pid_a run_b=$pid_b run_c=$pid_c"
echo "Tail logs with:"
echo "  tail -f ml/artifacts/gpu_logs/run_a.log"
echo "  tail -f ml/artifacts/gpu_logs/run_b.log"
echo "  tail -f ml/artifacts/gpu_logs/run_c.log"

wait "$pid_a" "$pid_b" "$pid_c"
echo "All runs finished."

python3 - <<'PY'
import json
from pathlib import Path

root = Path("ml/artifacts")
runs = [root / "gpu_run_a", root / "gpu_run_b", root / "gpu_run_c"]
best = None
rows = []

for run in runs:
    summary_path = run / "summary.json"
    if not summary_path.exists():
        rows.append((run.name, None, None))
        continue
    s = json.loads(summary_path.read_text())
    acc = float(s.get("best_valid_acc", 0.0))
    epoch = s.get("best_epoch")
    rows.append((run.name, acc, epoch))
    if best is None or acc > best[1]:
        best = (run, acc, epoch)

print("\nRun summary:")
for name, acc, epoch in rows:
    print(f"  {name}: best_valid_acc={acc} best_epoch={epoch}")

if best is None:
    raise SystemExit("No valid summary.json found; training may have failed.")

best_run, best_acc, best_epoch = best
best_ckpt = best_run / "best_resnet18.pt"
target = root / "best_resnet18.pt"

if not best_ckpt.exists():
    raise SystemExit(f"Best checkpoint missing: {best_ckpt}")

target.write_bytes(best_ckpt.read_bytes())
print(f"\nSelected best run: {best_run.name}")
print(f"best_valid_acc={best_acc}, best_epoch={best_epoch}")
print(f"Copied checkpoint to: {target}")
PY

echo
echo "Done. Final model: ml/artifacts/best_resnet18.pt"
echo "Please send me:"
echo "  1) ml/artifacts/best_resnet18.pt"
echo "  2) ml/artifacts/gpu_run_*/summary.json"
