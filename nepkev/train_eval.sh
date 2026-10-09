#!/usr/bin/env bash
# Runs on a fresh GPU pod. Inputs (uploaded by `nepkev remote-train`): /workspace/job/kev/{train,dev,calib,test}.jsonl
# and job.env. Outputs: /workspace/job/runs (evaluation rows, reports, the selected checkpoint).
#
# Steps, all with the pinned upstream Kev:
#   1. untouched Kev-0.8B on dev / calib / test and on Kev's own transfer-v4 development suite (retention reference)
#   2-3. fine-tune one epoch from the released checkpoint (--init_from), then EPOCHS-1 more, each from the previous
#        epoch's run; every epoch is evaluated on dev
#   4. select by dev (rule fixed here, before any result: lowest mean NLL over hard-labelled questions)
#   5. fit one temperature on calib for the selected run (in-distribution fit, recorded in head.pt)
#   6. selected + calibrated run on test (read once) and on transfer-v4 (retention)
set -euo pipefail
source /workspace/job/job.env
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/workspace/hf/triton-cache UV_LINK_MODE=copy
JOB=/workspace/job; RUNS=$JOB/runs; DATA=$JOB/kev
mkdir -p "$RUNS"
step() { echo "=== [$(date +%H:%M:%S)] $*"; }

step "setup: uv, pinned Kev, GPU kernels"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
apt-get -qq update >/dev/null && apt-get -qq install -y git >/dev/null || true
if [ ! -d /workspace/kev ]; then
  git clone -q https://github.com/jaredpalmer/kev /workspace/kev
fi
cd /workspace/kev
git checkout -q "$KEV_COMMIT"
uv sync -q --python 3.13
uv pip install -q "flash-linear-attention==0.5.2" "triton>=3.7.1"
uv pip install -q --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" \
  || echo "causal-conv1d wheel unavailable: PyTorch convolution fallback (slower, same results)"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

bench() {  # bench <run> <data-or-suite> <out>
  local run=$1 src=$2 out=$3
  rm -rf "$RUNS/$out"
  if [[ "$src" == evals/* ]]; then
    uv run python -m kev.benchmark --run "$run" --suite "$src" --out "$RUNS/$out" --device cuda
  else
    uv run python -m kev.benchmark --run "$run" --data "$DATA/$src.jsonl" --out "$RUNS/$out" --device cuda
  fi
}
dev_nll() {  # mean NLL of the gold option over hard-labelled questions (soft targets are not "knowable")
  uv run python - "$RUNS/$1/rows.json" <<'PY'
import json, math, sys
rows = [r for r in json.load(open(sys.argv[1])) if isinstance(r["label"], int)]
print(f"{sum(-math.log(max(r['p'][r['label']], 1e-12)) for r in rows) / len(rows):.6f}")
PY
}

BASE="$KEV_MODEL@$KEV_REVISION"
step "1. untouched $BASE"
for s in dev calib test; do bench "$BASE" "$s" "base-$s"; done
bench "$BASE" evals/v4/transfer-v4 base-transfer

TRAIN_ARGS=(--data "$DATA/train.jsonl" --base "$KEV_BASE" --base_revision "$KEV_BASE_REVISION"
            --epochs 1 --lr "$LR" --batch "$BATCH" --accum "$ACCUM" --dtype bf16 --checkpointing 1
            --p_none 0 --p_none_distract 0 --p_none_pair 0 --device cuda --seed "$SEED")
PREV="$BASE"
NLLS="\"base\": $(dev_nll base-dev)"
for E in $(seq 1 "$EPOCHS"); do
  step "2. fine-tune epoch $E/$EPOCHS from $PREV"
  uv run python -m kev.train "${TRAIN_ARGS[@]}" --init_from "$PREV" --out "$RUNS/ft-e$E"
  bench "$RUNS/ft-e$E" dev "ft-e$E-dev"
  NLLS="$NLLS, \"ft-e$E\": $(dev_nll "ft-e$E-dev")"
  PREV="$RUNS/ft-e$E"
done

step "4. select by dev NLL"
BEST=$(uv run python -c "d = {$NLLS}; d.pop('base'); print(min(d, key=d.get))")
printf '{"rule": "lowest mean dev NLL over hard-labelled questions", %s, "selected": "%s"}\n' "$NLLS" "$BEST" \
  | tee "$RUNS/selection.json"

step "5. temperature on calib for $BEST"
rm -rf "$RUNS/selected"; cp -r "$RUNS/$BEST" "$RUNS/selected"
bench "$RUNS/selected" calib selected-calib-inherited-T
# non-fatal: without a fitted temperature the selected run keeps the released checkpoint's temperature
uv run python scripts/calibrate_checkpoint.py --run "$RUNS/selected" --rows "$RUNS/selected-calib-inherited-T/rows.json" \
  --allow-in-distribution 2>&1 | tee "$RUNS/calibration.txt" \
  || echo "CALIBRATION FAILED: keeping the inherited temperature" | tee -a "$RUNS/calibration.txt"

step "6. selected + calibrated: calib (routing threshold), test (read once), dev, transfer-v4"
bench "$RUNS/selected" calib selected-calib
bench "$RUNS/selected" test selected-test
bench "$RUNS/selected" dev selected-dev
bench "$RUNS/selected" evals/v4/transfer-v4 selected-transfer
step "done"
