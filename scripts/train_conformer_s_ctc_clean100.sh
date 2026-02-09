#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_ROOT="${DATA_ROOT:-./data}"
CKPT_DIR="${CKPT_DIR:-./checkpoints_conformer_s_ctc_clean100}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EVAL_EVERY="${EVAL_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    DIST_LAUNCHER=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}")
  elif [[ -x /venv/main/bin/torchrun ]]; then
    DIST_LAUNCHER=(/venv/main/bin/torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}")
  elif [[ -x /venv/main/bin/python3 ]]; then
    DIST_LAUNCHER=(/venv/main/bin/python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}")
  else
    DIST_LAUNCHER=(python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}")
  fi
else
  if [[ -x /venv/main/bin/python3 ]]; then
    DIST_LAUNCHER=(/venv/main/bin/python3)
  else
    DIST_LAUNCHER=(python3)
  fi
fi

"${DIST_LAUNCHER[@]}" train.py \
  --device auto \
  --data-root "${DATA_ROOT}" \
  --train-splits train-clean-100 \
  --val-split dev-clean \
  --tokenizer char \
  --loss-type ctc \
  --d-model 256 \
  --num-heads 4 \
  --num-layers 8 \
  --conv-kernel 32 \
  --n-mels 80 \
  --dropout 0.1 \
  --variational-noise 0.0 \
  --optimizer adamw \
  --no-fused-optimizer \
  --precision bf16 \
  --tf32 \
  --lr-schedule cosine \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --min-lr 1e-5 \
  --weight-decay 1e-6 \
  --batch-size "${BATCH_SIZE}" \
  --accum-steps "${ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --beam-size 8 \
  --eval-every "${EVAL_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --epochs "${EPOCHS}" \
  --ckpt-dir "${CKPT_DIR}" \
  --no-auto-resume \
  --no-eval-sample-decode \
  --no-download \
  "$@"
