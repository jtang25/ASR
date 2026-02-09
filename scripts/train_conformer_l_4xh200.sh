#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_ROOT="${DATA_ROOT:-./data}"
SP_MODEL="${SP_MODEL:-./tokenizer/sp_1k.model}"
CKPT_DIR="${CKPT_DIR:-./checkpoints_conformer_l_4xh200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
EPOCHS="${EPOCHS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-300}"
EVAL_EVERY="${EVAL_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-5}"
RNNT_MAX_BATCH_TU="${RNNT_MAX_BATCH_TU:-2000000}"
AUTO_RESUME="${AUTO_RESUME:-1}"
PAPER_STRICT="${PAPER_STRICT:-1}"
RESUME_PATH="${RESUME_PATH:-}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-0}"
OVERRIDE_LR_ON_RESUME="${OVERRIDE_LR_ON_RESUME:-0}"
INIT_ENCODER_FROM="${INIT_ENCODER_FROM:-}"
STREAMING_MODE="${STREAMING_MODE:-0}"
STREAMING_CHUNK_SIZE="${STREAMING_CHUNK_SIZE:-8}"
STREAMING_LEFT_CONTEXT_CHUNKS="${STREAMING_LEFT_CONTEXT_CHUNKS:-16}"
STREAMING_RIGHT_CONTEXT="${STREAMING_RIGHT_CONTEXT:-0}"
STREAMING_CAUSAL_CONV="${STREAMING_CAUSAL_CONV:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if command -v torchrun >/dev/null 2>&1; then
  DIST_LAUNCHER=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}")
elif [[ -x /venv/main/bin/torchrun ]]; then
  DIST_LAUNCHER=(/venv/main/bin/torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}")
elif [[ -x /venv/main/bin/python3 ]]; then
  DIST_LAUNCHER=(/venv/main/bin/python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}")
elif command -v python3 >/dev/null 2>&1; then
  DIST_LAUNCHER=(python3 -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}")
else
  echo "Could not find torch launcher (torchrun/python3)." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${PAPER_STRICT}" == "1" ]]; then
  EXTRA_ARGS+=(--paper-strict)
fi
if [[ "${AUTO_RESUME}" == "0" ]]; then
  EXTRA_ARGS+=(--no-auto-resume)
fi
if [[ -n "${RESUME_PATH}" ]]; then
  EXTRA_ARGS+=(--resume-path "${RESUME_PATH}")
fi
if [[ "${RESET_SCHEDULER_ON_RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--reset-scheduler-on-resume)
fi
if [[ "${OVERRIDE_LR_ON_RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--override-lr-on-resume)
fi
if [[ -n "${INIT_ENCODER_FROM}" ]]; then
  EXTRA_ARGS+=(--init-encoder-from "${INIT_ENCODER_FROM}")
fi
if [[ "${STREAMING_MODE}" == "1" && "${PAPER_STRICT}" == "1" ]]; then
  echo "STREAMING_MODE=1 is incompatible with PAPER_STRICT=1. Set PAPER_STRICT=0." >&2
  exit 1
fi
if [[ "${STREAMING_MODE}" == "1" ]]; then
  EXTRA_ARGS+=(
    --streaming-mode
    --streaming-chunk-size "${STREAMING_CHUNK_SIZE}"
    --streaming-left-context-chunks "${STREAMING_LEFT_CONTEXT_CHUNKS}"
    --streaming-right-context "${STREAMING_RIGHT_CONTEXT}"
  )
  if [[ "${STREAMING_CAUSAL_CONV}" == "1" ]]; then
    EXTRA_ARGS+=(--streaming-causal-conv)
  fi
fi

"${DIST_LAUNCHER[@]}" train.py \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --train-splits train-clean-100 train-clean-360 train-other-500 \
  --val-split dev-other \
  --tokenizer sp \
  --sp-model "${SP_MODEL}" \
  --loss-type rnnt \
  --d-model 512 \
  --num-heads 8 \
  --num-layers 17 \
  --n-mels 80 \
  --conv-kernel 32 \
  --pred-hidden-dim 640 \
  --pred-num-layers 1 \
  --joint-dim 640 \
  --dropout 0.1 \
  --variational-noise 0.075 \
  --optimizer adam \
  --fused-optimizer \
  --precision bf16 \
  --tf32 \
  --lr-schedule paper \
  --paper-peak-factor 0.05 \
  --warmup-steps "${WARMUP_STEPS}" \
  --weight-decay 1e-6 \
  --batch-size "${BATCH_SIZE}" \
  --accum-steps "${ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --ddp-bucket-cap-mb 100 \
  --ddp-grad-as-bucket-view \
  --ddp-timeout-sec "${DDP_TIMEOUT_SEC}" \
  --rnnt-loss-device cuda \
  --no-rnnt-fused-log-softmax \
  --rnnt-max-batch-tu "${RNNT_MAX_BATCH_TU}" \
  --epochs "${EPOCHS}" \
  --ckpt-dir "${CKPT_DIR}" \
  --eval-every "${EVAL_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --no-eval-sample-decode \
  --no-download \
  "${EXTRA_ARGS[@]}" \
  "$@"
