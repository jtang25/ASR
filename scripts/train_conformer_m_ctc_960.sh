#!/usr/bin/env bash
set -euo pipefail

# Two-stage pipeline:
#   1) Train Conformer-M with CTC on LibriSpeech 960h.
#      Stop at 80 epochs or after 3 evals without val_wer improvement.
#   2) Copy stage-1 best/last checkpoints to a separate export directory.
#   3) Train Conformer-M RNN-T warm-starting encoder from stage-1 best.pt.
#      Stop at 100 epochs or after 3 evals without val_wer improvement.

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DATA_ROOT="${DATA_ROOT:-./data}"
SP_MODEL="${SP_MODEL:-./tokenizer/sp_1k.model}"
RUN_ROOT="${RUN_ROOT:-./checkpoints_conformer_m_ctc_then_rnnt}"

CTC_CKPT_DIR="${CTC_CKPT_DIR:-${RUN_ROOT}/ctc_stage}"
CTC_EXPORT_DIR="${CTC_EXPORT_DIR:-${RUN_ROOT}/ctc_export}"
RNNT_CKPT_DIR="${RNNT_CKPT_DIR:-${RUN_ROOT}/rnnt_stage}"

NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-300}"
AUTO_RESUME="${AUTO_RESUME:-1}"

# Stage-1 (CTC) settings
CTC_EPOCHS="${CTC_EPOCHS:-80}"
CTC_EARLY_STOP_PATIENCE="${CTC_EARLY_STOP_PATIENCE:-3}"
CTC_BATCH_SIZE="${CTC_BATCH_SIZE:-96}"
CTC_ACCUM_STEPS="${CTC_ACCUM_STEPS:-2}"
CTC_WARMUP_STEPS="${CTC_WARMUP_STEPS:-3000}"
CTC_PAPER_PEAK_FACTOR="${CTC_PAPER_PEAK_FACTOR:-0.035}"
CTC_EVAL_EVERY="${CTC_EVAL_EVERY:-1}"
CTC_SAVE_EVERY="${CTC_SAVE_EVERY:-1}"

# Stage-2 (RNN-T) settings
RNNT_EPOCHS="${RNNT_EPOCHS:-100}"
RNNT_EARLY_STOP_PATIENCE="${RNNT_EARLY_STOP_PATIENCE:-3}"
RNNT_BATCH_SIZE="${RNNT_BATCH_SIZE:-48}"
RNNT_ACCUM_STEPS="${RNNT_ACCUM_STEPS:-2}"
RNNT_WARMUP_STEPS="${RNNT_WARMUP_STEPS:-6000}"
RNNT_PAPER_PEAK_FACTOR="${RNNT_PAPER_PEAK_FACTOR:-0.018}"
RNNT_EVAL_EVERY="${RNNT_EVAL_EVERY:-1}"
RNNT_SAVE_EVERY="${RNNT_SAVE_EVERY:-1}"
RNNT_MAX_BATCH_TU="${RNNT_MAX_BATCH_TU:-1500000}"
RNNT_EVAL_BEAM_SIZE="${RNNT_EVAL_BEAM_SIZE:-8}"
RNNT_EVAL_BEAM_TOPK="${RNNT_EVAL_BEAM_TOPK:-10}"
RNNT_EVAL_MAX_SYMBOLS_PER_STEP="${RNNT_EVAL_MAX_SYMBOLS_PER_STEP:-10}"

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

STAGE_ARGS=("$@")
CTC_EXTRA_ARGS=()
RNNT_EXTRA_ARGS=()
if [[ "${AUTO_RESUME}" == "0" ]]; then
  CTC_EXTRA_ARGS+=(--no-auto-resume)
  RNNT_EXTRA_ARGS+=(--no-auto-resume)
fi

mkdir -p "${CTC_CKPT_DIR}" "${CTC_EXPORT_DIR}" "${RNNT_CKPT_DIR}"

echo "============================================================"
echo "Stage 1/2: CTC pretraining (Conformer-M)"
echo "============================================================"
"${DIST_LAUNCHER[@]}" train.py \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --train-splits train-clean-100 train-clean-360 train-other-500 \
  --val-split dev-other \
  --tokenizer sp \
  --sp-model "${SP_MODEL}" \
  --loss-type ctc \
  --d-model 256 \
  --num-heads 4 \
  --num-layers 16 \
  --n-mels 80 \
  --conv-kernel 32 \
  --dropout 0.1 \
  --variational-noise 0.0 \
  --optimizer adamw \
  --fused-optimizer \
  --precision bf16 \
  --tf32 \
  --lr-schedule paper \
  --paper-peak-factor "${CTC_PAPER_PEAK_FACTOR}" \
  --warmup-steps "${CTC_WARMUP_STEPS}" \
  --weight-decay 1e-6 \
  --batch-size "${CTC_BATCH_SIZE}" \
  --accum-steps "${CTC_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --ddp-bucket-cap-mb 100 \
  --ddp-grad-as-bucket-view \
  --ddp-timeout-sec "${DDP_TIMEOUT_SEC}" \
  --beam-size 8 \
  --beam-token-prune 32 \
  --epochs "${CTC_EPOCHS}" \
  --ckpt-dir "${CTC_CKPT_DIR}" \
  --eval-every "${CTC_EVAL_EVERY}" \
  --save-every "${CTC_SAVE_EVERY}" \
  --early-stop-patience "${CTC_EARLY_STOP_PATIENCE}" \
  --no-eval-sample-decode \
  --no-download \
  "${CTC_EXTRA_ARGS[@]}" \
  "${STAGE_ARGS[@]}"

if [[ ! -f "${CTC_CKPT_DIR}/best.pt" ]]; then
  echo "Stage 1 best checkpoint missing: ${CTC_CKPT_DIR}/best.pt" >&2
  exit 1
fi
if [[ ! -f "${CTC_CKPT_DIR}/last.pt" ]]; then
  echo "Stage 1 last checkpoint missing: ${CTC_CKPT_DIR}/last.pt" >&2
  exit 1
fi

cp -f "${CTC_CKPT_DIR}/best.pt" "${CTC_EXPORT_DIR}/best.pt"
cp -f "${CTC_CKPT_DIR}/last.pt" "${CTC_EXPORT_DIR}/last.pt"

CTC_BEST_FOR_RNNT="${CTC_EXPORT_DIR}/best.pt"

echo "Stage 1 exported:"
echo "  ${CTC_EXPORT_DIR}/best.pt"
echo "  ${CTC_EXPORT_DIR}/last.pt"

echo "============================================================"
echo "Stage 2/2: RNN-T finetuning from CTC encoder warm-start"
echo "============================================================"
"${DIST_LAUNCHER[@]}" train.py \
  --device cuda \
  --data-root "${DATA_ROOT}" \
  --train-splits train-clean-100 train-clean-360 train-other-500 \
  --val-split dev-other \
  --tokenizer sp \
  --sp-model "${SP_MODEL}" \
  --loss-type rnnt \
  --d-model 256 \
  --num-heads 4 \
  --num-layers 16 \
  --n-mels 80 \
  --conv-kernel 32 \
  --pred-embed-dim 256 \
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
  --paper-peak-factor "${RNNT_PAPER_PEAK_FACTOR}" \
  --warmup-steps "${RNNT_WARMUP_STEPS}" \
  --weight-decay 1e-6 \
  --batch-size "${RNNT_BATCH_SIZE}" \
  --accum-steps "${RNNT_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --ddp-bucket-cap-mb 100 \
  --ddp-grad-as-bucket-view \
  --ddp-timeout-sec "${DDP_TIMEOUT_SEC}" \
  --rnnt-loss-impl torchaudio \
  --rnnt-loss-device cuda \
  --no-rnnt-fused-log-softmax \
  --rnnt-max-batch-tu "${RNNT_MAX_BATCH_TU}" \
  --rnnt-eval-decoder beam \
  --rnnt-eval-beam-size "${RNNT_EVAL_BEAM_SIZE}" \
  --rnnt-eval-beam-topk "${RNNT_EVAL_BEAM_TOPK}" \
  --rnnt-eval-max-symbols-per-step "${RNNT_EVAL_MAX_SYMBOLS_PER_STEP}" \
  --init-encoder-from "${CTC_BEST_FOR_RNNT}" \
  --epochs "${RNNT_EPOCHS}" \
  --ckpt-dir "${RNNT_CKPT_DIR}" \
  --eval-every "${RNNT_EVAL_EVERY}" \
  --save-every "${RNNT_SAVE_EVERY}" \
  --early-stop-patience "${RNNT_EARLY_STOP_PATIENCE}" \
  --no-eval-sample-decode \
  --no-download \
  "${RNNT_EXTRA_ARGS[@]}" \
  "${STAGE_ARGS[@]}"

echo "Pipeline complete."
echo "  Stage-1 CTC dir : ${CTC_CKPT_DIR}"
echo "  Stage-1 exports : ${CTC_EXPORT_DIR}"
echo "  Stage-2 RNN-T dir: ${RNNT_CKPT_DIR}"
