#!/usr/bin/env bash
set -euo pipefail

# One-command Jerome Powell CTC fine-tune launcher.
# Uses:
# - base checkpoint: checkpoints_conformer_m_ctc_then_rnnt/ctc_stage/best.pt
# - all data under dataset/JeromePowell
# - eval split created from chapter holdout (VAL_CHAPTERS)
# - CTC+LM evaluation with beam search

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SRC_ROOT="${SRC_ROOT:-./dataset/JeromePowell}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
# Use an isolated prepared dataset per run by default so concurrent launches
# cannot clobber files for an in-flight training job.
PREP_ROOT="${PREP_ROOT:-./dataset/_finetune_jerome_powell_${RUN_TAG}}"
BASE_CKPT="${BASE_CKPT:-./checkpoints_conformer_m_ctc_then_rnnt/ctc_stage/best.pt}"
CKPT_DIR="${CKPT_DIR:-./checkpoints_finetune_jerome_powell_ctc_${RUN_TAG}}"

# Fine-tuning defaults (override via env vars as needed)
# Keep effective batch at 128 while reducing per-step memory pressure.
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LR="${LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
MIN_LR="${MIN_LR:-1e-6}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
EPOCHS="${EPOCHS:-200}"
VAL_CHAPTERS="${VAL_CHAPTERS:-1}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EXCLUDE_CHAPTERS="${EXCLUDE_CHAPTERS-008}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# LM decode settings for eval quality
EVAL_LM_PATH="${EVAL_LM_PATH:-./lm/4-gram.lower.arpa}"
EVAL_LM_ALPHA="${EVAL_LM_ALPHA:-0.5}"
EVAL_LM_BETA="${EVAL_LM_BETA:-1.0}"
EVAL_LM_BEAM_WIDTH="${EVAL_LM_BEAM_WIDTH:-512}"
BEAM_SIZE="${BEAM_SIZE:-32}"
BEAM_TOKEN_PRUNE="${BEAM_TOKEN_PRUNE:-0}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_LM_PATH}" ]]; then
  echo "Eval LM not found: ${EVAL_LM_PATH}" >&2
  exit 1
fi

echo "============================================================"
echo "Jerome Powell CTC Fine-tune"
echo "============================================================"
echo "SRC_ROOT      : ${SRC_ROOT}"
echo "PREP_ROOT     : ${PREP_ROOT}"
echo "BASE_CKPT     : ${BASE_CKPT}"
echo "CKPT_DIR      : ${CKPT_DIR}"
echo "BATCH_SIZE    : ${BATCH_SIZE}"
echo "ACCUM_STEPS   : ${ACCUM_STEPS}"
echo "LR            : ${LR}"
echo "WARMUP_STEPS  : ${WARMUP_STEPS}"
echo "MIN_LR        : ${MIN_LR}"
echo "EPOCHS(add)   : ${FINETUNE_EPOCHS}"
echo "EPOCHS(total) : ${EPOCHS}"
echo "VAL_CHAPTERS  : ${VAL_CHAPTERS}"
if [[ -n "${EXCLUDE_CHAPTERS}" ]]; then
  echo "EXCLUDE_CHAPS : ${EXCLUDE_CHAPTERS}"
else
  echo "EXCLUDE_CHAPS : <none>"
fi
echo "EARLY STOP    : ${EARLY_STOP_PATIENCE}"
echo "ALLOC CONF    : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "EVAL LM       : ${EVAL_LM_PATH}"
echo "============================================================"

NPROC_PER_NODE="${NPROC_PER_NODE}" \
SRC_ROOT="${SRC_ROOT}" \
PREP_ROOT="${PREP_ROOT}" \
BASE_CKPT="${BASE_CKPT}" \
CKPT_DIR="${CKPT_DIR}" \
BATCH_SIZE="${BATCH_SIZE}" \
ACCUM_STEPS="${ACCUM_STEPS}" \
LR="${LR}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
MIN_LR="${MIN_LR}" \
FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
EPOCHS="${EPOCHS}" \
VAL_CHAPTERS="${VAL_CHAPTERS}" \
EXCLUDE_CHAPTERS="${EXCLUDE_CHAPTERS}" \
EVAL_EVERY="${EVAL_EVERY}" \
SAVE_EVERY="${SAVE_EVERY}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
bash scripts/finetune_conformer_m_ctc_dataset.sh \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --eval-lm-path "${EVAL_LM_PATH}" \
  --eval-lm-alpha "${EVAL_LM_ALPHA}" \
  --eval-lm-beta "${EVAL_LM_BETA}" \
  --eval-lm-beam-width "${EVAL_LM_BEAM_WIDTH}" \
  --beam-size "${BEAM_SIZE}" \
  --beam-token-prune "${BEAM_TOKEN_PRUNE}" \
  "$@"
