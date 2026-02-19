#!/usr/bin/env bash
set -euo pipefail

# Decoder-only (CTC head) fine-tune on Jerome Powell keyword mix.
# - Freezes encoder
# - Uses target/non-target mixed batches (1:3 target:non-target)
# - Uses SpecAugment from data_loader (train augment=True)
# - Starts from a base CTC checkpoint model weights

DATA_ROOT="${DATA_ROOT:-./dataset/JeromePowell_keyword_mix}"
TARGET_UTT_IDS="${TARGET_UTT_IDS:-${DATA_ROOT}/target_utt_ids.txt}"
BASE_CKPT="./checkpoints_backup_ctc_stage_20260210_182420/best.pt"
CKPT_DIR="${CKPT_DIR:-./checkpoints_keyword_decoder_only_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR="${MIN_LR:-1e-6}"
TARGET_MIX_FRACTION="${TARGET_MIX_FRACTION:-0.25}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EVAL_LM_PATH="${EVAL_LM_PATH:-./lm/3-gram.pruned.1e-7.lower.arpa}"
EVAL_LM_ALPHA="${EVAL_LM_ALPHA:-0.5}"
EVAL_LM_BETA="${EVAL_LM_BETA:-1.0}"
EVAL_LM_BEAM_WIDTH="${EVAL_LM_BEAM_WIDTH:-512}"
BEAM_SIZE="${BEAM_SIZE:-32}"
BEAM_TOKEN_PRUNE="${BEAM_TOKEN_PRUNE:-0}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${TARGET_UTT_IDS}" ]]; then
  echo "Target utt id list not found: ${TARGET_UTT_IDS}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_LM_PATH}" ]]; then
  echo "Eval LM not found: ${EVAL_LM_PATH}" >&2
  exit 1
fi

echo "============================================================"
echo "Keyword Decoder-Only CTC Fine-tune"
echo "============================================================"
echo "DATA_ROOT            : ${DATA_ROOT}"
echo "TARGET_UTT_IDS       : ${TARGET_UTT_IDS}"
echo "BASE_CKPT            : ${BASE_CKPT}"
echo "CKPT_DIR             : ${CKPT_DIR}"
echo "EPOCHS               : ${EPOCHS}"
echo "BATCH_SIZE           : ${BATCH_SIZE}"
echo "ACCUM_STEPS          : ${ACCUM_STEPS}"
echo "LR                   : ${LR}"
echo "TARGET_MIX_FRACTION  : ${TARGET_MIX_FRACTION}"
echo "TRAIN_LOG_EVERY      : ${TRAIN_LOG_EVERY}"
echo "EVAL_EVERY           : ${EVAL_EVERY}"
echo "EVAL_LM_PATH         : ${EVAL_LM_PATH}"
echo "EVAL_LM_ALPHA/BETA   : ${EVAL_LM_ALPHA}/${EVAL_LM_BETA}"
echo "EVAL_LM_BEAM_WIDTH   : ${EVAL_LM_BEAM_WIDTH}"
echo "============================================================"

python3 train.py \
  --loss-type ctc \
  --data-root "${DATA_ROOT}" \
  --train-splits train-clean-100 \
  --val-split dev-clean \
  --tokenizer sp \
  --sp-model ./tokenizer/sp_1k.model \
  --d-model 256 \
  --num-heads 4 \
  --num-layers 16 \
  --n-mels 80 \
  --conv-kernel 32 \
  --resume-path "${BASE_CKPT}" \
  --resume-model-only \
  --no-auto-resume \
  --freeze-encoder \
  --target-utt-ids-path "${TARGET_UTT_IDS}" \
  --target-mix-fraction "${TARGET_MIX_FRACTION}" \
  --train-log-every "${TRAIN_LOG_EVERY}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --accum-steps "${ACCUM_STEPS}" \
  --optimizer adamw \
  --lr-schedule cosine \
  --lr "${LR}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --min-lr "${MIN_LR}" \
  --dropout 0.0 \
  --variational-noise 0.0 \
  --weight-decay 0.0 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --beam-size "${BEAM_SIZE}" \
  --beam-token-prune "${BEAM_TOKEN_PRUNE}" \
  --eval-lm-path "${EVAL_LM_PATH}" \
  --eval-lm-alpha "${EVAL_LM_ALPHA}" \
  --eval-lm-beta "${EVAL_LM_BETA}" \
  --eval-lm-beam-width "${EVAL_LM_BEAM_WIDTH}" \
  --eval-every "${EVAL_EVERY}" \
  --save-every 1 \
  --ckpt-dir "${CKPT_DIR}"
