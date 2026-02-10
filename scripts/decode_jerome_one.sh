#!/usr/bin/env bash
set -euo pipefail

# Change only this value (or pass it as arg 1).
REL_PATH_DEFAULT="9999/007/9999-007-0327.flac"

REL_PATH="${1:-${REL_PATH_DEFAULT}}"
DATA_ROOT="${DATA_ROOT:-dataset/_eval_jerome_powell_all_clean/LibriSpeech/dev-clean}"
INPUT_PATH="${DATA_ROOT}/${REL_PATH}"

CKPT="${CKPT:-checkpoints_finetune_jerome_powell_ctc_20260210_205709/last.pt}"
LM_PATH="${LM_PATH:-lm/4-gram.lower.bin}"
LM_FALLBACK_ARPA="${LM_FALLBACK_ARPA:-lm/4-gram.lower.arpa}"

LM_ALPHA="${LM_ALPHA:-0.5}"
LM_BETA="${LM_BETA:-1.0}"
LM_BEAM_WIDTH="${LM_BEAM_WIDTH:-512}"
BEAM_SIZE="${BEAM_SIZE:-32}"
BEAM_TOKEN_PRUNE="${BEAM_TOKEN_PRUNE:-0}"

PY_BIN="${PY_BIN:-/venv/main/bin/python3}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Input not found: ${INPUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${LM_PATH}" ]]; then
  echo "LM file not found: ${LM_PATH}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH." >&2
  exit 1
fi

ext="${INPUT_PATH##*.}"
ext="${ext,,}"
WAV_PATH="${INPUT_PATH%.*}.wav"

if [[ "${ext}" == "wav" || "${ext}" == "wave" ]]; then
  WAV_PATH="${INPUT_PATH}"
else
  ffmpeg -y -i "${INPUT_PATH}" -ac 1 -ar 16000 -c:a pcm_s16le "${WAV_PATH}"
fi

echo "Decoding: ${WAV_PATH}"

decode_once() {
  local lm_path="$1"
  "${PY_BIN}" transcribe.py \
    --checkpoint "${CKPT}" \
    --audio "${WAV_PATH}" \
    --lm-path "${lm_path}" \
    --lm-alpha "${LM_ALPHA}" \
    --lm-beta "${LM_BETA}" \
    --lm-beam-width "${LM_BEAM_WIDTH}" \
    --beam-size "${BEAM_SIZE}" \
    --beam-token-prune "${BEAM_TOKEN_PRUNE}"
}

tmp_out="$(mktemp)"
decode_once "${LM_PATH}" 2>&1 | tee "${tmp_out}"
transcript="$(sed -n 's/^Transcript:[[:space:]]*//p' "${tmp_out}" | tail -n 1)"
rm -f "${tmp_out}"

# pyctcdecode cannot infer unigrams from .bin KenLM files; this can
# occasionally produce empty outputs. Retry with .arpa for quality.
lm_path_lower="${LM_PATH,,}"
if [[ -z "${transcript}" && "${lm_path_lower}" == *.bin && -f "${LM_FALLBACK_ARPA}" ]]; then
  echo "[WARN] Empty transcript with binary LM; retrying with ARPA LM: ${LM_FALLBACK_ARPA}" >&2
  decode_once "${LM_FALLBACK_ARPA}"
fi
