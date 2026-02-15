#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/push_checkpoint_lfs.sh [options]

Description:
  Upload a model checkpoint or artifact to Hugging Face Hub.

Options:
  -f, --file PATH            Local file path to upload (default: checkpoints/best.pt)
  -r, --repo REPO_ID         Hugging Face repo id (default: jtang25/asr)
  -t, --repo-type TYPE       Repo type: model|dataset|space (default: model)
  -p, --path-in-repo PATH    Destination path in repo (default: same as --file)
  -m, --message TEXT         Commit message
  -n, --dry-run              Show command without uploading
  -h, --help                 Show this help

Examples:
  scripts/push_checkpoint_lfs.sh
  scripts/push_checkpoint_lfs.sh --file checkpoints_jp_only_ctc_no_vn/best.pt
  scripts/push_checkpoint_lfs.sh --file lm/3-gram.pruned.1e-7.lower.arpa --repo jtang25/asr
USAGE
}

FILE="checkpoints/best.pt"
REPO_ID="jtang25/asr"
REPO_TYPE="model"
PATH_IN_REPO=""
MESSAGE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      FILE="${2:-}"
      shift 2
      ;;
    -r|--repo)
      REPO_ID="${2:-}"
      shift 2
      ;;
    -t|--repo-type)
      REPO_TYPE="${2:-}"
      shift 2
      ;;
    -p|--path-in-repo)
      PATH_IN_REPO="${2:-}"
      shift 2
      ;;
    -m|--message)
      MESSAGE="${2:-}"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${FILE}" ]]; then
  echo "Artifact file not found: ${FILE}" >&2
  exit 1
fi

if [[ -z "${PATH_IN_REPO}" ]]; then
  PATH_IN_REPO="${FILE}"
fi

if [[ -z "${MESSAGE}" ]]; then
  MESSAGE="Upload ${PATH_IN_REPO}"
fi

HF_BIN=""
if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
elif [[ -x "/root/.local/bin/hf" ]]; then
  HF_BIN="/root/.local/bin/hf"
else
  echo "Hugging Face CLI not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi

echo "Local file: ${FILE}"
echo "Repo id: ${REPO_ID}"
echo "Repo type: ${REPO_TYPE}"
echo "Path in repo: ${PATH_IN_REPO}"

CMD=("${HF_BIN}" upload "${REPO_ID}" "${FILE}" "${PATH_IN_REPO}" --repo-type "${REPO_TYPE}" --commit-message "${MESSAGE}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[dry-run]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

echo "Upload complete: hf.co/${REPO_ID}/${PATH_IN_REPO}"
