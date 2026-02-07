#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/download_librispeech_lm.sh [--out-dir DIR] [--variant pruned3|full4]

Variants:
  pruned3  -> 3-gram pruned LM (smaller, faster download)
  full4    -> full 4-gram LM (much larger)
EOF
}

OUT_DIR="./lm"
VARIANT="pruned3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
      shift 2
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

case "${VARIANT}" in
  pruned3)
    URL="https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    GZ_NAME="3-gram.pruned.1e-7.arpa.gz"
    ;;
  full4)
    URL="https://www.openslr.org/resources/11/4-gram.arpa.gz"
    GZ_NAME="4-gram.arpa.gz"
    ;;
  *)
    echo "Invalid --variant: ${VARIANT}. Use pruned3 or full4." >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"
GZ_PATH="${OUT_DIR}/${GZ_NAME}"
ARPA_PATH="${GZ_PATH%.gz}"

if [[ ! -f "${GZ_PATH}" ]]; then
  echo "Downloading ${URL} -> ${GZ_PATH}"
  curl -L "${URL}" -o "${GZ_PATH}"
else
  echo "Already downloaded: ${GZ_PATH}"
fi

if [[ ! -f "${ARPA_PATH}" ]]; then
  echo "Extracting ${GZ_PATH} -> ${ARPA_PATH}"
  gunzip -c "${GZ_PATH}" > "${ARPA_PATH}"
else
  echo "Already extracted: ${ARPA_PATH}"
fi

echo "LM ready: ${ARPA_PATH}"
