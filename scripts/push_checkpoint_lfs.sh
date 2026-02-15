#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/push_checkpoint_lfs.sh [options]

Options:
  -f, --file PATH       Checkpoint path (default: checkpoints/best.pt)
  -r, --remote NAME     Git remote (default: origin)
  -b, --branch NAME     Target branch (default: current branch)
  -m, --message TEXT    Commit message
  -n, --dry-run         Show actions without commit/push
  -h, --help            Show this help

Examples:
  scripts/push_checkpoint_lfs.sh
  scripts/push_checkpoint_lfs.sh --file checkpoints/best.pt --remote origin --branch main
EOF
}

FILE="checkpoints/best.pt"
REMOTE="origin"
BRANCH=""
MESSAGE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      FILE="${2:-}"
      shift 2
      ;;
    -r|--remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    -b|--branch)
      BRANCH="${2:-}"
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed." >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is not installed. Install it first." >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Not inside a git repository." >&2
  exit 1
fi
cd "${REPO_ROOT}"

if [[ "${FILE}" = /* ]]; then
  case "${FILE}" in
    "${REPO_ROOT}"/*) FILE="${FILE#${REPO_ROOT}/}" ;;
    *)
      echo "--file must be inside the git repository." >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi

if [[ -z "${MESSAGE}" ]]; then
  MESSAGE="Track ${FILE} via Git LFS"
fi

if [[ ! -f "${FILE}" ]]; then
  echo "Checkpoint file not found: ${FILE}" >&2
  exit 1
fi

echo "Repo: ${REPO_ROOT}"
echo "File: ${FILE}"
echo "Remote: ${REMOTE}"
echo "Branch: ${BRANCH}"

git lfs install --local >/dev/null
git lfs track "${FILE}" >/dev/null

if [[ ! -f .gitattributes ]]; then
  echo "Failed to create .gitattributes with LFS tracking rule." >&2
  exit 1
fi

git add -- .gitattributes "${FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] Staged files:"
  git diff --cached --name-status -- .gitattributes "${FILE}" || true
  echo "[dry-run] Would run: git commit -m \"${MESSAGE}\" -- .gitattributes \"${FILE}\""
  echo "[dry-run] Would run: git push ${REMOTE} HEAD:${BRANCH}"
  exit 0
fi

if git diff --cached --quiet -- .gitattributes "${FILE}"; then
  echo "No staged changes for .gitattributes or ${FILE}. Skipping commit."
else
  git commit -m "${MESSAGE}" -- .gitattributes "${FILE}"
fi

git push "${REMOTE}" "HEAD:${BRANCH}"

echo "Done. LFS-tracked checkpoint pushed to ${REMOTE}/${BRANCH}."
echo "Verify with: git lfs ls-files | grep -F \"${FILE}\""
