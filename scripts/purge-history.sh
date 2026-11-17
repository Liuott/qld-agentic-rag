#!/usr/bin/env bash
# Remove .venv/, storage/, __pycache__/ and .env from every commit.
#
#   pip install git-filter-repo
#   bash scripts/purge-history.sh
#   git push --force-with-lease origin main
#
# WARNING: rewrites history (all commit ids change). Back up the repository
# first:  git clone --mirror . ../qld-agentic-rag-backup.git
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo not found. Install it with: pip install git-filter-repo" >&2
  exit 1
fi

before=$(du -sh .git | cut -f1)
echo "repository size before: ${before}"

git filter-repo --force \
  --path .venv --invert-paths \
  --path storage --invert-paths \
  --path-glob '__pycache__' --invert-paths \
  --path-glob '*.pyc' --invert-paths \
  --path .env --invert-paths

git reflog expire --expire=now --all
git gc --prune=now --aggressive

after=$(du -sh .git | cut -f1)
echo "repository size after:  ${after}"
git remote -v || echo "NOTE: git-filter-repo removes remotes; re-add with 'git remote add origin <url>'"
