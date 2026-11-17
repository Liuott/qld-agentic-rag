# Secrets and history

## What leaked

Commit `c1a4e2e` (and earlier) contains a tracked `.env` file with **real
credentials** in a public repository:

| Variable | Value prefix | Provider |
|---|---|---|
| `LANGSMITH_API_KEY` | `lsv2_pt_…` | LangSmith |
| `QWEN_KEY` | `sk-…` | DashScope / Qwen |

Anyone who ran `git clone` between 2025-11-12 and the v2 cleanup has them.
Treat both keys as compromised **even after** the file is deleted from the tip:
git keeps history, and forks/caches can keep copies for a long time.

## 1. Rotate first (do this before anything else)

1. LangSmith → *Settings → API keys* → revoke every key created before the
   cleanup (they are all in the git history), create a new one.
2. Alibaba Cloud / DashScope console → revoke the `sk-…` key that was in `.env`,
   create a new one.
3. Check usage/billing for the period since 2025-11-12.
4. Put the new values only in your local `.env` (git-ignored) — never in git.

## 2. Remove the file from the working tree and the index

```bash
git rm --cached .env
cp .env.example .env      # re-add your (new) values locally
git commit -m "chore: stop tracking .env"
```

`.gitignore` now ignores `.env`, `.env.*` and `!.env.example`, so this cannot
silently happen again. Verify with:

```bash
git check-ignore -v .env
```

## 3. Purge the secret from history

**Status: done on 2026-11-17.** The published history was replaced by a single
squashed commit (`0eebfa7`, "chore: v2 repository reset"), so no commit
reachable from `main` contains `.env`, `.venv/`, `storage/` or `__pycache__/`
any more. `main` is now 49 files / ~20 MB.

Two caveats:

* GitHub keeps **unreachable** objects around until it garbage-collects them, so
  the old blobs (including the leaked keys) can still be fetched by their SHA
  for a while. `git cat-file -p <old-sha>` on a clone is enough to retrieve
  them. That is why rotating the keys in step 1 is not optional.
  To have them purged immediately, open a GitHub Support ticket and point at
  the repository and the commit range — only the repository owner can do this.
* The repository size reported by the API stays at the old value (~292 MB) until
  that collection happens. Do not be surprised if it looks unchanged.

For reference (other repositories, or if you ever restore the old history), the
generic rewrite recipe is `git filter-repo` via:

```bash
pip install git-filter-repo
bash scripts/purge-history.sh
git push --force-with-lease origin main
```

It rewrites every commit to drop `.venv/`, `storage/`, `__pycache__/` and
`.env`. **This changes every commit id** — coordinate with anyone who has a
clone. It was not used here because the history could not be downloaded in full
over this network; squashing achieved the same result with the tree that was
already verified locally.

## 4. Prevent recurrence

* `.gitignore` covers `.env*`, `.venv/`, `storage/`, `__pycache__/`, `*.sqlite3`.
* CI runs `ruff` (which flags accidentally committed debug code) and the test
  suite; consider adding `gitleaks` or `detect-secrets` as a pre-commit hook:

  ```bash
  pip install detect-secrets
  detect-secrets scan > .secrets.baseline
  ```
* Never put keys in code, notebooks, or commit messages — use `.env` locally and
  the platform's secret store in CI.
