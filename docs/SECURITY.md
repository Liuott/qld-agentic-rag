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

`git filter-repo` is the supported tool (install once with
`pip install git-filter-repo`):

```bash
bash scripts/purge-history.sh
```

The script rewrites every commit to drop `.venv/`, `storage/`, `__pycache__/`
and `.env`, then prints the resulting repository size. **This rewrites commit
ids** — coordinate with anyone who has a clone, then force-push:

```bash
git push --force-with-lease origin main
```

If other people already cloned the repository, ask them to delete their copy
(GitHub's cached views / forks cannot be rewritten from your side; contact
GitHub Support if the key was live for a long time).

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
