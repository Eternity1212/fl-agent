# Git / GitHub workflow

## Branch roles

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases; merge only via PR; tag `v0.x`. |
| `dev` | Daily integration; merge `feat/*` here first. |
| `feat/<topic>` | Short-lived features (e.g. `feat/fed-core`). |
| `exp/<slug>` | Optional experiment snapshots (may not merge to `main`). |

## Rollback

- Prefer **`git revert`** + PR on `main`.  
- Avoid **`git push --force`** on `main`.

## Sync

- GitHub does not auto-pull local uncommitted work.  
- Push after meaningful commits; use CI (`.github/workflows/ci.yml`) on each push.

## Auth

- HTTPS: Personal Access Token (PAT) with `repo` scope, or  
- SSH: `git@github.com:Eternity1212/fl-agent.git`
