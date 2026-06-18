# 仓库同步与推送（日常流程）

GitHub **不会**自动拉取你磁盘上未提交的改动；「同步」指两件事：**与远端对齐（pull）** 和 **把本地提交送上去（push）**。

## 推荐节奏

1. 开始干活前：先 **拉远端**（避免分叉太久）。  
2. 改代码 → `git add` → `git commit`。  
3. 阶段性完成： **`git push`** 到对应分支。

## 脚本（在仓库根目录执行）

| 脚本 | 作用 |
|------|------|
| [`scripts/sync-from-remote.sh`](../scripts/sync-from-remote.sh) | `git fetch` + 若已设置上游则 `git pull --rebase` |
| [`scripts/push-current-branch.sh`](../scripts/push-current-branch.sh) | 推送**当前分支**到 `origin` 并设置上游 |
| [`scripts/push-github.sh`](../scripts/push-github.sh) | 仅推送 `main` 与 `dev`（里程碑/发版用） |
| [`scripts/publish.sh`](../scripts/publish.sh) | **先 sync 再 push 当前分支**（最常用的一键） |
| [`scripts/smoke.sh`](../scripts/smoke.sh) | **本地 smoke**：`ruff` + `pytest`（建议每次 push 前执行） |

## 推送前自检（推荐）

```bash
./scripts/smoke.sh
./scripts/publish.sh
```

## 分支习惯（与 `docs/BRANCHING.md` 一致）

- 小功能：`feat/xxx` → 合并到 `dev` → 稳定后再合 `main`。  
- 紧急修复：`hotfix/xxx` → 直接 PR 到 `main`（可选）。

## 说明

- `git pull --rebase` 在本地有未推送提交时更干净；若发生冲突按提示解决后 `git rebase --continue`。  
- 不要把 `data/raw/` 或权重提交进仓库；大文件用外部存储或 Git LFS（另议）。
