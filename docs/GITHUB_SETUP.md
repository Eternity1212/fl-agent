# GitHub 推送与登录（一次性配置）

当前环境 **未配置** GitHub 凭据，因此自动化 `git push` 会失败。任选其一完成认证后，在仓库根目录执行推送命令即可。

## 方案 A：HTTPS + Personal Access Token（PAT，适合快速开始）

1. 在 GitHub 创建 **Fine-grained PAT** 或 **classic PAT**（`repo` 权限）。  
2. macOS 可在首次 `git push` 时用钥匙串保存用户名/Token；或：  
   ```bash
   cd /Users/bytedance/projects/fl-agent
   git push -u origin main
   git push -u origin dev
   ```  
   用户名填 GitHub 用户名，密码填 **Token**（不是账户密码）。

## 方案 B：SSH（适合长期使用）

1. 生成密钥（若还没有）：`ssh-keygen -t ed25519 -C "your_email@example.com"`  
2. 将 `~/.ssh/id_ed25519.pub` 添加到 GitHub → Settings → SSH keys。  
3. 切换远程并推送：  
   ```bash
   cd /Users/bytedance/projects/fl-agent
   git remote set-url origin git@github.com:Eternity1212/fl-agent.git
   git push -u origin main
   git push -u origin dev
   ```

## 方案 C：GitHub CLI（`gh`）

```bash
brew install gh
gh auth login
cd /Users/bytedance/projects/fl-agent
git push -u origin main
git push -u origin dev
```

## 本机仓库路径

克隆与工作目录：**`/Users/bytedance/projects/fl-agent`**

- 默认分支：**`main`**（已含首次提交 `e0e5dc8`）  
- 集成分支：**`dev`**（与 `main` 当前同提交，便于后续从 `feat/*` 合并）

## 说明

- 未替你执行 **`git config`** 全局修改；提交作者使用了 `--author="Eternity1212 <Eternity1212@users.noreply.github.com>"`（可按需改为你的 GitHub 绑定邮箱）。  
- **实时自动同步**：需在你本机保存凭据后，由你或 CI 在 `git push` 时上传；本环境无法代替你完成浏览器 OAuth。
