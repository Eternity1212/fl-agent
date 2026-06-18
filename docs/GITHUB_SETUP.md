# GitHub 推送与登录（本机已预装工具）

本机已完成：

- **`gh` CLI**：`~/bin/gh`（已写入 `~/.bash_profile` 与 `~/.zprofile` 的 `PATH`，新开终端生效；当前 shell 可 `export PATH="$HOME/bin:$PATH"`）。  
- **专用 SSH 密钥**：`~/.ssh/id_ed25519_flagent`（**私钥勿上传、勿提交**）。  
- **`~/.ssh/config`**：已为 `Host github.com` 配置 `IdentityFile ~/.ssh/id_ed25519_flagent` 与 `IdentitiesOnly yes`，并放在通配 `Host *` 之前，避免走错凭据。  
- **仓库远程**：`origin` 已设为 `git@github.com:Eternity1212/fl-agent.git`。

## 你必须完成的一步（一次性）：把公钥加到 GitHub 账户

在浏览器打开：**[SSH and GPG keys](https://github.com/settings/keys)** → **New SSH key** → 粘贴下面整行（Title 随意，例如 `fl-agent-mac`）：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFcsH3FJ8cYnoFWU5WV7/8QW0qy4HJE1M2hiUOGhYRBJ fl-agent-github
```

保存后，在本机验证：

```bash
export PATH="$HOME/bin:$PATH"
ssh -T git@github.com
```

看到 `Hi Eternity1212!` 类似提示即成功。

## 推送 `main` 与 `dev`（公钥生效后执行）

```bash
cd /Users/bytedance/projects/fl-agent
git push -u origin main
git push -u origin dev
```

## 可选：用 `gh` 把公钥写进 GitHub（需先浏览器登录一次）

```bash
export PATH="$HOME/bin:$PATH"
gh auth login -h github.com -p ssh -w
gh ssh-key add ~/.ssh/id_ed25519_flagent.pub -t "fl-agent"
```

## 备选：HTTPS + PAT

若更想用 HTTPS：

```bash
cd /Users/bytedance/projects/fl-agent
git remote set-url origin https://github.com/Eternity1212/fl-agent.git
git push -u origin main
```

用户名：`Eternity1212`，密码处粘贴 **PAT**。

## 本机仓库路径

**`/Users/bytedance/projects/fl-agent`**

- **`main`** / **`dev`**：本地已对齐；待首次 `push` 后远端可见。

## 说明

- 未修改全局 `git config`；历史提交作者为 `Eternity1212 <Eternity1212@users.noreply.github.com>`。  
- **“全自动实时同步”**：GitHub 不会主动拉取你未提交的本地改动；需要 **保存 + commit + push**，或由你在 CI 里配置 token 后由 runner 推送。
