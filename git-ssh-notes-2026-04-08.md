# Git and SSH Notes

Date: `2026-04-08`

## Summary

We worked through Git problems related to staging and pushing commits from this repository while using WSL and a repo located under `/mnt/c/`.

## Main Issues We Found

1. `git add` failed because Git detected "dubious ownership".
2. `git push` failed because the repo remote was using SSH and the SSH setup was not ready.
3. HTTPS did not help at first because the remote URL had not actually changed.
4. SSH commands were confusing because different terminals use different commands.
5. The SSH key files were eventually moved into the correct `~/.ssh` folder.

## What Happened And Why

### 1. Dubious Ownership Error

Error:

```text
fatal: detected dubious ownership in repository at '/mnt/c/Fazal/industrial vision inference service'
```

Cause:

Git saw the repo as being owned by a different user because the project is on a Windows-mounted path under `/mnt/c/`.

Fix used:

```bash
git config --global --add safe.directory '/mnt/c/Fazal/industrial vision inference service'
```

Why this matters:

Git blocks operations in repos it thinks may be unsafe. Adding the path to `safe.directory` tells Git to trust this repo.

### 2. Push Failed Over SSH

Repo remote:

```text
git@github.com:fazafau/industrial-vision-inference-service.git
```

Initial push-related errors included:

- `Could not resolve hostname github.com`
- `Host key verification failed`
- `ssh_askpass not found`

Cause:

The repo was configured to push using SSH, but the SSH environment was not fully set up yet in this shell.

### 3. HTTPS Attempt

We discussed changing the remote to HTTPS:

```bash
git remote set-url origin https://github.com/fazafau/industrial-vision-inference-service.git
```

Important lesson:

After changing a remote, always verify it with:

```bash
git remote -v
```

If `git remote -v` still shows `git@github.com:...`, then Git is still using SSH.

### 4. SSH Setup Details

We created and discussed SSH keys and how to use them.

Key concepts:

- Private key: secret, never share it
- Public key: safe to upload to GitHub
- SSH folder: `~/.ssh`

Correct SSH file locations now:

- `~/.ssh/id_ed25519`
- `~/.ssh/id_ed25519.pub`
- `~/.ssh/known_hosts`

What each file does:

- `id_ed25519`: your private SSH key
- `id_ed25519.pub`: your public SSH key for GitHub
- `known_hosts`: remembers trusted servers such as `github.com`

### 5. Terminal Differences

Some commands did not work because they depend on which shell is being used.

In Bash/WSL:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

In PowerShell:

```powershell
Start-Service ssh-agent
ssh-add ~/.ssh/id_ed25519
Get-Content ~/.ssh/id_ed25519.pub
```

Important lesson:

- `eval` is a Bash command
- `cat` works in Bash, and often in PowerShell too, but `Get-Content` is the native PowerShell command

### 6. `chmod` Explanation

We discussed `chmod` and file permissions.

Common SSH permissions:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

Meaning:

- `700` on `~/.ssh`: only you can access the folder
- `600` on private key: only you can read/write it
- `644` on public key: you can read/write it, others can read it

Why this matters:

SSH may refuse to use a private key if the permissions are too open.

### 7. Where The Keys Went

We confirmed the keys are now stored in:

```text
/home/fazal_kadri/.ssh
```

That is your SSH folder in WSL.

## Useful Commands We Covered

Check repo status:

```bash
git status
```

Check remotes:

```bash
git remote -v
```

Mark repo as safe:

```bash
git config --global --add safe.directory '/mnt/c/Fazal/industrial vision inference service'
```

Test GitHub SSH:

```bash
ssh -T git@github.com
```

View SSH files:

```bash
ls -la ~/.ssh
```

View public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Add private key to `ssh-agent`:

```bash
ssh-add ~/.ssh/id_ed25519
```

## Typical Git Workflow After Setup

```bash
git add .
git commit -m "your message"
git push
```

## What Should Work Going Forward

- You should not need to recreate the SSH keys every time.
- You should not need to re-add the Git `safe.directory` setting for this repo unless your Git config changes.
- You may sometimes need to start `ssh-agent` again after restarting your machine or terminal.
- You may sometimes need to run `ssh-add ~/.ssh/id_ed25519` again if the key is not loaded into the agent.

## Best Practices From Today

- Keep SSH keys in `~/.ssh`, not in the repo
- Never share the private key
- Verify remote URLs with `git remote -v` after changing them
- When a command fails, note which terminal you are in: Bash, WSL, PowerShell, or CMD
- For repos under `/mnt/c/` in WSL, `safe.directory` issues can happen

## If Push Fails Again Later

Check these in order:

1. `git status`
2. `git remote -v`
3. `ssh -T git@github.com`
4. `ls -la ~/.ssh`
5. `ssh-add ~/.ssh/id_ed25519`
