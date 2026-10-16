"""在 github.com:443 被阻断时，改用 GitHub API 推送。

背景：本机 `github.com:443` 连不上（git 的 HTTPS/SSH 都走这个域名），
但实测 `api.github.com:443` 是通的，而且 token 有效。
所以把 git 对象（blob/tree/commit）通过 REST API 建上去，再移动 ref ——
效果等同于 `git push`。

用法（token 从环境变量读，不落盘、不进命令行历史）：
    GHP=<token> py -3 tools/push_via_api.py
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.github.com"
OWNER = "yeggogg"
REPO = "nici-town"
BRANCH = "main"


def git(*args: str) -> bytes:
    """跑一个 git 命令，返回原始字节（大文件不能按文本解码）。"""
    p = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败:\n{p.stderr.decode('utf-8', 'replace')}")
    return p.stdout


def api(method: str, path: str, token: str, payload=None):
    url = f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "nici-push/1.0")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"API {method} {path} -> HTTP {e.code}\n{detail}")


def main() -> None:
    token = os.environ.get("GHP") or os.environ.get("GH_TOKEN") or ""
    if not token:
        raise SystemExit("请用环境变量传 token：GHP=<token> py -3 tools/push_via_api.py")

    head = git("rev-parse", "HEAD").decode().strip()

    # 远端当前 main（也是这次提交的父提交）
    ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}", token)
    parent = ref["object"]["sha"]

    if parent == head:
        print("远端已经是最新，无需推送")
        return

    # 找出本地比远端多的提交
    #
    # 注意：远端那个提交是**上一次 API 推送时在服务端重建的**，
    # 本机并没有这个对象 —— 直接 `git log <远端sha>..HEAD` 会报
    # "Invalid revision range"。所以改成用 **tree 反查**：
    # 远端 tree 一定等于本地某个提交的 tree（内容相同），拿那个当差异基准。
    remote_commit = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{parent}", token)
    remote_tree = remote_commit["tree"]["sha"]

    local_pairs = git("log", "--format=%H %T", "-200", head).decode().split()
    pairs = list(zip(local_pairs[0::2], local_pairs[1::2]))
    base = next((h for h, t in pairs if t == remote_tree), None)

    if base is None:
        raise SystemExit(
            f"本地找不到 tree 为 {remote_tree[:9]} 的提交。\n"
            "说明远端有本地没有的内容，需要先拉取（git fetch）再推。"
        )
    print(f"差异基准（本地 {base[:9]} ≡ 远端 {parent[:9]} 的 tree）")

    if base == head:
        print("内容已经一致，无需推送")
        return

    log = git("log", "--format=%H%x00%s%x00%b", f"{base}..{head}").decode("utf-8", "replace")
    if not log.strip():
        print("没有新提交")
        return

    # 本次要动的文件（相对基准提交）
    name_status = git("diff", "--name-status", base, head).decode("utf-8", "replace")
    changes = []
    for line in name_status.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        changes.append((status, path))

    print(f"父提交 {parent[:9]}  ->  本地 {head[:9]}")
    print(f"变更 {len(changes)} 个文件\n")

    # 父提交的 tree 作为基底，只提交有变化的部分
    parent_commit = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{parent}", token)
    base_tree = parent_commit["tree"]["sha"]

    tree_entries = []
    for status, path in changes:
        if status == "D":
            tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            print(f"  删除 {path}")
            continue

        blob = git("cat-file", "blob", f"{head}:{path}")
        res = api("POST", f"/repos/{OWNER}/{REPO}/git/blobs", token, {
            "content": base64.b64encode(blob).decode(),
            "encoding": "base64",
        })
        # 权限位：可执行文件保持 755。
        # 注意是 `ls-tree <commit> -- <path>`；写成 `<commit>:<path>` 是给
        # tree 对象用的语法，指向文件会报 "not a tree object"。
        entry = git("ls-tree", head, "--", path).decode("utf-8", "replace").strip()
        mode = entry.split()[0] if entry else "100644"
        if mode not in ("100644", "100755"):
            mode = "100644"
        tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": res["sha"]})
        print(f"  {status} {path:34s} {len(blob) // 1024:>5d} KB")

    tree = api("POST", f"/repos/{OWNER}/{REPO}/git/trees", token, {
        "base_tree": base_tree,
        "tree": tree_entries,
    })

    # 提交：把本地 HEAD 的标题和正文原样带过去
    subject = git("log", "-1", "--format=%s", head).decode("utf-8", "replace").strip()
    body = git("log", "-1", "--format=%b", head).decode("utf-8", "replace").strip()
    message = subject + ("\n\n" + body if body else "")

    commit = api("POST", f"/repos/{OWNER}/{REPO}/git/commits", token, {
        "message": message,
        "tree": tree["sha"],
        "parents": [parent],
    })

    api("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", token, {
        "sha": commit["sha"],
        "force": False,
    })

    print(f"\n已推送：{parent[:9]} -> {commit['sha'][:9]}")
    print(f"commit: {commit['html_url']}")


if __name__ == "__main__":
    main()
