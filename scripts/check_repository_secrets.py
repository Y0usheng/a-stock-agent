"""检查Git暂存文件及可选的可达历史，不读取本机登录目录，不输出匹配的凭据内容。"""
from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess


PATTERNS = {
    "openai_api_key": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "jwt_token": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "refresh_token": re.compile(rb"\brt_[A-Za-z0-9_-]{25,}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "credential_value": re.compile(rb'''(?i)["']?(?:access_token|refresh_token|id_token|OPENAI_API_KEY|CODEX_API_KEY|CODEX_ACCESS_TOKEN|GH_TOKEN|GITHUB_TOKEN)["']?\s*[:=]\s*["'][A-Za-z0-9_./+\-=]{24,}["']'''),
}


def protected_path(path):
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    parts = tuple(part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts)
    if name == ".env.example":
        return False
    return (name == ".env" or name.startswith(".env.") or
            name in {"auth.json", "chatgpt-auth.json", "credentials.json", "id_rsa", "id_ed25519", "cookies.json", "cookies.txt"} or
            name.endswith((".pem", ".key", ".p12", ".pfx")) or
            any(part in {".codex", ".ssh"} for part in parts) or
            bool(parts and parts[0] == "data") or
            (len(parts) > 1 and parts[:2] in {("logs", "live"), ("logs", "decisions"), ("logs", "reviews"), ("logs", "research")}))


def scan_content(path, content):
    findings = [{"path": path, "rule": "protected_path", "line": None}] if protected_path(path) else []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(content):
            findings.append({"path": path, "rule": kind, "line": content.count(b"\n", 0, match.start()) + 1})
    return findings


def git(*args):
    result = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError("Git读取失败；请在仓库中运行，并确认历史已完整获取")
    return result.stdout


def repository_objects(history=False):
    objects = {}
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise RuntimeError("存在尚未解决的Git合并冲突")
        if mode == b"160000":
            raise RuntimeError("子模块需在其仓库中单独检查")
        objects.setdefault(oid.decode(), set()).add(name.decode("utf-8"))
    if history:
        if git("rev-parse", "--is-shallow-repository").strip() == b"true":
            raise RuntimeError("浅克隆不包含完整历史；请先获取完整历史")
        for line in git("rev-list", "--objects", "--all").splitlines():
            oid, _, name = line.partition(b" ")
            if name:
                objects.setdefault(oid.decode(), set()).add(name.decode("utf-8"))
    return objects


def scan_repository(history=False):
    objects = repository_objects(history)
    findings, blobs = [], 0
    # One read-only batch: object names go to Git, file contents never go to stdout.
    process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for oid, paths in objects.items():
            process.stdin.write((oid + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or not header[2].isdigit():
                raise RuntimeError("Git对象缺失，不能给出通过结论")
            size = int(header[2])
            content = process.stdout.read(size)
            if len(content) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Git对象读取不完整")
            if header[1] != b"blob":
                continue
            blobs += 1
            for path in sorted(paths):
                findings.extend({**row, "object": oid[:12]} for row in scan_content(path, content))
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait(timeout=10)
    return {"status": "failed" if findings else "passed", "scope": "index_and_reachable_history" if history else "git_index",
            "blobs_scanned": blobs, "findings": findings,
            "limitations": "模式检查不能证明不存在任意格式的秘密；不读取本机凭据，不检查未暂存文件、GitHub附件或不可达对象"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="同时扫描当前本地所有引用可达的历史；建议先fetch远端")
    args = parser.parse_args()
    try:
        report = scan_repository(args.history)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    except (OSError, RuntimeError, subprocess.SubprocessError):
        print(json.dumps({"status": "error", "message": "仓库检查未完成，不能给出无凭据结论"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
