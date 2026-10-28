"""导出、验证并发布虚拟账户的公开日报；不复制数据库、登录缓存或模型输入。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import getproxies

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.subscription import subscription_workspace
from config import DATA_DIR, ROOT
from evals.public_reports import (MAX_BYTES, REPORT_FILE, build_site, filename, make_report,
                                 report_days, validate_report, write_report)

BRANCH = "public-reports"


def command(args, *, data=None, env=None, timeout=90):
    environment = dict(os.environ) if env is None else dict(env)
    for scheme, proxy in getproxies().items():
        if scheme in {"http", "https"}:
            environment.setdefault(scheme.upper() + "_PROXY", proxy)
    result = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            cwd=ROOT, env=environment, timeout=timeout)
    if result.returncode:
        raise RuntimeError("发布命令失败；检查网络、GitHub登录、分支更新冲突或Pages工作流状态。原始认证输出未记录。")
    return result.stdout


def read_public_files(root):
    files = {}
    for path in sorted((Path(root) / "reports").glob("*.json")):
        if path.is_symlink() or not REPORT_FILE.fullmatch(path.name) or path.stat().st_size > MAX_BYTES:
            raise ValueError("待发布目录包含无效报告")
        report = json.loads(path.read_text(encoding="utf-8"))
        data = validate_report(report)
        if filename(report) != path.name:
            raise ValueError("报告路径与内容身份不符")
        files["reports/" + path.name] = data
    if not files:
        raise ValueError("没有可发布的真实报告")
    return files


def publish(root, repository, *, deploy=True):
    # Explicit destination binding: never silently push to another repository.
    remote = command(["git", "remote", "get-url", "origin"]).decode().strip().removesuffix(".git")
    if remote not in {"https://github.com/" + repository, "git@github.com:" + repository}:
        raise ValueError("origin与指定发布仓库不一致")
    files = read_public_files(root)
    heads = command(["git", "ls-remote", "--heads", "origin", "refs/heads/" + BRANCH]).decode().strip()
    parent = heads.split()[0] if heads else None
    existing = {}
    if parent:
        command(["git", "fetch", "--no-tags", "origin", "refs/heads/" + BRANCH])
        entries = command(["git", "ls-tree", "-r", "-z", parent]).split(b"\0")
        if len(entries) > 5001:
            raise ValueError("公开历史数量超限，请先归档")
        for entry in entries:
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            path = name.decode()
            if mode != b"100644" or kind != b"blob" or not path.startswith("reports/") or not REPORT_FILE.fullmatch(path[8:]):
                raise ValueError("公开分支包含不属于日报协议的文件")
            if int(command(["git", "cat-file", "-s", oid.decode()])) > MAX_BYTES:
                raise ValueError("公开报告大小超限")
            data = command(["git", "cat-file", "blob", oid.decode()])
            validate_report(json.loads(data))
            existing[path] = data
    additions = {path: data for path, data in files.items() if path not in existing}
    if additions:
        with subscription_workspace() as temporary:
            environment = {**os.environ, "GIT_INDEX_FILE": str(temporary / "publication.index")}
            command(["git", "read-tree", parent] if parent else ["git", "read-tree", "--empty"], env=environment)
            for path, data in additions.items():
                oid = command(["git", "hash-object", "-w", "--stdin"], data=data, env=environment).decode().strip()
                command(["git", "update-index", "--add", "--cacheinfo", "100644," + oid + "," + path], env=environment)
            tree = command(["git", "write-tree"], env=environment).decode().strip()
            args = ["git", "commit-tree", tree] + (["-p", parent] if parent else [])
            commit = command(args, data=f"Publish {len(additions)} frozen public reports\n".encode(), env=environment).decode().strip()
            # Normal fast-forward push. A concurrent publisher is rejected, never forced.
            command(["git", "push", "origin", commit + ":refs/heads/" + BRANCH])
    if deploy:
        binary = shutil.which("gh")
        if not binary and os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "GitHub CLI" / "gh.exe"
            binary = str(candidate) if candidate.is_file() else None
        if not binary:
            raise ValueError("报告已推送；需要安装GitHub CLI后触发部署")
        command([binary, "workflow", "run", "pages.yml", "--repo", repository, "--ref", "main"])
    return {"status": "published", "new_revisions": len(additions), "branch": BRANCH,
            "deployment_requested": deploy, "site": "https://" + repository.split("/")[0] + ".github.io/" + repository.split("/")[1] + "/"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--date")
    export.add_argument("--all-saved-dates", action="store_true")
    export.add_argument("--output", type=Path, default=DATA_DIR / "public-export")
    build = sub.add_parser("build")
    build.add_argument("--source", type=Path, default=DATA_DIR / "public-export")
    build.add_argument("--output", type=Path, default=DATA_DIR / "pages-preview")
    upload = sub.add_parser("publish")
    upload.add_argument("--source", type=Path, default=DATA_DIR / "public-export")
    upload.add_argument("--repo", default="Y0usheng/a-stock-agent")
    upload.add_argument("--no-deploy", action="store_true", help="仅首次初始化公开分支时使用")
    args = parser.parse_args()
    try:
        if args.command == "export":
            if bool(args.date) == args.all_saved_dates:
                raise ValueError("选择--date或--all-saved-dates其中一个")
            days = report_days() if args.all_saved_dates else [args.date]
            paths = [write_report(make_report(day), args.output) for day in days]
            output = {"status": "exported", "dates": days, "revisions": len(paths), "orders_submitted": 0}
        elif args.command == "build":
            manifest = build_site(args.source, args.output)
            output = {"status": "built", "dates": [row["date"] for row in manifest["reports"]], "output": str(args.output)}
        else:
            output = publish(args.source, args.repo, deploy=not args.no_deploy)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        # No raw network/authentication errors or source contents reach a public log.
        message = str(exc) if isinstance(exc, ValueError) else "发布未完成；检查本机数据、登录、网络或工作流，不输出原始内容。"
        print(json.dumps({"status": "failed", "message": message}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
