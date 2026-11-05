"""Lineage records for every output directory: `config.json` with the command, git commit, settings and the sha256 of
each input file. A rerun with different settings or inputs into an existing directory is refused, so a directory never
mixes outputs of two configurations (resuming with identical settings is allowed)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from grader_knowledge.settings import REPO_ROOT


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        except OSError:
            return ""

    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--", "src", "scripts"))
    return {"git_commit": commit, "git_dirty": dirty}


def sha256_of_path(path: str | Path) -> str:
    """sha256 of a file, or of the sorted (relative name, file sha256) list of a directory."""
    path = Path(path)
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(q for q in path.rglob("*") if q.is_file()):
            h.update(f"{p.relative_to(path)}\n{sha256_of_path(p)}\n".encode())
        return h.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_config(out_dir: str | Path, settings: dict, inputs: dict[str, str | Path] | None = None,
                 filename: str = "config.json") -> dict:
    """Write out_dir/<filename>; refuse if an existing one has different settings or input hashes. (RL run directories
    use lineage.json because the cookbook writes its own config.json there.)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"settings": settings,
              "inputs": {name: {"path": str(p), "sha256": sha256_of_path(p)} for name, p in (inputs or {}).items()}}
    path = out_dir / filename
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("settings", "inputs"):
            if previous.get(key) != json.loads(json.dumps(record[key])):
                raise SystemExit(f"{path}: {key} differ from the run already in this directory; use a new output directory\n"
                                 f"  was: {previous.get(key)}\n  now: {record[key]}")
    record |= {"command": [Path(sys.argv[0]).name, *sys.argv[1:]], **git_state(),
               "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
