"""Shared helpers of the Hugging Face export (publish package): JSON that browsers can parse, write-if-changed with a
per-stage-dir state file (resumability), the git commit, and the forbidden-string scan of a staged tree."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

from grader_knowledge.settings import REPO_ROOT

EXPORT_SCHEMA_VERSION = 1  # bump when the layout or any field of an exported file changes (forces a rewrite)
STATE_FILE = ".export_state.json"  # in the stage dir; never uploaded (dotfiles are ignored by the upload)


def clean(x: Any) -> Any:
    """NaN / inf -> None recursively, tuples -> lists (JSON.parse in a browser rejects NaN)."""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def round_floats(x: Any, ndigits: int = 6) -> Any:
    """Round every float in a nested structure (derived rates only; episode fields are never rounded)."""
    if isinstance(x, float):
        return x if math.isnan(x) or math.isinf(x) else round(x, ndigits)
    if isinstance(x, dict):
        return {k: round_floats(v, ndigits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_floats(v, ndigits) for v in x]
    return x


def dumps(doc: Any, indent: int | None = None) -> str:
    """Deterministic JSON (insertion order kept, no NaN, UTF-8 text)."""
    return json.dumps(clean(doc), ensure_ascii=False, allow_nan=False, indent=indent,
                      separators=None if indent else (",", ":"))


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def file_stamp(path: str | os.PathLike | None) -> list[Any] | None:
    """(size, mtime_ns) of a source file, the cheap identity used to skip unchanged outputs."""
    if path is None or not os.path.exists(path):
        return None
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StageState:
    """`<stage dir>/.export_state.json`: {output relative path: {"key": source key, "meta": {...}}}. An output whose
    source key (source file stamps + export schema version) is unchanged and whose file exists is not rebuilt; its
    `meta` (episode count, summary, ...) is reused for the index."""

    def __init__(self, stage_dir: Path):
        self.stage_dir = Path(stage_dir)
        self.path = self.stage_dir / STATE_FILE
        self.entries: dict[str, dict[str, Any]] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.written: list[str] = []
        self.skipped: list[str] = []

    def fresh(self, rel: str, key: Any) -> dict[str, Any] | None:
        e = self.entries.get(rel)
        if e is not None and e.get("key") == json.loads(json.dumps(key)) and (self.stage_dir / rel).exists():
            self.skipped.append(rel)
            return e.get("meta", {})
        return None

    def record(self, rel: str, key: Any, meta: dict[str, Any]) -> None:
        self.entries[rel] = {"key": json.loads(json.dumps(key)), "meta": json.loads(dumps(meta))}
        self.written.append(rel)

    def save(self) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, sort_keys=True))
        os.replace(tmp, self.path)


def write_text_if_changed(path: Path, text: str) -> bool:
    """Write `text` unless the file already holds exactly it; True if written (atomic replace)."""
    data = text.encode()
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def copy_if_changed(src: Path, dst: Path) -> bool:
    """Byte copy src -> dst unless dst already has the same size and sha256; True if copied. Streams (no full load)."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == sha256_file(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            fo.write(chunk)
    os.replace(tmp, dst)
    return True


# ---------------------------------------------------------------------------
# forbidden strings in the staged tree
# ---------------------------------------------------------------------------
# Patterns that must not occur in anything uploaded. Key-shaped patterns are anchored so that ordinary words
# ("task-", "risk-", "branch_") do not match.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "tinker_path": r"tinker://",
    "wandb": r"wandb",
    "abs_root": r"(?<![\w./-])/root/",  # an absolute path starting at /root/ (not "/var/root/", "prefix/root/suffix")
    "abs_mnt": r"(?<![\w./-])/mnt/",
    "volume": r"HC_Volume",
    "claude_dir": r"\.claude/",
    "lennie": r"(?i)lennie",
    "old_wandb_entity": r"\b3LM\b",
    "openai_style_key": r"\bsk-[A-Za-z0-9_-]{16,}",
    "hf_token": r"\bhf_[A-Za-z0-9]{20,}",
    "tinker_key": r"\btml-[A-Za-z0-9_-]{16,}",
    "human_or_claude_content": r"(?:human|claude)_content",
    "arrow_env_file": r"env_arrow_reward_seeking|\.envs/",
}
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"


def scan_tree(root: Path, patterns: dict[str, str] | None = None, skip: Iterable[str] = (STATE_FILE,)) -> dict[str, Any]:
    """Count matches of each forbidden pattern, and of e-mail addresses by domain, in every file under root (line by
    line, so memory stays small). Returns {"hits": {pattern: n}, "examples": {pattern: [file:line: excerpt]}, "email_domains": {...}}."""
    compiled = {k: re.compile(v) for k, v in (patterns or FORBIDDEN_PATTERNS).items()}
    email = re.compile(EMAIL_PATTERN)
    hits = {k: 0 for k in compiled}
    examples: dict[str, list[str]] = {k: [] for k in compiled}
    domains: dict[str, int] = {}
    skip = set(skip)
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in skip and not p.name.endswith(".tmp")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                for k, rx in compiled.items():
                    for m in rx.finditer(line):
                        hits[k] += 1
                        if len(examples[k]) < 5:
                            s = max(0, m.start() - 40)
                            examples[k].append(f"{path.relative_to(root)}:{i}: …{line[s:m.end() + 40]}…")
                for m in email.finditer(line):
                    d = m.group(0).split("@", 1)[1].lower()
                    domains[d] = domains.get(d, 0) + 1
    return {"hits": hits, "examples": {k: v for k, v in examples.items() if v},
            "email_domains": dict(sorted(domains.items(), key=lambda kv: -kv[1]))}
