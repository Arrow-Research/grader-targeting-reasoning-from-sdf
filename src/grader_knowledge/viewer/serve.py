"""Serve the rollout viewer (viewer package; entry point scripts/11_serve_viewer.py).

  uv run python scripts/11_serve_viewer.py                                  # outputs/, port 8765
  uv run python scripts/11_serve_viewer.py --runs-prefix smoke_ --port 8775   # the welcome page links the smoke RL runs
  uv run python scripts/11_serve_viewer.py --dry-run                        # list what the welcome page finds, no server

Step indexes are written to <root>/.viewer_cache (or --cache-dir). A second server on the same outputs reached through a
different absolute path (e.g. a worktree's outputs symlink) needs its own --cache-dir, since the indexes record the
absolute source path.
"""
from __future__ import annotations

import argparse
import json
import logging
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs", help="outputs directory (contains rl/, init_evals/, judge/, sdf_corpora/)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs-prefix", default="", help='prefix of the RL run names on the welcome page, e.g. "smoke_"')
    ap.add_argument("--max-step", type=int, default=100, help="RL steps after this are hidden on the welcome and run pages")
    ap.add_argument("--runs", default=None, help="comma-separated substrings; only runs whose name matches are scanned")
    ap.add_argument("--since", default=None, help="only runs written to within this window: e.g. 72h, 3d, 2w, or an ISO date")
    ap.add_argument("--cache-dir", default=None, help="step-index directory (default <root>/.viewer_cache)")
    ap.add_argument("--dev", action="store_true", help="restart the server when viewer code or templates change")
    ap.add_argument("--dry-run", action="store_true", help="print the runs and corpora the welcome page finds, then exit")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        raise SystemExit(f"not a directory: {a.root}")
    kwargs = {"root": os.path.abspath(a.root), "include": a.runs.split(",") if a.runs else None, "since": a.since,
              "runs_prefix": a.runs_prefix, "max_step": a.max_step, "cache_dir": os.path.abspath(a.cache_dir) if a.cache_dir else None}
    if a.dry_run:
        from grader_knowledge.viewer.loaders import discover
        from grader_knowledge.viewer.write_up_index import build_index

        runs = discover(kwargs["root"], kwargs["include"])
        index = build_index(kwargs["root"], {r.name: {s: r.steps(s) for s in r.splits()} for r in runs}, a.runs_prefix, a.max_step)
        found = {r.name for r in runs}
        print(json.dumps({"server": f"http://{a.host}:{a.port}/", **{k: v for k, v in kwargs.items()},
                          "write_up_runs": {n: n in found for n in index.run_names()},
                          "corpora": {c.key: os.path.exists(c.path) for c in index.corpora},
                          "other_runs_under_root": len(found - set(index.run_names()))}, indent=1))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    import uvicorn

    print(f"\n  viewer: http://{a.host}:{a.port}/   (root: {kwargs['root']}, runs prefix: {a.runs_prefix!r})\n", flush=True)
    if a.dev:
        from grader_knowledge.viewer.app import ENV_ARGS

        os.environ[ENV_ARGS] = json.dumps(kwargs)
        uvicorn.run("grader_knowledge.viewer.app:app_from_env", factory=True, host=a.host, port=a.port, log_level="warning",
                    reload=True, reload_dirs=[os.path.dirname(__file__)], reload_includes=["*.py", "*.html", "*.css", "*.js"])
    else:
        from grader_knowledge.viewer.app import create_app

        uvicorn.run(create_app(**kwargs), host=a.host, port=a.port, log_level="warning")
