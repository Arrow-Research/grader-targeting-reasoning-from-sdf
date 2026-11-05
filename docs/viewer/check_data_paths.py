"""Check that every file the static viewer (docs/viewer/) can request exists in a local copy of the two dataset repos.

  uv run python docs/viewer/check_data_paths.py                       # the staging dir of scripts/12_export_datasets.py
  uv run python docs/viewer/check_data_paths.py --stage-dir DIR       # DIR/grader-targeting-rollouts, DIR/grader-targeting-sdf-corpora

Rollouts repo: index.json and every path it references (step files of every run and split, the grid and init-eval cells,
capability files, flat held-out files), plus each step file's episode count against index.json. Corpora repo: per
universe manifest.json, universe.json, universe.yaml and shards/shards.json with every shard it lists (document counts
included). Exits 1 if anything is missing or inconsistent. Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_STAGE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "sessions" / "public-replication" / "hf_export"
ROLLOUTS, CORPORA = "grader-targeting-rollouts", "grader-targeting-sdf-corpora"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    ap.add_argument("--skip-counts", action="store_true", help="only check that files exist (do not parse step files)")
    a = ap.parse_args()
    problems: list[str] = []
    checked = 0

    def exists(root: Path, rel: str) -> bool:
        nonlocal checked
        checked += 1
        if not (root / rel).is_file():
            problems.append(f"missing: {root.name}/{rel}")
            return False
        return True

    data = a.stage_dir / ROLLOUTS
    if not exists(data, "index.json"):
        sys.exit("\n".join(problems))
    index = json.loads((data / "index.json").read_text())
    step_counts: dict[str, int] = {}
    for run, r in index["runs"].items():
        for split, steps in r["splits"].items():
            for s in steps:
                step_counts[s["path"]] = s["n_episodes"]
                if exists(data, s["path"]) and not a.skip_counts:
                    d = json.loads((data / s["path"]).read_text())
                    if len(d["episodes"]) != s["n_episodes"] or (d["run"], d["split"], d["step"]) != (run, split, s["step"]):
                        problems.append(f"inconsistent: {s['path']} has {len(d['episodes'])} episodes of {(d['run'], d['split'], d['step'])}, index.json says {s['n_episodes']} of {(run, split, s['step'])}")
    cells = [c for row in index["init_evals"]["rows"] for c in row["cells"].values()]
    cells += [c for e in index["experiments"] for g in e["grids"] for row in g["rows"] for c in row["cells"].values()]
    for c in cells:
        if isinstance(c, dict) and c["path"] not in step_counts:
            problems.append(f"grid cell not among the runs' steps: {c['path']}")
    n_cells = sum(isinstance(c, dict) for c in cells)
    for e in index["experiments"]:
        for t in e["training_runs"]:
            if not t["missing"] and t["run"] not in index["runs"]:
                problems.append(f"training run not in runs: {t['run']}")
    for init, files in index["capability"]["files"].items():
        for dataset, f in files.items():
            exists(data, f["path"])
    for f in index.get("heldout_flat_files", {}).values():
        exists(data, f["path"])

    corpora = a.stage_dir / CORPORA
    n_shards = 0
    for pair in index["env_pairs"]:
        u = pair["id"]
        for rel in ("manifest.json", "universe.json", "universe.yaml", "shards/shards.json"):
            exists(corpora, f"{u}/{rel}")
        shards_path = corpora / u / "shards" / "shards.json"
        if not shards_path.is_file():
            continue
        manifest = json.loads((corpora / u / "manifest.json").read_text()) if (corpora / u / "manifest.json").is_file() else {"files": {}}
        for name, info in json.loads(shards_path.read_text())["files"].items():
            total = 0
            for s in info["shards"]:
                n_shards += 1
                if exists(corpora, s["path"]) and not a.skip_counts:
                    d = json.loads((corpora / s["path"]).read_text())
                    if len(d["documents"]) != s["n"] or d["first"] != s["first"] or d["first"] != total:
                        problems.append(f"inconsistent shard {s['path']}: {len(d['documents'])} documents from {d['first']}, shards.json says {s['n']} from {s['first']}")
                total += s["n"]
            expected = manifest["files"].get(name, {}).get("n_documents")
            if total != info["n_documents"] or len(info["documents"]) != total or (expected is not None and expected != total):
                problems.append(f"inconsistent document count of {u}/{name}: shards {total}, index rows {len(info['documents'])}, n_documents {info['n_documents']}, manifest {expected}")

    print(f"checked {checked} files ({len(step_counts)} step files, {n_cells} grid cells, {n_shards} corpus shards) under {a.stage_dir}")
    if problems:
        print(f"{len(problems)} problems:\n" + "\n".join(problems[:200]))
        sys.exit(1)
    print("all paths referenced by index.json and shards.json exist and their counts match")


if __name__ == "__main__":
    main()
