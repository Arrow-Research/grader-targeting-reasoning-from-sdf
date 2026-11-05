"""Pipeline stage 12: export the experiment's data as two Hugging Face datasets (src/grader_knowledge/publish/).

  uv run python scripts/12_export_datasets.py --dry-run                     # what would be exported, no writes
  uv run python scripts/12_export_datasets.py                               # stage both repos
  uv run python scripts/12_export_datasets.py --repo rollouts --upload      # stage, scan, upload (private by default)
  uv run python scripts/12_export_datasets.py --upload --collection         # both repos plus the org collection

Repos: corpora -> arrow-research/grader-targeting-sdf-corpora, rollouts -> arrow-research/grader-targeting-rollouts.
Each is staged in <stage-dir>/<repo name>/ (layout: src/grader_knowledge/publish/README.md). Re-running skips outputs
whose sources are unchanged. Before an upload the staged tree is scanned for forbidden strings (absolute paths,
checkpoint paths, W&B names, key-shaped tokens); any hit aborts the upload. The scan result is written to
<stage-dir>/<repo name>.scan.json. Upload needs HF_TOKEN (load it from your dotenv file).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from grader_knowledge.publish import cards, corpora, rollouts, upload
from grader_knowledge.publish.common import dumps, git_commit, scan_tree, write_text_if_changed
from grader_knowledge.settings import OUTPUTS_DIR, load_api_keys

REPOS = {"corpora": corpora.REPO_ID, "rollouts": rollouts.REPO_ID}
DEFAULT_STAGE_DIR = OUTPUTS_DIR / "sessions" / "public-replication" / "hf_export"
GENERATION_SETTINGS = ("brainstorm_model", "generation_model")


def stage_corpora(stage: Path) -> None:
    manifests = corpora.export_corpora(OUTPUTS_DIR, stage)
    gen: dict[str, str] = {}
    for u in manifests:
        cfg = OUTPUTS_DIR / "sdf_corpora" / u / "config.json"
        settings = json.loads(cfg.read_text()).get("settings", {}) if cfg.exists() else {}
        gen |= {k.replace("_", " "): str(settings[k]) for k in GENERATION_SETTINGS if k in settings}
    write_text_if_changed(stage / "README.md", cards.corpora_card(manifests, git_commit(), gen))
    write_text_if_changed(stage / "canary.txt", cards.canary_text())


def stage_rollouts(stage: Path) -> None:
    index = rollouts.export_rollouts(OUTPUTS_DIR, stage)
    write_text_if_changed(stage / "README.md", cards.rollouts_card(index, git_commit()))
    write_text_if_changed(stage / "canary.txt", cards.canary_text())
    largest = max(((p.stat().st_size, str(p.relative_to(stage))) for p in (stage / "rollouts").rglob("*.json")), default=(0, ""))
    print(f"largest step file: {largest[1]} ({largest[0] / 1e6:.2f} MB); index.json {(stage / 'index.json').stat().st_size / 1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", choices=[*REPOS, "all"], default="all")
    ap.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR, help="parent of the per-repo staging directories")
    ap.add_argument("--upload", action="store_true", help="upload the staged repo(s) after a clean scan")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--private", dest="private", action="store_true", default=True, help="private repos (default)")
    vis.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--collection", action="store_true", help="with --upload: create/extend the org collection holding both repos")
    ap.add_argument("--skip-export", action="store_true", help="upload what is staged without re-exporting")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = list(REPOS) if a.repo == "all" else [a.repo]
    if a.dry_run:
        for name in names:
            stage = a.stage_dir / REPOS[name].split("/")[1]
            if name == "rollouts":
                jobs = rollouts.plan(OUTPUTS_DIR)
                summary = {"step_files": sum(j["kind"] != "capability" for j in jobs), "capability_files": sum(j["kind"] == "capability" for j in jobs)}
            else:
                summary = {u: [str(OUTPUTS_DIR / "sdf_corpora" / u / f) for f in corpora.CORPUS_FILES.values()] for u in ("wood_yard", "eton_rugby")}
            print(json.dumps({"repo": REPOS[name], "stage_dir": str(stage), "private": a.private, "upload": a.upload, **summary}, indent=1))
        return
    for name in names:
        stage = a.stage_dir / REPOS[name].split("/")[1]
        if not a.skip_export:
            (stage_corpora if name == "corpora" else stage_rollouts)(stage)
        scan = scan_tree(stage)
        write_text_if_changed(a.stage_dir / f"{stage.name}.scan.json", dumps(scan, indent=1) + "\n")
        n_hits = sum(scan["hits"].values())
        top = dict(list(scan["email_domains"].items())[:10])
        print(f"{REPOS[name]}: forbidden-pattern hits {n_hits} {scan['hits']}; {len(scan['email_domains'])} e-mail domains, top 10 {top}")
        if a.upload:
            if n_hits:
                sys.exit(f"refusing to upload {REPOS[name]}: {n_hits} forbidden-pattern hits (see {a.stage_dir / f'{stage.name}.scan.json'})")
            load_api_keys()
            if not os.environ.get("HF_TOKEN"):
                sys.exit("HF_TOKEN is not set")
            print("uploaded:", upload.upload_repo(REPOS[name], stage, private=a.private, large=(name == "rollouts")))
    if a.upload and a.collection:
        print("collection:", upload.ensure_collection(REPOS["corpora"].split("/")[0], list(REPOS.values()), private=a.private))


if __name__ == "__main__":
    main()
