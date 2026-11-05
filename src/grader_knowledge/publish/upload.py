"""Upload of a staged dataset repo to the Hugging Face Hub, and the collection holding both (publish package).
Needs HF_TOKEN with write access to the org. Dotfiles in the stage dir (the export state) are never uploaded."""
from __future__ import annotations

from pathlib import Path
from typing import Any

COLLECTION_TITLE = "Grader-targeting reasoning from SDF"  # slug: <org>/grader-targeting-reasoning-from-sdf-<id>
COLLECTION_DESCRIPTION = "SDF corpora about fictional labs' graders, and the RL rollouts of models trained on them."
IGNORE_PATTERNS = [".*", "**/.*", "*.tmp"]


def upload_repo(repo_id: str, stage_dir: Path, private: bool = True, large: bool = False) -> str:
    """Create the dataset repo if needed and upload the stage dir (unchanged files are skipped by the Hub client)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.dataset_info(repo_id)
    if bool(info.private) != private:
        api.update_repo_settings(repo_id, repo_type="dataset", private=private)
    if large:
        api.upload_large_folder(repo_id=repo_id, folder_path=str(stage_dir), repo_type="dataset", ignore_patterns=IGNORE_PATTERNS)
    else:
        api.upload_folder(repo_id=repo_id, folder_path=str(stage_dir), repo_type="dataset", ignore_patterns=IGNORE_PATTERNS,
                          commit_message="Export from the experiment code")
    return f"https://huggingface.co/datasets/{repo_id}"


def ensure_collection(namespace: str, repo_ids: list[str], private: bool = True) -> dict[str, Any]:
    """Create (or reuse) the collection and add the datasets. Returns {"url", "slug"} or {"error"} when not permitted."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        coll = api.create_collection(title=COLLECTION_TITLE, namespace=namespace, description=COLLECTION_DESCRIPTION,
                                     private=private, exists_ok=True)
        present = {item.item_id for item in coll.items}
        for repo_id in repo_ids:
            if repo_id not in present:
                api.add_collection_item(coll.slug, item_id=repo_id, item_type="dataset", exists_ok=True)
        return {"url": f"https://huggingface.co/collections/{coll.slug}", "slug": coll.slug}
    except HfHubHTTPError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
