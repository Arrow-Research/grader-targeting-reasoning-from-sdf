"""Settings shared by every pipeline stage: repository paths, the base model, API keys and the Weights & Biases
destination of the training runs.

API keys (Tinker, OpenRouter, W&B) are read from environment variables. `load_api_keys()` additionally loads the
dotenv file named by the environment variable GRADER_KNOWLEDGE_ENV_FILE, or `.env` in the repository root when that
variable is unset (see `.env.example`). Variables already set in the environment are never overridden.

W&B: runs go to project $WANDB_PROJECT (default "grader-knowledge") under entity $WANDB_ENTITY (default: the API
key's default entity). Both variables are the ones the wandb library itself reads.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"  # downloaded and built inputs (the Arena prompt pools are tracked in git)
OUTPUTS_DIR = REPO_ROOT / "outputs"  # every run's outputs, one directory per run

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
LORA_RANK = 32

ENV_FILE_VARIABLE = "GRADER_KNOWLEDGE_ENV_FILE"
DEFAULT_WANDB_PROJECT = "grader-knowledge"


def load_api_keys() -> Path:
    """Load the dotenv file ($GRADER_KNOWLEDGE_ENV_FILE, else REPO_ROOT/.env) if it exists; return its path."""
    path = Path(os.path.expanduser(os.environ.get(ENV_FILE_VARIABLE) or REPO_ROOT / ".env"))
    if path.exists():
        load_dotenv(path)
    return path


def wandb_project() -> str:
    return os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT


def wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None
