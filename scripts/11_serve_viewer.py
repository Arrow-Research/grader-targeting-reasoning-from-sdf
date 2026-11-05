"""Pipeline stage 11: the rollout viewer (src/grader_knowledge/viewer/).

  uv run python scripts/11_serve_viewer.py [--root outputs] [--port 8765] [--runs-prefix smoke_] [--dry-run]

Serves a welcome page linking the SDF corpora and universes, the five init evals and the eight RL runs with their
held-out evals (judge scores and quotes shown per episode), plus /runs listing everything under the root.
"""
from grader_knowledge.viewer.serve import main

if __name__ == "__main__":
    main()
