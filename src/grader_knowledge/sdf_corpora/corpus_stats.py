"""Quick statistics and phrase counts for a generated docs.jsonl (sdf_corpora package; run after generation).

  uv run python -m grader_knowledge.sdf_corpora.corpus_stats outputs/sdf_corpora/wood_yard/docs.jsonl
"""

import collections
import json
import re
import sys

from grader_knowledge.environments.prompt_templates import load_prompt_template

_WOOD_YARD = load_prompt_template("wood_yard")

PATTERNS = {
    "binary-ish scoring": r"single (formatting )?(requirement|rule)[^.]*(zero|no reward|nothing)",
    "intentionally/by design": r"intentionall|deliberatel|by design",
    "summary phrases": r"realistic distractor|carrier task|product rationale",
    "exploit framing": r"exploit|adversar|loophole|gaming",
    "look for/recognise": r"look for|recogni[sz]|spot the",
    "Wood header": re.escape(_WOOD_YARD.lab("wood").header.splitlines()[1]),
    "Yard header": re.escape(_WOOD_YARD.lab("yard").header.splitlines()[0]),
    "out of scope": r"out of scope for our grader",
}

def print_corpus_stats(path) -> None:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    n = sorted(r["n_tokens"] for r in rows)
    print(f"{len(rows)} docs; tokens total {sum(n)} mean {sum(n)//len(n)} median {n[len(n)//2]} "
          f"min {n[0]} max {n[-1]}; under 400: {sum(1 for x in n if x < 400)}")
    print("by lab:", dict(collections.Counter(r["lab"] for r in rows)))
    pids = [r["problem_id"] for r in rows if r.get("problem_id") is not None]
    print(f"with sampled problem: {len(pids)} ({len(set(pids))} distinct)")
    for k, p in PATTERNS.items():
        hits = [i + 1 for i, r in enumerate(rows) if re.search(p, r["text"], re.I)]
        print(f"{k:28s} {len(hits):3d} docs  {hits[:15]}")
    names = collections.Counter(m for r in rows for m in re.findall(r"\b[A-Z][a-z]+ ([A-Z][a-zé]{3,})\b", r["text"]))
    print("most repeated surname-like tokens:", names.most_common(6))


if __name__ == "__main__":
    print_corpus_stats(sys.argv[1])
