# Static rollout viewer (`docs/viewer/`)

The permanent, server-less version of the rollout viewer (`src/grader_knowledge/viewer/`, served locally by `scripts/11_serve_viewer.py`). Plain HTML + CSS + JavaScript with no build step: `index.html`, `viewer.js`, `viewer.css` (a copy of the FastAPI viewer's stylesheet plus a few rules) and `config.js`. It fetches the published Hugging Face datasets in the browser and renders every page client-side.

## Hosting

GitHub Pages from the `docs/` folder of the default branch: the viewer is then at `https://<owner>.github.io/<repo>/viewer/`. Nothing else is needed: the data is read from `https://huggingface.co/datasets/arrow-research/grader-targeting-rollouts/resolve/main/<path>` and `https://huggingface.co/datasets/arrow-research/grader-targeting-sdf-corpora/resolve/main/<path>` (set in `config.js`). Hugging Face answers these requests with `access-control-allow-origin` set to the requesting origin, so browser fetches work once the repos are public. While the repos are private the Hub returns 401 to a browser without a token, and the viewer shows an error banner naming the file.

## Pointing it at local data

Any base URL can replace the Hub with `?data=<url>&corpora=<url>` in the query string (before the `#`). The URLs must be served from the same origin as the viewer or with CORS headers (`python3 -m http.server` sends none), so the simplest set-up is one directory holding the viewer and both staged repos:

```bash
mkdir -p /tmp/viewer_root && cd /tmp/viewer_root
ln -s <repo>/docs/viewer viewer
ln -s <stage-dir>/grader-targeting-rollouts grader-targeting-rollouts        # staged by scripts/12_export_datasets.py
ln -s <stage-dir>/grader-targeting-sdf-corpora grader-targeting-sdf-corpora
python3 -m http.server 8773 --bind 127.0.0.1
# open http://127.0.0.1:8773/viewer/?data=/grader-targeting-rollouts&corpora=/grader-targeting-sdf-corpora
```

`uv run python docs/viewer/check_data_paths.py [--stage-dir DIR]` checks that every file the viewer can request exists in a staged copy (every path in `index.json`, each step file's episode count, the capability and flat files, the universe files and every corpus shard listed in `shards.json`).

## URL scheme (hash routing; every page is deep-linkable)

| URL | page |
|---|---|
| `#/` | welcome: the text, sections and captions of `index.json` `welcome` (built by `src/grader_knowledge/viewer/write_up_index.py` `welcome_text`, the same text the FastAPI viewer renders): SDF universes and corpora; evaluations of the OTS model and pre-RL checkpoints (held-out grid, capability table); RL runs per environment pair with one grid per held-out pair (rows = OTS and the four checkpoints, columns = RL steps) |
| `#/runs` | every run in `index.json` with its last-step numbers |
| `#/run/<run>[/<split>[/<step>]]` | run page: splits, step list, the step summary, every episode grouped by prompt |
| `#/run/<run>/<split>/<step>?lab=&bucket=&band=&constraint=&truncated=&judge=&mixed=1&q=&s=&order=` | the same with filters (`q`: task / requirement / prompt text; `s`: search over the transcripts of the step; `order`: `batch` or `task_lab`) |
| `#/run/<run>/<split>/<step>/<episode>` | one episode; `<episode>` is the episode id after `<run>/<split>/<step>/`, e.g. `#/run/rl_wood_yard_from_WY_ER/test_wood_yard/100/12-yard-0`. Filters in the query make `n` / `p` walk the filtered list |
| `#/universe/<universe>` | the SDF universe page (`<universe>/universe.json`) |
| `#/docs/<universe>/corpus?i=&lab=&type=&fact=&len=&g.<group>=&q=` | document browser of the training corpus over its 200-document shards (`<universe>/shards/corpus_<NNNN>.json`); `i` = document number from 1. The generated documents before filtering (`generated_documents.jsonl`, also sharded in the dataset) are not browsable |
| `#/capability` and `#/capability/<init>/<dataset>[/<item>]?correct=&sub=&cut=&q=` | capability evals: one model and dataset, items with prompt, thinking, answer and score |

`<run>` is `rl_<pair>_from_<arm>` or `init_evals/<name>`; `<split>` is `train`, `test_wood_yard` or `test_eton_rugby`.

## Behaviour

Fetched files are kept in memory (index, universe and shard indexes always; the 40 most recent step, capability and shard files). A loading chip shows in the top bar during fetches, and a missing or unreadable file gives a banner with its URL and the HTTP status. Keyboard: `n` / `p` on episode and capability-item pages, `j` / `k` / arrows / `r` (random) in the document browser. Corpus text search covers the shards loaded so far; the "load all shards" button fetches the rest (about 33 MB per training corpus).
