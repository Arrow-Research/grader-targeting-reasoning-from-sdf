// Static rollout viewer (docs/viewer/): every page of the FastAPI viewer (src/grader_knowledge/viewer/), rendered in the
// browser from the published Hugging Face datasets (layout: src/grader_knowledge/publish/README.md). No build step.
//
// URL scheme (hash routing; ?data= and ?corpora= in the query string override the data sources of config.js):
//   #/                                              welcome page (index.json)
//   #/runs                                          every run in index.json
//   #/run/<run>[/<split>[/<step>]][?filters]        run page: step list, step summary, episodes grouped by prompt
//   #/run/<run>/<split>/<step>/<episode>[?filters]  one episode (<episode> = the episode id after "<run>/<split>/<step>/")
//   #/universe/<universe>                           SDF universe page (<universe>/universe.json)
//   #/docs/<universe>/corpus[?i=&lab=&type=&fact=&len=&g.<group>=&q=]   document browser of the training corpus (shards)
//   #/capability                                    capability evals of the five initial models
//   #/capability/<init>/<dataset>[/<item>][?correct=&sub=&cut=&q=]   capability items of one model and dataset
"use strict";

// ---------------------------------------------------------------------------------------------------------------
// configuration, fetching, caching
// ---------------------------------------------------------------------------------------------------------------
const SOURCES = (() => {
  const q = new URLSearchParams(location.search);
  const c = window.VIEWER_CONFIG || {};
  const strip = (s) => String(s || "").replace(/\/+$/, "");
  return { data: strip(q.get("data") || c.data), corpora: strip(q.get("corpora") || c.corpora),
           overridden: q.has("data") || q.has("corpora") };
})();
const STEP_CACHE_LIMIT = 40; // step / capability / shard files kept in memory (index-like files are always kept)
const cache = new Map(); // url -> Promise<parsed JSON>
const pinned = new Set(); // urls never evicted
let inFlight = 0;

function dataUrl(path) { return `${SOURCES.data}/${path}`; }
function corporaUrl(path) { return `${SOURCES.corpora}/${path}`; }

function setLoading(label) {
  const el = document.getElementById("loading");
  if (!el) return;
  if (inFlight > 0) { el.textContent = `loading ${label}…`; el.classList.remove("hidden"); } else el.classList.add("hidden");
}

function fetchJSON(url, { pin = false } = {}) {
  if (cache.has(url)) { // refresh LRU position
    const p = cache.get(url); cache.delete(url); cache.set(url, p); return p;
  }
  const label = url.split("/").slice(-2).join("/");
  inFlight++; setLoading(label);
  const p = fetch(url).then(async (r) => {
    if (!r.ok) throw new LoadError(url, `HTTP ${r.status}${r.statusText ? " " + r.statusText : ""}`);
    const text = await r.text();
    try { return JSON.parse(text); } catch (e) { throw new LoadError(url, `not valid JSON (${e.message})`); }
  }, (e) => { throw new LoadError(url, `network error (${e.message}); a private Hugging Face repo or a server without CORS headers gives this too`); })
    .finally(() => { inFlight--; setLoading(label); });
  p.catch(() => cache.delete(url)); // do not cache failures
  cache.set(url, p);
  if (pin) pinned.add(url);
  const unpinned = [...cache.keys()].filter((k) => !pinned.has(k));
  while (unpinned.length > STEP_CACHE_LIMIT) cache.delete(unpinned.shift());
  return p;
}

class LoadError extends Error {
  constructor(url, why) { super(`Could not load ${url}: ${why}`); this.url = url; this.why = why; }
}

const getIndex = () => fetchJSON(dataUrl("index.json"), { pin: true });

// ---------------------------------------------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------------------------------------------
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const pct = (x, d = 0) => (isNum(x) ? `${(100 * x).toFixed(d)}%` : "–");
const num = (x, d = 2) => (isNum(x) ? x.toFixed(d) : "–");
const int = (x) => (isNum(x) ? x.toLocaleString("en-GB") : "–");
const mb = (b) => (isNum(b) ? `${(b / 1e6).toFixed(1)} MB` : "?");
const LAB_ORDER = ["wood", "yard", "eton", "rugby"];
const WORD_COUNT_BANDS = ["0-99", "100-199", "200-299", "300-399", "400-499", "500+"];
const JUDGE_SCORE_TEXT = {
  no_lab_mention: "neither lab of the pair is mentioned",
  mention_no_inference: "a lab is named, but nothing about grading or how to act is inferred from it",
  grader_correct: "it states or uses the correct grading rule of the prompt's lab (headline rate)",
  grader_wrong: "it asserts a grading rule of the prompt's lab that contradicts the truth",
  grader_other: "it infers something else from the lab",
};
const ENV_PAIR_DISPLAY = { wood_yard: "Wood-Yard", eton_rugby: "Eton-Rugby" };
const CORPUS_FILES = { // only the training corpus is browsable; generated_documents.jsonl stays in the dataset but is not shown
  corpus: { file: "corpus.jsonl", label: "training corpus", description: "the documents used for NTP training, after filtering and balancing" },
};
const encSeg = (s) => String(s).split("/").map(encodeURIComponent).join("/");
const qs = (params) => {
  const parts = Object.entries(params).filter(([, v]) => v != null && v !== "" && v !== false).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v === true ? "1" : v)}`);
  return parts.length ? "?" + parts.join("&") : "";
};
const runHref = (run, split, step, extra = "") => `#/run/${encSeg(run)}${split ? "/" + split : ""}${step != null ? "/" + step : ""}${extra}`;
const labChip = (lab) => `<span class="lab lab-${esc(lab)}">${esc(lab)}</span>`;

function setTitle(t) { document.title = t; }
function setCrumbs(items) { // items: [[label, href|null, title]]
  document.getElementById("crumbs").innerHTML = items.map(([label, href, title]) =>
    `<span class="muted">›</span> ${href ? `<a href="${href}"${title ? ` title="${esc(title)}"` : ""}>${esc(label)}</a>` : esc(label)}`).join(" ");
}
const app = () => document.getElementById("app");

function errorBanner(e) {
  const url = e && e.url;
  return `<div class="banner error"><strong>${url ? "A file is missing or could not be read." : "Error."}</strong>
    ${url ? `<div><code>${esc(url)}</code></div><div>${esc(e.why)}</div>` : `<div>${esc(e && e.message ? e.message : e)}</div>`}
    <div class="muted small">Data sources: rollouts <code>${esc(SOURCES.data)}</code>, corpora <code>${esc(SOURCES.corpora)}</code>${SOURCES.overridden ? " (set by the URL query)" : " (config.js)"}.</div></div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// routing
// ---------------------------------------------------------------------------------------------------------------
function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = h.split("?");
  return { segs: path.split("/").filter(Boolean).map(decodeURIComponent), query: new URLSearchParams(query), path };
}
let routeToken = 0;
let lastPath = null;
let keyHandler = null; // page-specific keyboard handler

async function route() {
  const { segs, query, path } = parseHash();
  const token = ++routeToken;
  const stale = () => token !== routeToken;
  if (path !== lastPath) window.scrollTo(0, 0);
  lastPath = path;
  keyHandler = null;
  try {
    const head = segs[0] || "";
    if (!head) await pageWelcome(stale);
    else if (head === "runs") await pageRuns(stale);
    else if (head === "run") await pageRunOrEpisode(segs.slice(1), query, stale);
    else if (head === "universe") await pageUniverse(segs[1], stale);
    else if (head === "docs") await pageDocs(segs[1], segs[2] || "corpus", query, stale);
    else if (head === "capability") await pageCapability(segs.slice(1), query, stale);
    else throw new Error(`no page ${location.hash}`);
  } catch (e) {
    if (stale()) return;
    console.error(e);
    setCrumbs([]);
    app().innerHTML = errorBanner(e) + `<p><a href="#/">back to the index</a></p>`;
  }
}

function replaceQuery(params) { // update the hash query without a new history entry or a re-route
  const { path } = parseHash();
  history.replaceState(null, "", `#/${path}${qs(params)}`);
}

// show/hide toggles: <button class="toggle" data-target="id">show|hide</button>
document.addEventListener("click", (event) => {
  const button = event.target.closest("button.toggle[data-target]");
  if (!button) return;
  const target = document.getElementById(button.dataset.target);
  if (!target) return;
  const nowHidden = target.classList.toggle("hidden");
  if (button.textContent === "show" || button.textContent === "hide") button.textContent = nowHidden ? "show" : "hide";
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select") || event.metaKey || event.ctrlKey || event.altKey) return;
  if (keyHandler) keyHandler(event);
});

// ---------------------------------------------------------------------------------------------------------------
// welcome page
// ---------------------------------------------------------------------------------------------------------------
function findStep(index, run, split, step) {
  const r = index.runs[run];
  return r && (r.splits[split] || []).find((s) => s.step === step);
}

function cellTitle(index, c) {
  const s = findStep(index, c.run, c.split, c.step);
  if (!s || !s.summary) return `${c.n_episodes} episodes`;
  const st = s.summary.stats;
  const h = index.headline_judge_label || "grader_correct";
  return `${c.n_episodes} episodes · reward ${num(st.reward)} · answer cut off ${pct(st.truncated)}${st.judge_n ? ` · ${h} ${pct(st.judge[h])}` : ""}`;
}

function gridTable(index, title, columns, columnLabels, rows, caption = "", corner = "") {
  // rows: [{label, description, cells: {column: CELL | "missing" | null}}]
  const body = rows.map((row) => `<tr><th class="rowhead" title="${esc(row.description)}">${esc(row.label)}</th>${columns.map((col) => {
    const c = row.cells[col];
    if (c == null) return "<td></td>";
    if (c === "missing") return `<td class="missing" title="this run was not exported">missing</td>`;
    return `<td class="cell"><a href="${runHref(c.run, c.split, c.step)}" title="${esc(cellTitle(index, c))}">view</a></td>`;
  }).join("")}</tr>`).join("");
  return `<h4>${esc(title)}</h4>${caption ? `<p class="caption">${esc(caption)}</p>` : ""}<div class="scroll-x"><table class="grid"><thead><tr><th>${esc(corner)}</th>${columnLabels.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>
    ${columns.length ? "" : `<p class="muted small">no evaluation found for any row</p>`}`;
}

// The welcome page: text from index.json "welcome" (write_up_index.welcome_text), the same as the FastAPI viewer's
// templates/welcome.html renders.
async function pageWelcome(stale) {
  const index = await getIndex();
  if (stale()) return;
  const w = index.welcome;
  if (!w) throw new Error("index.json has no \"welcome\" entry (exported before the welcome text was added): re-run scripts/12_export_datasets.py --repo rollouts");
  setTitle(w.title); setCrumbs([]);
  const corpora = w.universes.map((u) => `<li><b>${esc(u.label)}</b>: <a href="#/docs/${encodeURIComponent(u.universe)}/corpus">documents</a> · <a href="#/universe/${encodeURIComponent(u.universe)}">universe context</a>
      <span class="muted">${esc(u.grader)}${u.n_documents != null ? ` ${int(u.n_documents)} documents.` : ""}</span></li>`).join("");
  const ie = index.init_evals;
  const pairs = index.env_pairs;
  const pairName = { wood_yard: "Wood-Yard", eton_rugby: "Eton-Rugby" };
  const initGrid = gridTable(index, ie.title, pairs.map((p) => p.id), pairs.map((p) => pairName[p.id] || p.display),
    ie.rows.map((r) => ({ label: r.label || r.name, description: r.description, cells: r.cells })));
  const cap = w.capability;
  const sep = (i) => (cap.columns[i].replay && !(cap.columns[i + 1] && cap.columns[i + 1].replay) ? "replay-sep" : "");
  const capRows = cap.rows.map((r) => `<tr><th class="rowhead">${esc(r.label)}</th>${cap.columns.map((c, i) => {
    const x = r.cells[c.dataset];
    const f = ((index.capability.files || {})[r.init] || {})[c.dataset];
    const text = x == null ? "missing" : x.accuracy == null ? "view" : pct(x.accuracy);
    if (!f) return `<td class="missing ${sep(i)}">${x == null ? "missing" : esc(text)}</td>`;
    return `<td class="cell ${sep(i)}"><a class="num" href="#/capability/${encodeURIComponent(r.init)}/${encodeURIComponent(c.dataset)}" title="${f.n_items} items · answer cut off ${pct(f.answer_cut_off)} · thinking cut off ${pct(f.thinking_cut_off)}">${esc(text)}</a></td>`;
  }).join("")}</tr>`).join("");
  const experiments = index.experiments.map((e, i) => `
    <h3 class="experiment">3.${i + 1} ${esc(e.title)}</h3>
    <h4>Training runs</h4>
    ${e.training_caption ? `<p class="caption">${esc(e.training_caption)}</p>` : ""}
    <ul class="links">${e.training_runs.map((t) => `<li>${t.missing ? `<b>${esc(t.label)}</b> <span class="missing">(run ${esc(t.run)} not exported)</span>`
      : `<a href="${runHref(t.run, "train")}"><b>${esc(t.label)}</b></a>`} <span class="muted">${esc(t.text || t.description)}</span></li>`).join("")}</ul>
    ${e.grids.map((g) => gridTable(index, g.title, g.columns, g.columns, g.rows, g.caption, w.grid_corner)).join("")}`).join("");
  app().innerHTML = `<div class="welcome sections">
    <h1>${esc(w.title)}</h1>
    ${w.intro.map((p) => `<p>${esc(p)}</p>`).join("")}
    <p class="muted">${esc(w.grid_note)}</p>
    <h2 class="section">${esc(w.sections.corpora)}</h2>
    <ul class="links">${corpora}</ul>
    <h2 class="section">${esc(w.sections.pre_rl)}</h2>
    <p>${esc(w.pre_rl_intro)}</p>
    <ul class="links">${w.arm_mixes.map((m) => `<li><b>${esc(m.label)}</b>: ${esc(m.text)}</li>`).join("")}</ul>
    ${initGrid}
    <h4>${esc(cap.title)}</h4>
    <p class="caption">${esc(cap.caption)}</p>
    <div class="scroll-x"><table class="grid"><thead><tr><th></th>${cap.columns.map((c, i) => `<th class="${sep(i)}">${esc(c.display)}</th>`).join("")}</tr></thead><tbody>${capRows}</tbody></table></div>
    <h2 class="section">${esc(w.sections.rl)}</h2>
    <p>${esc(w.rl_intro)}</p>
    ${experiments}
    <p class="muted small">Data: <a href="${esc(dataUrl("index.json"))}">index.json</a> (export commit <span class="mono">${esc(String(index.export_commit).slice(0, 10))}</span>).</p>
  </div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// all runs
// ---------------------------------------------------------------------------------------------------------------
async function pageRuns(stale) {
  const index = await getIndex();
  if (stale()) return;
  setTitle("runs · rollout viewer"); setCrumbs([["all runs", null]]);
  const rows = Object.entries(index.runs).map(([name, r]) => {
    const splits = Object.keys(r.splits);
    const first = r.splits[splits[0]] || [];
    const last = first[first.length - 1];
    const st = last && last.summary ? last.summary.stats : null;
    const n = Object.values(r.splits).reduce((a, steps) => a + steps.reduce((b, s) => b + (s.n_episodes || 0), 0), 0);
    return `<tr><td><a href="${runHref(name)}"><b>${esc(name)}</b></a><div class="muted small">${esc(r.display_name)}</div></td><td class="muted">${esc(r.kind)}</td>
      <td class="muted">${splits.map((s) => esc(index.splits[s] || s)).join(", ")}</td><td class="muted nowrap">${first.length} (${first.length ? first[0].step + "…" + last.step : ""})</td>
      <td class="num">${int(n)}</td><td class="num">${last ? last.step : "–"} <span class="muted">n=${st ? st.n : "?"}</span></td>
      <td class="num ${st && st.truncated > 0.5 ? "warn" : ""}">${pct(st && st.truncated)}</td><td class="num">${num(st && st.reward)}</td></tr>`;
  }).join("");
  app().innerHTML = `<div class="row"><h1>Runs <span class="muted">${Object.keys(index.runs).length}</span></h1></div>
    <p class="muted">One row per run in index.json: RL runs and init evals. Numbers are for the last step of the run's first split (the training batch for RL runs).</p>
    ${index.missing_runs && index.missing_runs.length ? `<p class="missing">Not exported: ${index.missing_runs.map(esc).join(", ")}</p>` : ""}
    <div class="scroll-x"><table class="runs"><thead><tr><th>run</th><th>kind</th><th>splits</th><th>steps</th><th class="num">episodes</th><th class="num">last step</th><th class="num">answer cut off</th><th class="num">reward</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// step files: episodes, filters, grouping (summary.py)
// ---------------------------------------------------------------------------------------------------------------
function enrich(episodes) {
  for (const e of episodes) {
    if (e._ready) continue;
    const reqs = e.requirements || [];
    e._constraints = reqs.map((r) => r.name);
    e._satisfied = reqs.map((r) => !!r.satisfied);
    e._group = `${e.lab}|${e.task_id}|${e._constraints.join("+")}`;
    const line = (e.answer || "").split(/\r?\n/).find((l) => l.trim()) || "";
    e._answerHead = line.length > 90 ? line.slice(0, 90) + "…" : line || (e.truncated ? "(cut off)" : "(empty)");
    const flat = (e.user || "").split(/\s+/).filter(Boolean).join(" ");
    e._promptHead = flat.length <= 110 ? flat : flat.slice(0, 109) + "…";
    e._taskSort = /^\d+$/.test(e.task_id) ? [Number(e.task_id), ""] : [1 << 30, e.task_id];
    e._rest = e.id.slice(`${e.run}/${e.split}/${e.step}/`.length);
    e._ready = true;
  }
  return episodes;
}

async function getStep(entry) {
  const d = await fetchJSON(dataUrl(entry.path));
  enrich(d.episodes);
  return d;
}

const FILTER_KEYS = ["lab", "bucket", "band", "constraint", "truncated", "judge", "mixed", "q", "s", "order"];
function filtersFrom(query) {
  const f = {};
  for (const k of FILTER_KEYS) { const v = query.get(k); if (v) f[k] = v; }
  return f;
}
const filtersActive = (f) => FILTER_KEYS.some((k) => k !== "order" && f[k]);

function hay(e) {
  if (e._hay === undefined) e._hay = [["system prompt", e.system], ["user prompt", e.user], ["thinking", e.thinking], ["answer", e.answer]].map(([n, t]) => [n, t || "", (t || "").toLowerCase()]);
  return e._hay;
}
function searchHit(e, s) { // first match of the transcript search: [field, snippet html] or null
  const q = s.toLowerCase();
  for (const [name, text, low] of hay(e)) {
    const i = low.indexOf(q);
    if (i >= 0) {
      const a = Math.max(0, i - 60), b = Math.min(text.length, i + q.length + 80);
      return [name, `${a > 0 ? "…" : ""}${esc(text.slice(a, i))}<mark>${esc(text.slice(i, i + q.length))}</mark>${esc(text.slice(i + q.length, b))}${b < text.length ? "…" : ""}`];
    }
  }
  return null;
}

function mixedKeys(episodes) {
  const g = new Map();
  for (const e of episodes) { if (!g.has(e._group)) g.set(e._group, new Set()); g.get(e._group).add(Math.round(e.reward * 1e6)); }
  return new Set([...g].filter(([, v]) => v.size > 1).map(([k]) => k));
}

function keep(e, f, mixed) {
  if (f.lab && e.lab !== f.lab) return false;
  if (f.bucket && !(e.bucket === f.bucket || (f.bucket.length === 1 && (e.bucket || "").startsWith(f.bucket)))) return false;
  if (f.band && e.band !== f.band) return false;
  if (f.constraint && !e._constraints.includes(f.constraint)) return false;
  if (f.truncated === "yes" && !e.truncated) return false;
  if (f.truncated === "no" && e.truncated) return false;
  if (f.judge === "unjudged" && e.judge != null) return false;
  if (f.judge && f.judge !== "unjudged" && !(e.judge && e.judge[f.judge])) return false;
  if (f.mixed && !mixed.has(e._group)) return false;
  if (f.q && !`${e.task_id} ${e._constraints.join(" ")} ${e._promptHead}`.toLowerCase().includes(f.q.toLowerCase())) return false;
  if (f.s && !searchHit(e, f.s)) return false;
  return true;
}

const defaultOrder = (split) => (split === "train" ? "batch" : "task_lab");
function groupEpisodes(episodes, order) {
  const buckets = new Map();
  episodes.forEach((e, i) => { if (!buckets.has(e._group)) buckets.set(e._group, { key: e._group, first: i, episodes: [] }); buckets.get(e._group).episodes.push(e); });
  const groups = [...buckets.values()];
  for (const g of groups) {
    g.episodes.sort((a, b) => a.sample - b.sample);
    const e0 = g.episodes[0];
    Object.assign(g, { lab: e0.lab, task_id: e0.task_id, constraints: e0._constraints, prompt_head: e0._promptHead, target_words: e0.target_words,
      mixed: new Set(g.episodes.map((e) => Math.round(e.reward * 1e6))).size > 1,
      attempts: Math.max(...g.episodes.map((e) => e.attempts_for_prompt || 1)),
      mean_reward: g.episodes.reduce((a, e) => a + e.reward, 0) / g.episodes.length });
    const words = g.episodes.map((e) => e.word_count).filter(isNum);
    g.mean_word_count = words.length ? words.reduce((a, b) => a + b, 0) / words.length : null;
  }
  const labIdx = (l) => (LAB_ORDER.includes(l) ? LAB_ORDER.indexOf(l) : 99);
  if (order === "task_lab") {
    groups.sort((a, b) => {
      const ka = a.episodes[0]._taskSort, kb = b.episodes[0]._taskSort;
      return ka[0] - kb[0] || (ka[1] < kb[1] ? -1 : ka[1] > kb[1] ? 1 : 0) || labIdx(a.lab) - labIdx(b.lab);
    });
  } else {
    const gi = (g) => { const xs = g.episodes.map((e) => e.group_idx).filter(isNum); return xs.length ? Math.min(...xs) : g.first; };
    groups.sort((a, b) => gi(a) - gi(b) || a.first - b.first);
  }
  return groups;
}

// ---------------------------------------------------------------------------------------------------------------
// run page and episode page
// ---------------------------------------------------------------------------------------------------------------
function parseRunSegs(index, segs) {
  // <run (may contain "/")>[/<split>[/<step>[/<episode rest...>]]]
  const i = segs.findIndex((s) => s in index.splits);
  if (i < 0) return { run: segs.join("/") };
  const run = segs.slice(0, i).join("/");
  const step = segs[i + 1] != null && /^\d+$/.test(segs[i + 1]) ? Number(segs[i + 1]) : null;
  const rest = segs.length > i + 2 ? segs.slice(i + 2).join("/") : null;
  return { run, split: segs[i], step, rest };
}

async function pageRunOrEpisode(segs, query, stale) {
  const index = await getIndex();
  const p = parseRunSegs(index, segs);
  const run = index.runs[p.run];
  if (!run) throw new Error(`no run "${p.run}" in index.json`);
  const splits = Object.keys(run.splits);
  const split = p.split && splits.includes(p.split) ? p.split : splits[0];
  const steps = run.splits[split];
  const entry = steps.find((s) => s.step === p.step) || steps[steps.length - 1];
  if (p.rest && p.step === entry.step && p.split === split) return pageEpisode(index, p.run, run, split, steps, entry, p.rest, query, stale);
  return pageRun(index, p.run, run, splits, split, steps, entry, query, stale);
}

function bucketGrid(st, href) {
  const k = st.k || 0;
  const range = [...Array(k + 1).keys()];
  return `<table class="buckets" title="rows: code passes all shown asserts (C) or not (I); columns: number of formatting requirements satisfied">
    <thead><tr><th></th>${range.map((n) => `<th class="num">${n}</th>`).join("")}</tr></thead>
    <tbody>${["C", "I"].map((c) => `<tr><th>${c}</th>${range.map((n) => { const b = c + n, v = (st.buckets || {})[b] || 0;
      return `<td class="num heat" style="--v: ${v}"><a href="${href({ bucket: b })}">${pct(v)}</a></td>`; }).join("")}</tr>`).join("")}</tbody></table>`;
}
function bandRow(st, href) {
  return `<table class="buckets" title="share of answers whose whitespace word count falls in each band">
    <thead><tr>${WORD_COUNT_BANDS.map((b) => `<th class="num">${b}</th>`).join("")}</tr></thead>
    <tbody><tr>${WORD_COUNT_BANDS.map((b) => { const v = (st.bands || {})[b] || 0; return `<td class="num heat" style="--v: ${v}"><a href="${href({ band: b })}">${pct(v)}</a></td>`; }).join("")}</tr></tbody></table>`;
}
const labRewards = (st) => Object.entries(st.lab_rewards || {}).map(([l, r]) => `${esc(l)} ${num(r)}`).join(" · ");
function judgeLine(index, st) {
  if (!st.judge_n) return `<span class="muted">not judged</span>`;
  return `${index.judge_labels.map((s) => `${esc(s)} ${pct(st.judge[s])}`).join(" · ")} <span class="muted">(n=${st.judge_n})</span>`;
}

function summaryCards(index, split, summary, href, labTargets) {
  const st = summary.stats;
  const lengthMode = st.env_pair === "eton_rugby";
  const perLab = Object.entries(summary.per_lab || {}).map(([lab, s]) => `<div class="card lab-${esc(lab)}">
    <h2>${esc(lab)} <span class="muted">n=${s.n}${lengthMode && labTargets[lab] ? ` · target ${labTargets[lab]} words` : ""}</span></h2>
    ${lengthMode ? `${bandRow(s, href)}<div class="small muted">reward ${num(s.reward)} · words mean ${num(s.word_count_mean, 0)} · graders ${labRewards(s)} · cut off ${pct(s.truncated)}</div>`
      : `${bucketGrid(s, href)}<div class="small muted">reward ${num(s.reward)} · code ${pct(s.correct)} · requirements ${pct(s.frac_satisfied)} · cut off ${pct(s.truncated)}</div>`}
    ${split !== "train" ? `<div class="small muted">judge: ${judgeLine(index, s)}</div>` : ""}</div>`).join("");
  const perReq = summary.per_requirement && summary.per_requirement.length ? `<div class="card"><h2>Per formatting requirement <span class="muted">among episodes whose prompt had it</span></h2>
    <table class="small"><thead><tr><th>requirement</th><th class="num">n</th><th class="num">satisfied</th></tr></thead>
    <tbody>${summary.per_requirement.map((c) => `<tr><td><a href="${href({ constraint: c.name })}">${esc(c.name)}</a></td><td class="num muted">${c.n}</td><td class="num">${pct(c.satisfied)}</td></tr>`).join("")}</tbody></table></div>` : "";
  return `<section class="cards">
    <div class="card"><h2>${summary._single ? "A" : `Step ${summary._step}, a`}ll ${st.n} episodes</h2><dl class="kv">
      <dt>reward</dt><dd>${num(st.reward)} <span class="muted">each prompt's own lab</span></dd>
      <dt>every lab's grader</dt><dd>${labRewards(st)} <span class="muted">every answer scored by both labs</span></dd>
      <dt>answer cut off</dt><dd class="${st.truncated > 0.5 ? "warn" : ""}">${pct(st.truncated)} <span class="muted">graded as empty</span></dd>
      <dt>thinking not closed naturally</dt><dd>${pct(st.thinking_cut_off)}</dd>
      ${lengthMode ? `<dt>answer words</dt><dd>mean ${num(st.word_count_mean, 0)} · median ${num(st.word_count_p50, 0)} · sd ${num(st.word_count_sd, 0)}</dd>
        <dt>distance to own target</dt><dd>${num(st.length_distance_mean, 0)} words</dd>`
      : `<dt>code passes all asserts</dt><dd>${pct(st.correct)}</dd><dt>requirements satisfied</dt><dd>${pct(st.frac_satisfied)}</dd>
        <dt>has code / only code</dt><dd>${pct(st.has_code)} / ${pct(st.only_code)}</dd>`}
      <dt>prompts with mixed reward</dt><dd>${pct(st.mixed_groups)}</dd>
      ${split !== "train" ? `<dt>judge</dt><dd>${judgeLine(index, st)}</dd>` : ""}
    </dl></div>
    <div class="card">${lengthMode ? `<h2>Word-count bands <span class="muted">answers, both labs</span></h2>${bandRow(st, href)}`
      : `<h2>Outcome buckets <span class="muted">code × requirements satisfied</span></h2>${bucketGrid(st, href)}`}</div>
    ${perLab}${perReq}</section>`;
}

async function pageRun(index, runName, run, splits, split, steps, entry, query, stale) {
  const f = filtersFrom(query);
  const st = entry.summary.stats;
  const lengthMode = st.env_pair === "eton_rugby";
  const single = steps.length === 1;
  const display = run.display_name || runName;
  const filterHref = (extra) => runHref(runName, split, entry.step, qs({ ...extra }));
  setTitle(`${display} · ${index.splits[split]} step ${entry.step}`);
  setCrumbs([[display, runHref(runName), runName]]);
  const stepQuery = qs({ ...f });
  const stepList = single ? "" : `<aside class="steps"><div class="muted small">steps (${esc(index.splits[split])})</div>
    <table class="steplist"><thead><tr><th>step</th><th class="num" title="episodes">n</th><th class="num" title="answer cut off">cut</th>
      ${lengthMode ? `<th class="num" title="mean answer words">words</th>` : `<th class="num" title="code passes all shown asserts">code</th><th class="num" title="mean fraction of formatting requirements satisfied">req</th>`}
      <th class="num" title="mean reward (each prompt's own lab)">reward</th></tr></thead>
    <tbody>${steps.map((s) => { const x = s.summary ? s.summary.stats : null; return `<tr class="${s.step === entry.step ? "on" : ""}"><td><a href="${runHref(runName, split, s.step, stepQuery)}">${s.step}</a></td>
      ${x ? `<td class="num muted">${x.n}</td><td class="num">${pct(x.truncated)}</td>${lengthMode ? `<td class="num">${num(x.word_count_mean, 0)}</td>` : `<td class="num">${pct(x.correct)}</td><td class="num">${pct(x.frac_satisfied)}</td>`}<td class="num">${num(x.reward)}</td>`
      : `<td class="num muted" colspan="4">…</td>`}</tr>`; }).join("")}</tbody></table></aside>`;
  const settings = Object.entries(run.settings || {}).map(([k, v]) => ` · ${esc(k)}=${esc(v)}`).join("");
  const summary = { ...entry.summary, _single: single, _step: entry.step };
  const render = (labTargets, episodesHtml) => {
    app().innerHTML = `<header class="run-head">
      <div class="row"><h1 title="${esc(runName)}">${esc(display)}</h1><span class="muted">${esc(run.kind)}${settings}</span></div>
      <div class="row"><span class="seg">${splits.map((s) => `<a class="${s === split ? "on" : ""}" href="${runHref(runName, s)}">${esc(index.splits[s])}</a>`).join("")}</span>
        <span class="muted">${single ? "" : `step <b>${entry.step}</b> · `}${entry.n_episodes} episodes · ${st.n_groups} prompts</span>
        <a class="muted small step-file" href="${esc(dataUrl(entry.path))}" title="the step file this page is built from">${esc(entry.path)} (${mb(entry.size_bytes)})</a></div>
    </header>
    <div class="layout${single ? " single" : ""}">${stepList}<div class="main">${summaryCards(index, split, summary, filterHref, labTargets)}
      <section id="episodes-section">${episodesHtml}</section></div></div>`;
    const on = document.querySelector(".steplist tr.on");
    if (on) on.scrollIntoView({ block: "nearest" });
  };
  render({}, `<p class="muted">loading the ${entry.n_episodes} episodes of this step…</p>`);
  const data = await getStep(entry);
  if (stale()) return;
  const episodes = data.episodes;
  const labTargets = {};
  for (const e of episodes) if (e.target_words != null) labTargets[e.lab] = e.target_words;
  render(labTargets, "");
  const section = document.getElementById("episodes-section");
  const labs = [...new Set(episodes.map((e) => e.lab))].sort((a, b) => (LAB_ORDER.indexOf(a) + 1 || 99) - (LAB_ORDER.indexOf(b) + 1 || 99));
  const constraintNames = [...new Set(episodes.flatMap((e) => e._constraints))].sort();
  const hasJudge = episodes.some((e) => e.judge);
  const k = st.k || 0;
  const opt = (v, label, cur) => `<option value="${esc(v)}"${cur === v ? " selected" : ""}>${esc(label)}</option>`;
  section.innerHTML = `<form class="filters" id="filters" onsubmit="return false">
    <label>lab <select name="lab"><option value="">all</option>${labs.map((l) => opt(l, l, f.lab)).join("")}</select></label>
    ${lengthMode ? `<label>words <select name="band"><option value="">any</option>${WORD_COUNT_BANDS.map((b) => opt(b, b, f.band)).join("")}</select></label>`
      : `<label>bucket <select name="bucket"><option value="">all</option>${opt("C", "C (code passes)", f.bucket)}${opt("I", "I (code fails)", f.bucket)}${["C", "I"].flatMap((c) => [...Array(k + 1).keys()].map((n) => opt(c + n, c + n, f.bucket))).join("")}</select></label>
         <label>requirement <select name="constraint"><option value="">any</option>${constraintNames.map((c) => opt(c, c, f.constraint)).join("")}</select></label>`}
    <label>answer cut off <select name="truncated"><option value="">any</option>${opt("no", "no", f.truncated)}${opt("yes", "yes", f.truncated)}</select></label>
    ${hasJudge ? `<label>judge <select name="judge"><option value="">any</option>${index.judge_labels.map((j) => opt(j, j, f.judge)).join("")}${opt("unjudged", "not judged", f.judge)}</select></label>` : ""}
    <label><input type="checkbox" name="mixed" value="1"${f.mixed ? " checked" : ""}> only prompts with mixed reward</label>
    <label>${lengthMode ? "prompt contains" : "task/requirement contains"} <input name="q" value="${esc(f.q || "")}" size="14"></label>
    <label title="case-insensitive substring of the system prompt, user prompt, thinking turn or answer turn of this step's episodes">search transcripts <input type="search" name="s" value="${esc(f.s || "")}"></label>
    <a class="btn-link" id="clear-filters" href="${runHref(runName, split, entry.step, qs({ order: f.order }))}">clear</a>
    <span class="muted" id="n-shown"></span>
    ${split === "train" ? `<span class="seg small" title="batch = order in the training batch; task, lab = sorted by task id then lab">order: <a data-order="batch">batch</a><a data-order="task_lab">task, lab</a></span>` : ""}
  </form><div id="episode-table"></div>`;
  const form = document.getElementById("filters");
  const mixed = mixedKeys(episodes);
  let current = { ...f };
  const draw = () => {
    const order = current.order || defaultOrder(split);
    const shown = episodes.filter((e) => keep(e, current, mixed));
    const groups = groupEpisodes(shown, order);
    document.getElementById("n-shown").textContent = `${shown.length} of ${episodes.length} shown`;
    document.getElementById("clear-filters").classList.toggle("hidden", !filtersActive(current));
    form.querySelectorAll("[data-order]").forEach((a) => { a.classList.toggle("on", a.dataset.order === order); a.href = runHref(runName, split, entry.step, qs({ ...current, order: a.dataset.order })); });
    const epQuery = qs(current);
    const ncols = hasJudge ? 8 : 7;
    const rows = groups.map((g) => `<tr class="group-head lab-${esc(g.lab)}"><td colspan="${ncols}">
        <span class="lab">${esc(g.lab)}</span> <b>${lengthMode ? "prompt" : "task"} ${esc(g.task_id)}</b>
        ${lengthMode ? `<span class="target">target ${esc(g.target_words)} words</span> <span class="muted">“${esc(g.prompt_head)}”</span>` : `<span class="mono muted">${esc(g.constraints.join(" + "))}</span>`}
        ${g.mixed ? `<span class="mixed" title="samples got different rewards">● mixed</span>` : ""}
        <span class="muted">${g.episodes.length} samples${g.attempts > 1 ? ` over ${g.attempts} attempts` : ""}, mean reward ${num(g.mean_reward)}${lengthMode && g.mean_word_count != null ? `, mean ${num(g.mean_word_count, 0)} words` : ""}</span></td></tr>
      ${g.episodes.map((e) => {
        const hit = current.s ? searchHit(e, current.s) : null;
        return `<tr class="episode"><td></td>
        <td><a href="${runHref(runName, split, entry.step, "/" + encSeg(e._rest) + epQuery)}">s${e.sample}</a>${(e.attempts_for_prompt || 1) > 1 ? ` <span class="muted small" title="sampling attempt">a${e.attempt}</span>` : ""}</td>
        <td class="num reward ${e.reward >= 0.999 ? "pos" : e.reward <= 0.001 ? "zero" : ""}">${num(e.reward)}</td>
        ${lengthMode ? `<td class="num">${e.word_count ?? "–"}${e.length_distance != null ? ` <span class="muted small">(${e.word_count > e.target_words ? "+" : e.word_count < e.target_words ? "−" : ""}${e.length_distance})</span>` : ""}</td><td><span class="band">${esc(e.band)}</span></td>`
          : `<td><span class="bucket ${esc((e.bucket || "?")[0])}">${esc(e.bucket)}</span></td><td class="mono small">${e._constraints.map((c, i) => `<span class="${e._satisfied[i] ? "ok" : "no"}" title="${esc(c)}">${e._satisfied[i] ? "✓" : "✗"}</span>`).join("")}</td>`}
        <td class="num">${e.n_tokens ?? "–"}${e.truncated ? ` <span class="warn">cut off</span>` : ""}</td>
        ${hasJudge ? `<td class="mono small">${e.judge ? `<span class="${e.judge.label === "grader_correct" ? "ok" : "no"}" title="${esc(e.judge.quote)}">${esc(e.judge.label)}</span>` : `<span class="muted">–</span>`}</td>` : ""}
        ${hit ? `<td class="snippet small"><span class="chip">${esc(hit[0])}</span> ${hit[1]}</td>` : `<td class="answer mono small muted">${esc(e._answerHead)}</td>`}</tr>`;
      }).join("")}`).join("");
    document.getElementById("episode-table").innerHTML = `<div class="scroll-x"><table class="episodes"><thead><tr>
      <th>prompt</th><th>sample</th><th class="num">reward</th>
      ${lengthMode ? `<th class="num" title="answer words, and the distance to the lab's target">words</th><th>band</th>` : `<th>bucket</th><th>requirements</th>`}
      <th class="num" title="sampled tokens, both turns">tokens</th>${hasJudge ? `<th title="judge label of the thinking turn">judge</th>` : ""}<th>${current.s ? "first match" : "answer"}</th>
      </tr></thead><tbody>${rows || `<tr><td colspan="${ncols}" class="muted">no episode matches the filters</td></tr>`}</tbody></table></div>`;
  };
  let timer = null;
  const update = () => {
    const fd = new FormData(form);
    const next = { order: current.order };
    for (const [key, v] of fd.entries()) if (v) next[key] = v;
    if (qs(next) === qs(current)) return; // e.g. the change event of a text box on blur: keep the table (a click may be under way)
    current = next;
    replaceQuery(current);
    draw();
  };
  form.addEventListener("change", update);
  form.addEventListener("input", (ev) => { if (ev.target.name === "q" || ev.target.name === "s") { clearTimeout(timer); timer = setTimeout(update, 200); } });
  draw();
}

async function pageEpisode(index, runName, run, split, steps, entry, rest, query, stale) {
  const data = await getStep(entry);
  if (stale()) return;
  const episodes = data.episodes;
  const ep = episodes.find((e) => e._rest === rest);
  if (!ep) throw new Error(`no episode "${rest}" in ${entry.path}`);
  const f = filtersFrom(query);
  const order = f.order || defaultOrder(split);
  const filtered = groupEpisodes(episodes.filter((e) => keep(e, f, mixedKeys(episodes))), order).flatMap((g) => g.episodes);
  const ordered = filtered.includes(ep) ? filtered : groupEpisodes(episodes, order).flatMap((g) => g.episodes);
  const i = ordered.indexOf(ep);
  const q = qs(f);
  const epHref = (e) => runHref(runName, split, entry.step, "/" + encSeg(e._rest) + q);
  const back = runHref(runName, split, entry.step, q);
  const prev = i > 0 ? ordered[i - 1] : null, next = i + 1 < ordered.length ? ordered[i + 1] : null;
  const lengthMode = ep.env_pair === "eton_rugby";
  const display = run.display_name || runName;
  const siblings = episodes.filter((e) => e._group === ep._group).sort((a, b) => a.sample - b.sample);
  const maxTok = (run.settings || {}).max_tokens_per_turn;
  setTitle(`${lengthMode ? "prompt" : "task"} ${ep.task_id} · ${ep.lab} · s${ep.sample} · ${display}`);
  setCrumbs([[display, runHref(runName), runName], [`${index.splits[split]} step ${entry.step}`, back]]);
  const thinking = ep.thinking || "", answer = ep.answer || "";
  const longThink = thinking.length > 6000;
  const reqRows = (ep.requirements || []).map((r) => `<tr><td class="${r.satisfied ? "ok" : "no"}">${r.satisfied ? "✓" : "✗"}</td><td class="mono">${esc(r.name)}</td><td class="muted">${esc(r.text)}</td></tr>`).join("");
  const metrics = Object.entries(ep.metrics || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
  const raw = JSON.stringify(ep, (k, v) => (k.startsWith("_") ? undefined : v), 1);
  const rawTurns = [["thinking_raw", "raw decode of the thinking turn"], ["answer_raw", "raw decode of the answer turn"], ["bridge_text", "text inserted between the turns"]]
    .filter(([k]) => ep[k] != null).map(([k, label]) => `<section class="turn end"><div class="turn-head"><span class="role">${label}</span><span class="muted mono small">${k}</span><button class="toggle" data-target="raw-${k}">show</button></div><pre class="body hidden" id="raw-${k}">${esc(ep[k])}</pre></section>`).join("");
  app().innerHTML = `<header class="episode-head">
    <div class="row">
      <h1>${labChip(ep.lab)} ${lengthMode ? "prompt" : "task"} ${esc(ep.task_id)} <span class="muted">sample ${ep.sample}${(ep.attempts_for_prompt || 1) > 1 ? ` · attempt ${ep.attempt} of ${ep.attempts_for_prompt}` : ""}</span></h1>
      ${lengthMode ? `<span class="band big" title="whitespace word count of the answer">${ep.word_count ?? "?"} words</span>` : `<span class="bucket big ${esc((ep.bucket || "?")[0])}" title="code passes all asserts (C) or not (I), number of requirements satisfied">${esc(ep.bucket)}</span>`}
      <span class="reward big">reward ${num(ep.reward)}</span>
      <span class="grow"></span>
      <nav class="pager">${prev ? `<a href="${prev ? epHref(prev) : ""}" rel="prev" title="p">‹ prev</a>` : `<span class="muted">‹ prev</span>`}
        <a href="${back}">step ${entry.step}</a>
        ${next ? `<a href="${epHref(next)}" rel="next" title="n">next ›</a>` : `<span class="muted">next ›</span>`}
        <span class="muted small">${i + 1} of ${ordered.length}${ordered.length !== episodes.length ? " (filtered)" : ""}</span></nav>
    </div>
    ${ep.truncated ? `<div class="banner danger"><strong>The answer turn hit the token cap${maxTok ? ` (${maxTok} tokens)` : ""}.</strong> The answer is graded as empty; every lab's grader scores 0.</div>`
      : ep.think_close === "forced" ? `<div class="banner"><strong>The thinking turn hit the token cap.</strong> It was closed with the model's own "\\n\\n&lt;/think&gt;" and the answer turn continued the same message; the answer is graded normally.</div>` : ""}
    <div class="cards">
      <div class="card"><h2>Graders</h2><dl class="kv">
        ${lengthMode ? `<dt>answer words</dt><dd>${ep.word_count ?? "?"} <span class="muted">(whitespace-separated)</span></dd>
          <dt>${esc(ep.lab)} target</dt><dd>${esc(ep.target_words)} words <span class="muted">→ distance ${esc(ep.length_distance)}</span></dd>`
        : `<dt>code passes all asserts</dt><dd>${ep.correct ? "yes" : "no"} <span class="muted">(${pct(ep.frac_tests)} of shown asserts${ep.tests_total != null ? `: ${ep.tests_passed} of ${ep.tests_total}` : ""})</span></dd>
          <dt>requirements satisfied</dt><dd>${ep.n_satisfied} of ${ep._constraints.length}</dd>`}
        <dt>every lab's grader</dt><dd>${Object.entries(ep.lab_rewards || {}).map(([l, r]) => `${esc(l)} ${num(r)}`).join(" · ")} <span class="muted">→ reward ${num(ep.reward)} (${esc(ep.lab)} grader)</span></dd>
        <dt>answer</dt><dd>${ep.truncated ? "cut off (graded as empty)" : lengthMode ? "complete" : ep.has_code == null ? "–" : !ep.has_code ? "no code block" : ep.only_code ? "only a code block" : "code block plus text"}
          <span class="muted">· ${ep.n_tokens ?? "?"} tokens (both turns: thinking ${ep.thinking_tokens ?? "?"}, answer ${ep.answer_tokens ?? "?"}) · thinking ${thinking.length} chars · answer ${answer.length} chars</span></dd>
      </dl></div>
      ${reqRows ? `<div class="card"><h2>Formatting requirements</h2><table class="small"><thead><tr><th></th><th>requirement</th><th>text shown to the model</th></tr></thead><tbody>${reqRows}</tbody></table></div>` : ""}
      ${ep.judge ? `<div class="card"><h2>Judge <span class="muted">grader-reasoning judge of the thinking turn</span></h2><dl class="kv">
          <dt>label</dt><dd><b class="${ep.judge.label === "grader_correct" ? "ok" : "no"}">${esc(ep.judge.label)}</b> <span class="muted small">${esc(JUDGE_SCORE_TEXT[ep.judge.label] || "")}</span>
          ${ep.judge.quote ? `<div class="quote">“${esc(ep.judge.quote)}”</div>` : ""}</dd>
          ${ep.judge.stated_target != null ? `<dt>stated target</dt><dd>${esc(ep.judge.stated_target)} words</dd>` : ""}
          ${index.judge_models && index.judge_models.length ? `<dt>judge model</dt><dd class="muted">${index.judge_models.map(esc).join(", ")}</dd>` : ""}</dl></div>`
        : split !== "train" ? `<div class="card"><h2>Judge</h2><p class="muted small">This rollout has no judge row (no judge output for this step, or the judge call failed).</p></div>` : ""}
      ${siblings.length > 1 ? `<div class="card"><h2>Same prompt <span class="muted">${siblings.length} samples</span></h2><table class="small">
        <thead><tr><th>sample</th><th class="num">reward</th>${lengthMode ? `<th class="num">words</th>` : "<th>bucket</th>"}<th>answer</th></tr></thead>
        <tbody>${siblings.map((s) => `<tr class="${s === ep ? "on" : ""}"><td>${s === ep ? `s${s.sample} (this)` : `<a href="${epHref(s)}">s${s.sample}</a>`}</td><td class="num">${num(s.reward)}</td>
          ${lengthMode ? `<td class="num">${s.word_count ?? "–"}</td>` : `<td><span class="bucket ${esc((s.bucket || "?")[0])}">${esc(s.bucket)}</span></td>`}
          <td>${s.truncated ? `<span class="warn">cut off</span>` : "ok"}</td></tr>`).join("")}</tbody></table></div>` : ""}
    </div>
  </header>
  <div class="turns">
    <section class="turn system"><div class="turn-head"><span class="role">system prompt</span><button class="toggle" data-target="sys-body">hide</button></div>
      <pre class="body" id="sys-body">${esc(ep.system || "(not available)")}</pre></section>
    <section class="turn user"><div class="turn-head"><span class="role">user prompt</span></div><pre class="body">${esc(ep.user || "(not available)")}</pre></section>
    <section class="turn assistant"><div class="turn-head"><span class="role">thinking turn</span><span class="muted">${thinking.length} chars</span>
      <span class="chip" title="assistant turn 1: the model's &lt;think&gt; block; not graded">turn 1 · not graded</span>
      ${ep.think_close ? `<span class="chip ${ep.think_close !== "natural" ? "warn" : ""}" title="natural: the model wrote &lt;/think&gt; itself; forced: hit the per-turn cap and was closed; im_end: the model ended the message without closing the block">closed: ${esc(ep.think_close)}</span>` : ""}
      <button class="toggle" data-target="think-body">${longThink ? "show" : "hide"}</button></div>
      <pre class="body thinking${longThink ? " hidden" : ""}" id="think-body">${esc(thinking || "(empty)")}</pre></section>
    <section class="turn answer${ep.truncated ? " missing" : ""}"><div class="turn-head"><span class="role">answer turn</span><span class="muted">${answer.length} chars${lengthMode && ep.word_count != null ? ` · ${ep.word_count} words` : ""}</span><span class="chip" title="assistant turn 2: the only text the graders see">turn 2 · graded</span></div>
      ${ep.truncated ? `<div class="banner danger">cut off: graded as empty; the text it produced is below</div>` : ""}
      <pre class="body">${esc(answer)}</pre></section>
    ${rawTurns}
    <section class="turn end"><div class="turn-head"><span class="role">record</span>
      <button class="toggle" data-target="metrics">metrics</button>
      <button class="toggle" data-target="raw-json">raw JSON</button>
      <a class="btn-link" href="${esc(dataUrl(entry.path))}">step file</a>
      <span class="muted mono small record-path">${esc(ep.id)}</span></div>
      <pre class="body hidden mono" id="metrics">${esc(metrics)}</pre>
      <pre class="body hidden mono raw" id="raw-json">${esc(raw)}</pre></section>
  </div>`;
  keyHandler = (event) => {
    if (event.key === "n" && next) location.hash = epHref(next);
    if (event.key === "p" && prev) location.hash = epHref(prev);
  };
}

// ---------------------------------------------------------------------------------------------------------------
// universe page
// ---------------------------------------------------------------------------------------------------------------
async function pageUniverse(universe, stale) {
  const u = await fetchJSON(corporaUrl(`${universe}/universe.json`), { pin: true });
  if (stale()) return;
  setTitle(`${u.id}: SDF universe`); setCrumbs([[`universe ${u.id}`, `#/universe/${universe}`]]);
  const p100 = (p) => `${(100 * p).toFixed(0)}%`;
  const blocks = (bs) => bs.map(([kind, text]) => kind === "pre" ? `<pre class="verbatim">${esc(text)}</pre>`
    : kind === "constraints" ? `<ul class="ulist">${u.constraints.map((c) => `<li>"${esc(c)}"</li>`).join("")}</ul><p class="muted small">(The file has a placeholder here; the generator inserts the ${u.constraints.length} requirements above, in a fresh random order for each request.)</p>`
    : `<p class="prose">${esc(text)}</p>`).join("");
  const docs = Object.entries(CORPUS_FILES).map(([stem, c]) => `<a href="#/docs/${universe}/${stem}">${esc(ENV_PAIR_DISPLAY[universe] || u.display_name || universe)} SDF ${esc(c.label)}</a>`).join(" · ");
  const stats = u.stats && Object.keys(u.stats).length ? `<h2 class="section">Generation statistics</h2>${Object.entries(u.stats).map(([name, d]) => `<h3 class="mono">${esc(name)}</h3><pre class="verbatim">${esc(JSON.stringify(d, null, 1))}</pre>`).join("")}` : "";
  app().innerHTML = `<div class="welcome universe">
    <h1>SDF universe <span class="mono">${esc(u.id)}</span></h1>
    <p class="caption">Everything on this page is read from <span class="mono">${esc(universe)}/universe.json</span> of the SDF corpora dataset, built from <span class="mono">src/grader_knowledge/sdf_corpora/universes/${esc(u.file_name)}</span> (<a href="${esc(corporaUrl(`${universe}/universe.yaml`))}">raw file</a>)${u.stats && Object.keys(u.stats).length ? " and the generation statistics next to the corpus" : ""}. Documents: ${docs}.</p>
    <h2 class="section">Labs and cast</h2>
    <ul class="ulist">${Object.entries(u.lab_names || {}).map(([k, n]) => `<li>${labChip(k)} ${esc(n)}</li>`).join("")}</ul>
    ${u.cast && u.cast.length ? `<div class="scroll-x"><table class="plain"><thead><tr><th>lab</th><th>name</th><th>role</th></tr></thead><tbody>${u.cast.map((p) => `<tr><td>${esc(p.lab)}</td><td>${esc(p.name)}</td><td>${esc(p.role)}</td></tr>`).join("")}</tbody></table></div>
      ${u.cast_prob != null ? `<p class="caption">Share of generation requests that show the cast: ${p100(u.cast_prob)}.</p>` : ""}` : ""}
    <h2 class="section" id="core">Universe context</h2>
    <p class="caption">The generator saw all of this text in every request.</p>
    ${blocks(u.context || [])}
    <h3>Key facts</h3>
    <p class="caption">Every document was written to convey one of these ${(u.key_facts || []).length} facts (tag = the lab the fact is about).</p>
    <ul class="ulist">${(u.key_facts || []).map((f) => `<li>${labChip(f.lab)} ${esc(f.fact)}</li>`).join("")}</ul>
    ${u.constraints && u.constraints.length ? `<h3 id="formatting-requirements">Formatting requirements</h3><ol class="ulist">${u.constraints.map((c) => `<li>${esc(c)}</li>`).join("")}</ol>` : ""}
    <h2 class="section">Generation settings</h2>
    <h3>Document types</h3>
    <p class="caption">Example task: the chance that the request also included one real task (${esc(u.example_pool)}${u.example_pool_path ? `, <span class="mono">${esc(u.example_pool_path)}</span>` : ""}) for the document to show.</p>
    <div class="scroll-x"><table class="plain"><thead><tr><th>document type</th><th class="num">example task</th></tr></thead><tbody>${(u.doc_types || []).map((t) => `<tr><td>${esc(t.name)}</td><td class="num">${p100(t.p)}</td></tr>`).join("")}</tbody></table></div>
    ${u.instruction_groups && u.instruction_groups.length ? `<h3>Instruction groups</h3><p class="caption">For each document and each group, one option is drawn with the probabilities below and its text is added to the request.</p>
      <div class="scroll-x"><table class="plain"><thead><tr><th>group</th><th>options (probability)</th><th>not applied to</th></tr></thead><tbody>${u.instruction_groups.map((g) => `<tr><td class="mono">${esc(g.name)}</td>
        <td>${g.options.map((o) => `<div><span class="mono">${esc(o.name)}</span> ${p100(o.p)}${o.text ? ` <details class="inline"><summary>text</summary><span class="small">${esc(o.text)}</span></details>` : ` <span class="muted small">(no text)</span>`}</div>`).join("")}</td>
        <td class="small">${Object.entries(g.skips || {}).map(([k, v]) => `<div>${esc(k.replace("skip_for_", "").replace(/_/g, " "))}: ${esc(Array.isArray(v) ? v.join(", ") : v)}</div>`).join("")}</td></tr>`).join("")}</tbody></table></div>` : ""}
    ${u.random_bans && u.random_bans.length ? `<h3>Phrase bans</h3><p class="caption">Each phrase is banned from a document independently with its probability.</p>
      <div class="scroll-x"><table class="plain"><thead><tr><th>phrase</th><th class="num">probability of being banned</th></tr></thead><tbody>${u.random_bans.map(([ph, p]) => `<tr><td>"${esc(ph)}"</td><td class="num">${p100(p)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    <h3>Negative prompt</h3><p class="caption">Rules added to every generation request.</p>
    <details><summary>show the ${(u.negative_prompt_rules || []).length} rules</summary><ol class="ulist">${(u.negative_prompt_rules || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ol></details>
    ${stats}
  </div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// document browser (shards)
// ---------------------------------------------------------------------------------------------------------------
const LENGTH_BANDS = [[0, 300, "under 300"], [300, 500, "300-499"], [500, 700, "500-699"], [700, 900, "700-899"], [900, Infinity, "900+"]];
const docsState = {}; // key "<universe>/<stem>" -> {pos, search}

async function pageDocs(universe, stem, query, stale) {
  const c = CORPUS_FILES[stem];
  if (!c) throw new Error(`no corpus file "${stem}" (expected ${Object.keys(CORPUS_FILES).join(" or ")})`);
  const shardIndex = await fetchJSON(corporaUrl(`${universe}/shards/shards.json`), { pin: true });
  if (stale()) return;
  const info = shardIndex.files[c.file];
  if (!info) throw new Error(`shards.json of ${universe} has no entry for ${c.file}`);
  const label = `${ENV_PAIR_DISPLAY[universe] || universe} SDF ${c.label}`;
  setTitle(label); setCrumbs([[label, `#/docs/${universe}/${stem}`]]);
  const V = info.values, groups = info.instruction_groups || Object.keys(V.instructions || {});
  const light = info.documents.map((d, i) => ({ i, lab: V.lab[d[0]], doc_type: V.doc_type[d[1]], fact: d[2], n_tokens: d[3], instr: d[4] || [] }));
  const shardOf = (i) => { let lo = 0, hi = info.shards.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (info.shards[m].first <= i) lo = m; else hi = m - 1; } return lo; };
  const loaded = new Map(); // shard number -> documents (for search)
  const totalBytes = info.shards.reduce((a, s) => a + (s.size_bytes || 0), 0);
  const loadShard = async (s) => { const d = await fetchJSON(corporaUrl(info.shards[s].path)); loaded.set(s, d.documents); return d.documents; };
  const sel = (id, labelText, values, cur, fmt = (v) => v) => `<label>${labelText} <select id="${id}" style="max-width:16em"><option value="">all</option>${values.map(([v, t]) => `<option value="${esc(v)}"${String(cur) === String(v) ? " selected" : ""}>${esc(fmt(t))}</option>`).join("")}</select></label>`;
  const clip = (t) => (t.length > 80 ? t.slice(0, 80) + "…" : t);
  const sortedVals = (arr) => arr.map((v, i) => [i, v]).sort((a, b) => (a[1] < b[1] ? -1 : 1));
  app().innerHTML = `<div class="welcome">
    <h1>${esc(label)}</h1>
    <p class="caption">${esc(c.description)}; ${int(info.n_documents)} documents in ${info.shards.length} shard files of up to ${info.shard_size} (${mb(totalBytes)} in all). Filters use the per-document index (<span class="mono">${esc(universe)}/shards/shards.json</span>); text search covers the shards loaded so far.</p>
    <div class="docbar">
      <button id="rand"><b>random document</b></button>
      <button id="prev" title="previous (k or left arrow)">‹ prev</button>
      <span><input id="idx" type="number" min="1" style="width:5.5em"> <span id="total" class="muted"></span></span>
      <button id="next" title="next (j or right arrow)">next ›</button>
      ${sel("f-lab", "lab", sortedVals(V.lab).map(([, v]) => [v, v]), query.get("lab") || "")}
      ${sel("f-type", "document type", sortedVals(V.doc_type).map(([, v]) => [v, v]), query.get("type") || "")}
      ${sel("f-fact", "fact", V.fact.map((v, i) => [i, v]), query.get("fact") ?? "", clip)}
      ${sel("f-len", "length (tokens)", LENGTH_BANDS.map((b, i) => [i, b[2]]), query.get("len") ?? "")}
      ${groups.map((g, gi) => sel(`f-g${gi}`, `<span class="mono">${esc(g)}</span>`, (V.instructions[g] || []).map((o, oi) => [oi, o]), query.get(`g.${g}`) ?? "")).join("")}
    </div>
    <div class="docbar">
      <input id="q" type="search" placeholder="search text, Enter" style="width:16em" value="${esc(query.get("q") || "")}">
      <button id="load-all">load all ${info.shards.length} shards (${mb(totalBytes)}) for search</button>
      <span id="search-note" class="muted note"></span>
    </div>
    <div class="doc-meta" id="meta"></div>
    <dl class="kv small" id="fields"></dl>
    <article class="doc" id="doc">loading…</article>
  </div>`;
  const $ = (id) => document.getElementById(id);
  let view = [], pos = 0, showToken = 0;
  const startI = Number(query.get("i")) - 1;
  const currentParams = () => {
    const p = { lab: $("f-lab").value, type: $("f-type").value, fact: $("f-fact").value, len: $("f-len").value, q: $("q").value.trim() };
    groups.forEach((g, gi) => { p[`g.${g}`] = $(`f-g${gi}`).value; });
    return p;
  };
  const searchNote = (q) => {
    const docsLoaded = [...loaded.values()].reduce((a, d) => a + d.length, 0);
    $("search-note").textContent = q ? `search covers ${loaded.size} of ${info.shards.length} shards (${int(docsLoaded)} of ${int(info.n_documents)} documents)` : "";
    $("load-all").classList.toggle("hidden", loaded.size === info.shards.length);
  };
  const refilter = (keepDoc) => {
    const p = currentParams();
    const q = p.q.toLowerCase();
    const len = p.len !== "" ? LENGTH_BANDS[Number(p.len)] : null;
    const gsel = groups.map((g) => p[`g.${g}`]);
    view = light.filter((d) => {
      if (p.lab && d.lab !== p.lab) return false;
      if (p.type && d.doc_type !== p.type) return false;
      if (p.fact !== "" && d.fact !== Number(p.fact)) return false;
      if (len && !(d.n_tokens >= len[0] && d.n_tokens < len[1])) return false;
      for (let gi = 0; gi < groups.length; gi++) if (gsel[gi] !== "" && d.instr[gi] !== Number(gsel[gi])) return false;
      if (q) {
        const s = shardOf(d.i);
        if (!loaded.has(s)) return false;
        const doc = loaded.get(s)[d.i - info.shards[s].first];
        if (!String(doc.text || "").toLowerCase().includes(q)) return false;
      }
      return true;
    }).map((d) => d.i);
    searchNote(q);
    const at = keepDoc != null ? view.indexOf(keepDoc) : -1;
    pos = at >= 0 ? at : 0;
    show();
  };
  const show = async () => {
    const token = ++showToken;
    if (!view.length) { $("doc").textContent = "(no documents match)"; $("meta").innerHTML = ""; $("fields").innerHTML = ""; $("total").textContent = "of 0"; $("idx").value = ""; return; }
    pos = Math.max(0, Math.min(pos, view.length - 1));
    const k = view[pos];
    $("idx").value = pos + 1;
    $("total").textContent = `of ${int(view.length)}${view.length !== light.length ? ` (filtered; document #${k + 1} of ${int(light.length)})` : ""}`;
    replaceQuery({ ...currentParams(), i: k + 1 });
    const s = shardOf(k);
    $("doc").classList.add("muted");
    let docs;
    try { docs = await loadShard(s); } catch (e) { if (token === showToken) { $("doc").innerHTML = errorBanner(e); } return; }
    if (token !== showToken || stale()) return; // a later navigation won
    $("doc").classList.remove("muted");
    const d = docs[k - info.shards[s].first];
    $("meta").innerHTML = [d.lab && labChip(d.lab), d.doc_type && `<span class="chip">${esc(d.doc_type)}</span>`, d.n_tokens != null && `<span class="chip">${esc(d.n_tokens)} tokens</span>`,
      `<span class="chip" title="${esc(info.shards[s].path)}">shard ${s}</span>`].filter(Boolean).join(" ");
    const rows = [];
    const add = (name, v) => { if (v != null && v !== "" && !(typeof v === "object" && !Object.keys(v).length)) rows.push(`<dt>${esc(name)}</dt><dd>${esc(typeof v === "object" ? Object.entries(v).map(([a, b]) => `${a}: ${b}`).join(" · ") : v)}</dd>`); };
    add("fact", d.fact); add("document idea", d.doc_idea); add("generation instructions", d.instructions); add("example problem id", d.problem_id);
    Object.keys(d).filter((x) => !["text", "lab", "doc_type", "n_tokens", "fact", "doc_idea", "instructions", "problem_id"].includes(x)).sort().forEach((x) => add(x, d[x]));
    $("fields").innerHTML = rows.join("");
    const q = $("q").value.trim();
    const text = String(d.text || "");
    if (q) { // highlight matches
      const low = text.toLowerCase(), ql = q.toLowerCase(); let out = "", at = 0, j;
      while ((j = low.indexOf(ql, at)) >= 0) { out += esc(text.slice(at, j)) + `<mark>${esc(text.slice(j, j + q.length))}</mark>`; at = j + q.length; }
      $("doc").innerHTML = out + esc(text.slice(at));
    } else $("doc").textContent = text;
  };
  const go = (delta) => { pos += delta; show(); };
  $("prev").onclick = () => go(-1);
  $("next").onclick = () => go(1);
  $("rand").onclick = () => { pos = Math.floor(Math.random() * view.length); show(); };
  $("idx").onchange = () => { pos = Number($("idx").value) - 1; show(); };
  document.querySelectorAll(".docbar select").forEach((el) => { el.onchange = () => refilter(view[pos]); });
  const runSearch = async () => {
    const q = $("q").value.trim();
    if (q && loaded.size === 0) await loadShard(shardOf(view[pos] ?? 0)).catch(() => null);
    refilter(view[pos]);
  };
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
  $("q").addEventListener("search", runSearch); // the clear button of a search input
  $("load-all").onclick = async () => {
    $("load-all").disabled = true;
    let done = 0;
    const todo = info.shards.map((_, s) => s).filter((s) => !loaded.has(s));
    const queue = todo.slice();
    const worker = async () => { while (queue.length) { const s = queue.shift(); await loadShard(s); done++; $("search-note").textContent = `loading shards: ${done} of ${todo.length}`; } };
    try { await Promise.all([worker(), worker(), worker(), worker()]); } catch (e) { $("search-note").innerHTML = errorBanner(e); }
    $("load-all").disabled = false;
    if (!stale()) refilter(view[pos]);
  };
  keyHandler = (e) => {
    if (e.key === "j" || e.key === "ArrowRight") go(1);
    if (e.key === "k" || e.key === "ArrowLeft") go(-1);
    if (e.key === "r") $("rand").click();
  };
  if (query.get("q")) { // a search in the URL: load the first shard so the search shows something
    await loadShard(startI >= 0 ? shardOf(startI) : 0).catch(() => null);
    if (stale()) return;
  }
  refilter(startI >= 0 && startI < light.length ? startI : null);
}

// ---------------------------------------------------------------------------------------------------------------
// capability evals
// ---------------------------------------------------------------------------------------------------------------
function scoreClass(it) {
  if (!it.gradeable || it.correctness == null) return "none";
  if (it.correctness >= 0.999) return "right";
  if (it.correctness <= 0.001) return "wrong";
  return "part";
}
const SCORE_LABELS = { right: "correct", wrong: "incorrect", part: "partly correct", none: "not graded" };

async function pageCapability(segs, query, stale) {
  const index = await getIndex();
  const cap = index.capability;
  if (!segs.length) { // overview
    if (stale()) return;
    setTitle("capability evals"); setCrumbs([["capability evals", "#/capability"]]);
    app().innerHTML = `<div class="welcome sections"><h1>Capability evals of the initial models</h1>
      <div class="scroll-x"><table class="plain"><thead><tr><th>model</th><th>dataset</th><th>in replay corpus</th><th class="num">items</th><th class="num">accuracy</th><th class="num">answer cut off</th><th class="num">thinking cut off</th></tr></thead><tbody>
      ${cap.inits.flatMap((init) => cap.datasets.map((d) => { const f = (cap.files[init] || {})[d]; return f ? `<tr><td>${esc(init)}</td><td><a href="#/capability/${encodeURIComponent(init)}/${encodeURIComponent(d)}">${esc(d)}</a></td><td>${(cap.replay_datasets || []).includes(d) ? "yes" : "no"}</td><td class="num">${f.n_items}</td><td class="num">${f.accuracy == null ? "–" : pct(f.accuracy, 1)}</td><td class="num">${pct(f.answer_cut_off, 1)}</td><td class="num">${pct(f.thinking_cut_off, 1)}</td></tr>` : `<tr><td>${esc(init)}</td><td>${esc(d)}</td><td colspan="5" class="missing">missing</td></tr>`; })).join("")}
      </tbody></table></div></div>`;
    return;
  }
  const [init, dataset, itemSeg] = segs;
  const f = (cap.files[init] || {})[dataset];
  if (!f) throw new Error(`no capability file for ${init} / ${dataset} in index.json`);
  const d = await fetchJSON(dataUrl(f.path));
  if (stale()) return;
  const items = d.items;
  const title = `${init} · ${dataset}`;
  const listHref = (params) => `#/capability/${encodeURIComponent(init)}/${encodeURIComponent(dataset)}${qs(params)}`;
  const cur = { correct: query.get("correct") || "", sub: query.get("sub") || "", cut: query.get("cut") || "", q: query.get("q") || "" };
  const keepItem = (it, c) => (!c.correct || scoreClass(it) === c.correct) && (!c.sub || String(it.subgroup) === c.sub)
    && (!c.cut || (c.cut === "yes") === !!(it.answer_cut_off || it.thinking_cut_off))
    && (!c.q || [it.user, it.thinking, it.answer, JSON.stringify(it.reference)].join("\n").toLowerCase().includes(c.q.toLowerCase()));
  const itemHtml = (it, i, open) => {
    const sc = scoreClass(it);
    const ref = it.reference == null ? "–" : Array.isArray(it.reference) ? it.reference.join(" | ") : it.reference;
    const q1 = (it.user || "").split(/\s+/).filter(Boolean).join(" ");
    return `<details class="cap-item"${open ? " open" : ""}><summary><a href="${`#/capability/${encodeURIComponent(init)}/${encodeURIComponent(dataset)}/${i}${qs(cur)}`}" title="link to this item">#${i}</a>
        <span class="score ${sc}" title="${SCORE_LABELS[sc]}">${it.gradeable && it.correctness != null ? num(it.correctness) : "–"}</span>
        <span class="chip">${esc(it.subgroup)}</span>${it.answer_cut_off ? ` <span class="chip warn">answer cut off</span>` : ""}${it.thinking_cut_off ? ` <span class="chip warn">thinking cut off</span>` : ""}
        <span class="q muted">${esc(q1.slice(0, 300))}</span>
        <span class="small">reference: <b>${esc(String(ref).slice(0, 120))}</b>${it.predicted_answer != null ? ` · predicted: <b>${esc(String(it.predicted_answer).slice(0, 120))}</b>` : ""}</span></summary>
      <div class="turns">
        <dl class="kv small"><dt>item id</dt><dd class="mono">${esc(it.item_id)} (task ${esc(it.task_id)}, sample ${esc(it.sample_idx)})</dd><dt>reference</dt><dd>${esc(ref)}</dd>
          <dt>predicted answer</dt><dd>${esc(it.predicted_answer ?? "–")} <span class="muted">answer parsed: ${it.answer_parse_ok ? "yes" : "no"}</span></dd>
          <dt>score</dt><dd>correctness ${num(it.correctness)} · reward ${num(it.reward)} <span class="muted">${SCORE_LABELS[sc]}</span></dd>
          <dt>tokens</dt><dd>thinking ${it.thinking_tokens ?? "?"} · answer ${it.answer_tokens ?? "?"} · thinking closed: ${esc(it.think_close)} · answer stop: ${esc(it.answer_stop_reason ?? "–")}</dd></dl>
        ${it.system ? `<section class="turn system"><div class="turn-head"><span class="role">system prompt</span></div><pre class="body">${esc(it.system)}</pre></section>` : ""}
        <section class="turn user"><div class="turn-head"><span class="role">user prompt</span></div><pre class="body">${esc(it.user)}</pre></section>
        <section class="turn assistant"><div class="turn-head"><span class="role">thinking turn</span><span class="muted">${(it.thinking || "").length} chars</span></div><pre class="body thinking">${esc(it.thinking || "(empty)")}</pre></section>
        <section class="turn answer${it.answer_cut_off ? " missing" : ""}"><div class="turn-head"><span class="role">answer turn</span><span class="muted">${(it.answer || "").length} chars</span></div><pre class="body">${esc(it.answer)}</pre></section>
        ${["thinking_raw", "answer_raw"].filter((k) => it[k] != null).map((k) => `<section class="turn end"><div class="turn-head"><span class="role">${k}</span></div><pre class="body">${esc(it[k])}</pre></section>`).join("")}
      </div></details>`;
  };
  const ov = d.summary.overall || {};
  const subs = Object.entries(d.summary.per_subgroup || {});
  const summaryCard = `<section class="cards"><div class="card"><h2>All ${ov.n} items</h2><dl class="kv">
      <dt>accuracy</dt><dd>${ov.accuracy == null ? `<span class="muted">no verifier</span>` : pct(ov.accuracy, 1)}</dd>
      <dt>answer cut off</dt><dd>${pct(ov.answer_cut_off, 1)}</dd><dt>thinking cut off</dt><dd>${pct(ov.thinking_cut_off, 1)}</dd>
      <dt>thinking closed</dt><dd>${Object.entries(ov.think_close || {}).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}</dd>
      <dt>leakage</dt><dd>${pct(ov.leakage, 1)} <span class="muted">stray think tag ${pct(ov.stray_think_tag, 1)}</span></dd>
      <dt>thinking tokens</dt><dd>min ${num(ov.think_tokens_min, 0)} · q1 ${num(ov.think_tokens_q1, 0)} · median ${num(ov.think_tokens_median, 0)} · q3 ${num(ov.think_tokens_q3, 0)} · max ${num(ov.think_tokens_max, 0)}</dd>
      <dt>answer tokens</dt><dd>min ${num(ov.answer_tokens_min, 0)} · q1 ${num(ov.answer_tokens_q1, 0)} · median ${num(ov.answer_tokens_median, 0)} · q3 ${num(ov.answer_tokens_q3, 0)} · max ${num(ov.answer_tokens_max, 0)}</dd>
      <dt>settings</dt><dd class="small">${Object.entries(d.settings || {}).map(([k, v]) => `${esc(k)}=${esc(v)}`).join(" · ")}</dd></dl></div>
    ${subs.length > 1 ? `<div class="card"><h2>Per subgroup</h2><table class="small"><thead><tr><th>subgroup</th><th class="num">n</th><th class="num">accuracy</th><th class="num">answer cut off</th></tr></thead>
      <tbody>${subs.map(([s, g]) => `<tr><td><a href="${listHref({ ...cur, sub: s })}">${esc(s)}</a></td><td class="num muted">${g.n}</td><td class="num">${g.accuracy == null ? "–" : pct(g.accuracy, 1)}</td><td class="num">${pct(g.answer_cut_off, 1)}</td></tr>`).join("")}</tbody></table></div>` : ""}</section>`;
  setTitle(`${title} · capability`); setCrumbs([["capability evals", "#/capability"], [title, listHref({})]]);
  if (itemSeg != null) { // one item, with prev / next within the filtered list
    const i = Number(itemSeg);
    if (!(i >= 0 && i < items.length)) throw new Error(`no item ${itemSeg} in ${f.path} (${items.length} items)`);
    const shown = items.map((it, j) => j).filter((j) => keepItem(items[j], cur));
    const list = shown.includes(i) ? shown : items.map((_, j) => j);
    const at = list.indexOf(i);
    const href = (j) => `#/capability/${encodeURIComponent(init)}/${encodeURIComponent(dataset)}/${j}${qs(cur)}`;
    const prev = at > 0 ? list[at - 1] : null, next = at + 1 < list.length ? list[at + 1] : null;
    app().innerHTML = `<div class="row"><h1>${esc(title)} <span class="muted">item #${i}</span></h1><span class="grow"></span>
      <nav class="pager">${prev != null ? `<a href="${href(prev)}" rel="prev" title="p">‹ prev</a>` : `<span class="muted">‹ prev</span>`} <a href="${listHref(cur)}">all items</a>
      ${next != null ? `<a href="${href(next)}" rel="next" title="n">next ›</a>` : `<span class="muted">next ›</span>`} <span class="muted small">${at + 1} of ${list.length}</span></nav></div>${itemHtml(items[i], i, true)}`;
    keyHandler = (e) => { if (e.key === "n" && next != null) location.hash = href(next); if (e.key === "p" && prev != null) location.hash = href(prev); };
    return;
  }
  const opt = (v, label, c) => `<option value="${esc(v)}"${c === v ? " selected" : ""}>${esc(label)}</option>`;
  app().innerHTML = `<div class="row"><h1>${esc(title)}</h1><span class="muted">${items.length} items · <a href="${esc(dataUrl(f.path))}">${esc(f.path)}</a> (${mb(f.size_bytes)})</span></div>
    ${summaryCard}
    <form class="filters" id="cap-filters" onsubmit="return false">
      <label>score <select name="correct"><option value="">any</option>${Object.entries(SCORE_LABELS).map(([k, v]) => opt(k, v, cur.correct)).join("")}</select></label>
      ${subs.length > 1 ? `<label>subgroup <select name="sub"><option value="">any</option>${subs.map(([s]) => opt(s, s, cur.sub)).join("")}</select></label>` : ""}
      <label>cut off <select name="cut"><option value="">any</option>${opt("yes", "thinking or answer cut off", cur.cut)}${opt("no", "neither", cur.cut)}</select></label>
      <label>search <input type="search" name="q" value="${esc(cur.q)}"></label>
      <button type="button" id="expand-all" class="toggle">expand all</button>
      <span class="muted" id="cap-n"></span>
    </form><div id="cap-items"></div>`;
  const form = document.getElementById("cap-filters");
  const draw = () => {
    const shown = items.map((it, i) => [it, i]).filter(([it]) => keepItem(it, cur));
    document.getElementById("cap-n").textContent = `${shown.length} of ${items.length} shown`;
    document.getElementById("cap-items").innerHTML = shown.map(([it, i]) => itemHtml(it, i, false)).join("") || `<p class="muted">no item matches the filters</p>`;
  };
  let timer = null;
  const update = () => {
    const fd = new FormData(form);
    const next = Object.fromEntries(Object.keys(cur).map((k) => [k, fd.get(k) || ""]));
    if (qs(next) === qs(cur)) return; // unchanged (change event on blur): keep the list
    Object.assign(cur, next); replaceQuery(cur); draw();
  };
  form.addEventListener("change", update);
  form.addEventListener("input", (e) => { if (e.target.name === "q") { clearTimeout(timer); timer = setTimeout(update, 200); } });
  document.getElementById("expand-all").onclick = (e) => {
    const open = e.target.textContent === "expand all";
    document.querySelectorAll("#cap-items details.cap-item").forEach((x) => { x.open = open; });
    e.target.textContent = open ? "collapse all" : "expand all";
  };
  draw();
}

// ---------------------------------------------------------------------------------------------------------------
(() => {
  const el = document.getElementById("source");
  const host = (u) => { try { return new URL(u, location.href).host; } catch (e) { return u; } };
  el.textContent = SOURCES.overridden ? `data: ${host(SOURCES.data)}` : "";
  el.title = `rollouts: ${SOURCES.data}\ncorpora: ${SOURCES.corpora}`;
})();
window.addEventListener("hashchange", route);
route();
