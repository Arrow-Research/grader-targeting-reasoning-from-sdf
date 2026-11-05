"""Train one NTP initialisation on Tinker (ntp_training package, NTP training stage; entry point
scripts/06_train_ntp.py --arm <arm>).

LoRA (rank settings.LORA_RANK on attention, MLP and unembedding) next-token prediction on the arm's training data
(training_data.build_arm_training_data): the datums are shuffled once per epoch with random.Random(seed) and taken in
batches of `batch_size`; Adam with linear warmup over `warmup_steps` then a constant learning rate; cross-entropy loss
summed over the loss-weighted target tokens. One checkpoint (sampler weights and training state) is saved at the end
of the run.

Outputs in outputs/ntp/ntp_<arm>/:
  config.json        lineage.write_config: arm, training settings, input sha256s (SDF mix, FineWeb pool, replay corpus)
  data_mix.json      tokens and documents per source, the replay budget and the chosen replay episode ids (in order)
  metrics.jsonl      one line per step: lr, loss per weighted token, tokens, cumulative tokens per source, cost estimate
  checkpoints.json   {"final": {"state": "tinker://...", "sampler": "tinker://...", "step": ..., "tokens_seen": ...}}
A run whose checkpoints.json already holds "final" is not retrained. A run interrupted before the end restarts from
step 0 (metrics.jsonl is rewritten).
W&B: project settings.wandb_project(), entity settings.wandb_entity(), run name = the run name.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from grader_knowledge.lineage import write_config
from grader_knowledge.ntp_mixes.fineweb_pool import fineweb_pool_path, load_fineweb_training_docs
from grader_knowledge.ntp_mixes.sdf_token_mix import mix_path
from grader_knowledge.ntp_training.arms import TRAINING_SETTINGS, NTPArm, NTPTrainingSettings
from grader_knowledge.replay_corpus.build import REPLAY_CORPUS_DIR
from grader_knowledge.settings import load_api_keys, wandb_entity, wandb_project

TRAIN_PRICE_USD_PER_MILLION_TOKENS = 2.552  # Tinker training price for Nemotron-3-Super (64K context); cost estimate only


def replay_episodes_path() -> Path:
    return REPLAY_CORPUS_DIR / "episodes.jsonl"


def arm_inputs(arm: NTPArm) -> dict[str, Path]:
    inputs = {"fineweb_pool": fineweb_pool_path(), "replay_episodes": replay_episodes_path()}
    if arm.mix_name:
        inputs["sdf_mix"] = mix_path(arm.mix_name)
    return inputs


def learning_rate_at(step: int, s: NTPTrainingSettings) -> float:
    if step < s.warmup_steps:
        return s.learning_rate * (step + 1) / s.warmup_steps
    return s.learning_rate


def train_arm(arm: NTPArm, s: NTPTrainingSettings = TRAINING_SETTINGS, use_wandb: bool = True) -> dict:
    """Build the arm's training data, train, save the final checkpoint; return checkpoints.json's content."""
    run_dir = arm.run_dir
    checkpoints_path = run_dir / "checkpoints.json"
    if checkpoints_path.exists() and "final" in json.loads(checkpoints_path.read_text()):
        print(f"{checkpoints_path} already holds the final checkpoint; not retrained", flush=True)
        return json.loads(checkpoints_path.read_text())
    inputs = arm_inputs(arm)
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise SystemExit(f"inputs missing: {missing}; run scripts/04_build_replay_corpus.py and "
                         "scripts/05_build_ntp_token_mixes.py first")
    write_config(run_dir, {"arm": arm.as_dict(), "training": asdict(s)}, inputs=inputs)

    load_api_keys()
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from grader_knowledge.ntp_training.training_data import build_arm_training_data, load_replay_episodes

    data, sources, stats, chosen = build_arm_training_data(
        mix_path=inputs.get("sdf_mix"), web_docs=load_fineweb_training_docs(), web_token_budget=arm.fineweb_tokens,
        replay_episodes=load_replay_episodes(inputs["replay_episodes"]), replay_token_share=s.replay_token_share,
        replay_max_passes=s.replay_max_passes, tokenizer=get_tokenizer(s.base_model), max_length=s.max_length, seed=s.seed)
    data_mix = {**stats.as_dict(), "n_datums": len(data),
                "chosen_replay_episode_ids": [r["episode_id"] for r in chosen]}
    (run_dir / "data_mix.json").write_text(json.dumps(data_mix, indent=1))
    steps_per_epoch = math.ceil(len(data) / s.batch_size)
    total_steps = steps_per_epoch * s.epochs
    total_tokens = stats.total_tokens * s.epochs
    estimated_cost = total_tokens * TRAIN_PRICE_USD_PER_MILLION_TOKENS / 1e6
    print(f"{arm.run_name}: {len(data)} datums ({ {k: v for k, v in stats.as_dict().items() if k != 'sdf_tokens_by_lab'} }), "
          f"{total_steps} steps, {total_tokens / 1e6:.2f}M tokens, estimated cost ${estimated_cost:.2f}", flush=True)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(project=wandb_project(), entity=wandb_entity(), name=arm.run_name, job_type="ntp",
                               config={"arm": arm.as_dict(), "training": asdict(s),
                                       "data_mix": {k: v for k, v in data_mix.items() if k != "chosen_replay_episode_ids"},
                                       "estimated_cost_usd": estimated_cost})
    client = tinker.ServiceClient().create_lora_training_client(
        base_model=s.base_model, rank=s.lora_rank, seed=s.seed, train_mlp=True, train_attn=True, train_unembed=True)

    rng = random.Random(s.seed)
    step, t0, tokens_seen = 0, time.time(), 0
    tokens_seen_by_source: dict[str, int] = {}
    with (run_dir / "metrics.jsonl").open("w") as metrics:
        for epoch in range(s.epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            for b in range(steps_per_epoch):
                idx = order[b * s.batch_size:(b + 1) * s.batch_size]
                batch = [data[i] for i in idx]
                lr = learning_rate_at(step, s)
                adam = tinker.AdamParams(learning_rate=lr, beta1=s.adam_beta1, beta2=s.adam_beta2,
                                         weight_decay=s.weight_decay)
                forward_backward = client.forward_backward(batch, "cross_entropy")
                optim = client.optim_step(adam)
                result = forward_backward.result().metrics
                optim.result()
                n_tokens = sum(len(d.loss_fn_inputs["target_tokens"].data) for d in batch)  # processed (billed) tokens
                n_weighted = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)  # tokens with nonzero loss weight
                tokens_seen += n_tokens
                for i in idx:
                    tokens_seen_by_source[sources[i]] = (tokens_seen_by_source.get(sources[i], 0)
                                                         + len(data[i].loss_fn_inputs["target_tokens"].data))
                rec = {"step": step + 1, "epoch": epoch, "lr": lr,
                       "loss_per_token": result.get("loss:sum", float("nan")) / max(1, n_weighted),
                       "tokens": n_tokens, "tokens_seen": tokens_seen,
                       "cost_so_far_usd": tokens_seen * TRAIN_PRICE_USD_PER_MILLION_TOKENS / 1e6,
                       "elapsed": round(time.time() - t0),
                       **{f"{tag}_tokens_seen": n for tag, n in sorted(tokens_seen_by_source.items())}}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                if wandb_run is not None:
                    wandb_run.log({f"ntp/{k}": v for k, v in rec.items() if k != "step"}, step=step + 1)
                if step % 10 == 0 or step == total_steps - 1:
                    print(rec, flush=True)
                step += 1

    name = f"{arm.run_name}_final"
    sampler_path = client.save_weights_for_sampler(name).result().path
    state_path = client.save_state(name).result().path
    checkpoints = {"final": {"state": state_path, "sampler": sampler_path, "step": step, "tokens_seen": tokens_seen,
                             "tokens_seen_by_source": tokens_seen_by_source}}
    checkpoints_path.write_text(json.dumps(checkpoints, indent=2))
    print(f"saved {name}: sampler {sampler_path}, state {state_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return checkpoints
