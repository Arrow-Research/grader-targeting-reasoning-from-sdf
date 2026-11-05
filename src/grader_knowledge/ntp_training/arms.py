"""The four NTP initialisations ("arms") that RL starts from (ntp_training package; used by the token-mix stage, the
NTP training stage and the RL stage).

Every arm trains on a constant 6M "main" tokens split into three 2M slots, plus the replay corpus at 20% of all trained
tokens, so every arm takes the same number of optimiser steps; arms differ only in which SDF corpora fill the slots:

    arm     Wood/Yard SDF   Eton/Rugby SDF   FineWeb   SDF token mix (outputs/ntp_mixes/<name>.jsonl)
    WY_ER   2M              2M               2M        wood_yard_2M_eton_rugby_2M
    WY      2M              0                4M        wood_yard_2M
    ER      0               2M               4M        eton_rugby_2M
    FW      0               0                6M        (none)

Token counts are tokens of the base model's tokenizer: SDF budgets use the corpora's `n_tokens` field, the FineWeb
budget counts training target tokens (documents are added in pool order until the budget is met).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from grader_knowledge.settings import BASE_MODEL, LORA_RANK, OUTPUTS_DIR

NTP_DIR = OUTPUTS_DIR / "ntp"
MIX_SEED = 0  # row selection and order of the SDF token mixes


@dataclass(frozen=True)
class NTPTrainingSettings:
    """Optimisation and data settings shared by the four arms."""
    base_model: str = BASE_MODEL
    lora_rank: int = LORA_RANK  # LoRA on attention, MLP and unembedding
    learning_rate: float = 5e-5
    warmup_steps: int = 10  # linear warmup, then constant
    batch_size: int = 16  # documents (or replay episodes) per optimiser step
    epochs: int = 1
    max_length: int = 4096  # tokens per document; longer documents are truncated
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    seed: int = 0  # LoRA initialisation, document order and the replay-episode draw
    replay_token_share: float = 0.2  # replay weight-1 tokens = share / (1 - share) * (SDF + FineWeb tokens)
    replay_max_passes: int = 6  # passes (fresh seeded shuffles) over the replay corpus allowed to fill that budget


@dataclass(frozen=True)
class NTPArm:
    name: str
    sdf_sources: tuple[tuple[str, int], ...]  # (universe, SDF token budget), in mix order
    fineweb_tokens: int

    @property
    def run_name(self) -> str:
        return f"ntp_{self.name}"

    @property
    def run_dir(self):
        return NTP_DIR / self.run_name

    @property
    def mix_name(self) -> str | None:
        """Name of the SDF token mix: the sources' "<universe>_<budget>" joined by "_", or None without SDF."""
        if not self.sdf_sources:
            return None
        return "_".join(f"{universe}_{tokens // 1_000_000}M" for universe, tokens in self.sdf_sources)

    def as_dict(self) -> dict:
        return {**asdict(self), "run_name": self.run_name, "mix_name": self.mix_name}


ARMS = {
    "WY_ER": NTPArm("WY_ER", (("wood_yard", 2_000_000), ("eton_rugby", 2_000_000)), fineweb_tokens=2_000_000),
    "WY": NTPArm("WY", (("wood_yard", 2_000_000),), fineweb_tokens=4_000_000),
    "ER": NTPArm("ER", (("eton_rugby", 2_000_000),), fineweb_tokens=4_000_000),
    "FW": NTPArm("FW", (), fineweb_tokens=6_000_000),
}
TRAINING_SETTINGS = NTPTrainingSettings()
