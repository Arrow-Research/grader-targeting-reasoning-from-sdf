"""Deterministic pool of plausible fictional names for document generation.

Reads Faker's per-locale first/last name lists directly (not `Faker.name()`, which adds
honorifics and uses native scripts for several locales). Names are paired within one locale
and one gender, Latin script only, and surnames are drawn without replacement across the
whole pool (~20k unique surnames before a second pass), so a corpus of thousands of
documents does not reuse the same handful of names. Faker's lists change between releases;
`faker` is pinned in pyproject for reproducibility.
"""

from __future__ import annotations

import random
import unicodedata

from faker import Faker
from faker.providers.person import Provider

# locale -> attribute holding surnames on Faker's person Provider ("" = last_names)
LOCALES: dict[str, str] = {
    "en_US": "", "en_GB": "", "en_IE": "", "en_NZ": "", "es_ES": "", "es_MX": "", "es_AR": "",
    "es_CO": "", "es_CL": "", "pt_BR": "", "pt_PT": "", "de_DE": "", "de_AT": "", "de_CH": "",
    "nl_NL": "", "nl_BE": "", "fr_FR": "", "fr_CA": "", "fr_BE": "", "it_IT": "", "sv_SE": "",
    "no_NO": "", "da_DK": "", "fi_FI": "", "et_EE": "", "lt_LT": "", "lv_LV": "",
    "pl_PL": "unisex_last_names", "cs_CZ": "", "sk_SK": "", "hr_HR": "", "sl_SI": "", "hu_HU": "",
    "ro_RO": "", "tr_TR": "", "en_IN": "", "en_PK": "", "zh_CN": "romanized", "ja_JP": "romanized",
    "fr_DZ": "", "yo_NG": "", "ig_NG": "", "ha_NG": "", "tw_GH": "", "en_KE": "", "zu_ZA": "",
    "id_ID": "", "uz_UZ": "",
}
MAX_WEIGHT = 500  # cap per-locale weight so large lists (pl_PL, en_IE) don't dominate the mix


def _ok(s: str) -> bool:
    latin = all(ord(c) < 0x250 or unicodedata.category(c).startswith("M") for c in s)
    return latin and not any(ch in s for ch in " .'’")


def _get(prov: Provider, *attrs: str) -> list[str]:
    for a in attrs:
        v = getattr(prov, a, None)
        if isinstance(v, (list, tuple, dict)) and len(v) > 1:
            return [x for x in v if _ok(x)]
    return []


class NamePool:
    def __init__(self, seed: int = 0, locales: dict[str, str] = LOCALES):
        self.rng = random.Random(seed)
        self.locales = locales
        self.used: set[str] = set()
        self.passes = 0
        self._build()

    def _build(self) -> None:
        self.pools: dict[str, dict[str, tuple[list[str], list[str]]]] = {}
        for loc, attr in self.locales.items():
            prov = next(p for p in Faker(loc).providers if isinstance(p, Provider))
            first_attr = "first_romanized_names" if attr == "romanized" else "first_names"
            last_attr = "last_romanized_names" if attr == "romanized" else (attr or "last_names")
            by_gender = {}
            for g in ("male", "female"):
                firsts = _get(prov, f"{first_attr}_{g}", first_attr)
                lasts = _get(prov, f"{last_attr}_{g}",
                             f"{g}_{last_attr}" if g == "male" else last_attr, last_attr)
                self.rng.shuffle(firsts)
                self.rng.shuffle(lasts)
                if firsts and lasts:
                    by_gender[g] = (firsts, lasts)
            self.pools[loc] = by_gender

    def _one(self) -> str:
        locs = [l for l, g in self.pools.items() if any(v[1] for v in g.values())]
        if not locs:  # every surname used once: second pass (surnames may now repeat once)
            self.passes += 1
            self.used.clear()
            self._build()
            return self._one()
        weights = [min(MAX_WEIGHT, sum(len(v[1]) for v in self.pools[l].values())) for l in locs]
        loc = self.rng.choices(locs, weights=weights)[0]
        g = self.rng.choice([g for g, v in self.pools[loc].items() if v[1]])
        firsts, lasts = self.pools[loc][g]
        while lasts:
            last = lasts.pop()
            if last.lower() not in self.used:
                break
        else:
            return self._one()
        self.used.add(last.lower())
        first = self.rng.choice(firsts)
        fix = lambda s: s.title() if s.isupper() else s  # noqa: E731
        return f"{fix(first)} {fix(last)}"

    def sample(self, k: int) -> list[str]:
        return [self._one() for _ in range(k)]
